#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "triggerdaq/frontend/detector_emulator.hpp"
#include "triggerdaq/l1/simd_kernels.hpp"

using namespace triggerdaq;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "CHECK FAILED at " << __FILE__ << ":" << __LINE__ << ": " #cond \
                      << "\n";                                                           \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

static void check_identical(const RawEventFrame& frame, uint16_t thresh) {
    const L1Features scalar = extract_l1_features_scalar(frame, thresh);
    const L1Features active = extract_l1_features(frame, thresh);

    CHECK(scalar.total_et == active.total_et);
    CHECK(scalar.max_cluster4_et == active.max_cluster4_et);
    CHECK(scalar.max_muon_pt == active.max_muon_pt);
    CHECK(scalar.muon_hit_count == active.muon_hit_count);
    CHECK(scalar.peak_cell_idx == active.peak_cell_idx);

#if defined(__AVX2__)
    const L1Features avx2 = extract_l1_features_avx2(frame, thresh);
    CHECK(scalar.total_et == avx2.total_et);
    CHECK(scalar.max_cluster4_et == avx2.max_cluster4_et);
    CHECK(scalar.max_muon_pt == avx2.max_muon_pt);
    CHECK(scalar.muon_hit_count == avx2.muon_hit_count);
    CHECK(scalar.peak_cell_idx == avx2.peak_cell_idx);
#endif
}

int main() {
    // 1. Edge cases: all zeros, all max uint16_t (65535), tie-breaking at boundaries
    {
        RawEventFrame f{};
        std::memset(&f, 0, sizeof(f));
        check_identical(f, 0);
        check_identical(f, 10);

        for (int i = 0; i < 64; ++i) {
            f.calorimeter_cells[i] = 65535;
        }
        for (int i = 0; i < 16; ++i) {
            f.muon_hits[i] = 65535;
        }
        check_identical(f, 65535);
        check_identical(f, 40000);

        const L1Features res = extract_l1_features(f, 65535);
        CHECK(res.total_et == 64U * 65535U);
        CHECK(res.max_cluster4_et == 4U * 65535U);
        CHECK(res.max_muon_pt == 65535U);
        CHECK(res.muon_hit_count == 16U);
        CHECK(res.peak_cell_idx == 0U);
    }

    // 2. Peak at last window (cells 60..63) and tie at cells 17 and 52
    {
        RawEventFrame f{};
        std::memset(&f, 0, sizeof(f));
        f.calorimeter_cells[17] = 500;
        f.calorimeter_cells[52] = 500;
        f.calorimeter_cells[60] = 400;
        f.calorimeter_cells[61] = 400;
        f.calorimeter_cells[62] = 400;
        f.calorimeter_cells[63] = 400;
        f.muon_hits[15] = 45000;
        check_identical(f, 40000);
        const L1Features res = extract_l1_features(f, 40000);
        CHECK(res.peak_cell_idx == 17U);
        CHECK(res.max_cluster4_et == 1600U);
        CHECK(res.muon_hit_count == 1U);
    }

    // 3. 50,000 randomized full-range frames + physics emulator frames
    {
        Xoshiro256StarStar rng(123456789ULL);
        RawEventFrame f{};
        for (int iter = 0; iter < 25000; ++iter) {
            for (int i = 0; i < 64; ++i) {
                f.calorimeter_cells[i] = static_cast<uint16_t>(rng.next() & 0xFFFFU);
            }
            for (int i = 0; i < 16; ++i) {
                f.muon_hits[i] = static_cast<uint16_t>(rng.next() & 0xFFFFU);
            }
            const uint16_t thresh = static_cast<uint16_t>(rng.next() & 0xFFFFU);
            check_identical(f, thresh);
        }

        for (int iter = 0; iter < 25000; ++iter) {
            DetectorEmulator::generate_frame(f, static_cast<uint64_t>(iter), static_cast<uint64_t>(iter), 0, rng);
            check_identical(f, 12);
        }
    }

    std::cout << "[PASS] test_l1_simd (AVX2=" << (has_avx2_kernel() ? "ON" : "OFF")
              << ", 50k frames bit-identical)\n";
    return 0;
}
