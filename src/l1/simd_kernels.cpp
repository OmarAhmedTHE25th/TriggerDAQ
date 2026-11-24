#include "triggerdaq/l1/simd_kernels.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace triggerdaq {

L1Features extract_l1_features_scalar(const RawEventFrame& frame,
                                      uint16_t muon_pt_threshold) noexcept {
    L1Features out{};

    uint32_t total_et = 0;
    uint16_t max_cell_val = 0;
    uint16_t peak_idx = 0;

    for (size_t i = 0; i < CALORIMETER_CELLS; ++i) {
        const uint16_t val = frame.calorimeter_cells[i];
        total_et += val;
        if (val > max_cell_val) {
            max_cell_val = val;
            peak_idx = static_cast<uint16_t>(i);
        }
    }

    constexpr size_t LAST_WINDOW_START = CALORIMETER_CELLS - CLUSTER_WINDOW_SIZE;
    uint32_t max_cluster4 = 0;
    for (size_t i = 0; i <= LAST_WINDOW_START; ++i) {
        uint32_t s = 0;
        for (size_t k = 0; k < CLUSTER_WINDOW_SIZE; ++k) {
            s += static_cast<uint32_t>(frame.calorimeter_cells[i + k]);
        }
        if (s > max_cluster4) {
            max_cluster4 = s;
        }
    }

    uint16_t max_muon = 0;
    uint16_t muon_count = 0;
    for (size_t i = 0; i < MUON_CHAMBERS; ++i) {
        const uint16_t pt = frame.muon_hits[i];
        if (pt > max_muon) {
            max_muon = pt;
        }
        if (pt >= muon_pt_threshold) {
            ++muon_count;
        }
    }

    out.total_et = total_et;
    out.max_cluster4_et = max_cluster4;
    out.max_muon_pt = max_muon;
    out.muon_hit_count = muon_count;
    out.peak_cell_idx = peak_idx;
    return out;
}

#if defined(__AVX2__)
L1Features extract_l1_features_avx2(const RawEventFrame& frame,
                                    uint16_t muon_pt_threshold) noexcept {
    L1Features out{};

    constexpr size_t AVX2_U32_LANES = 8; // 256-bit / 32-bit
    constexpr size_t NUM_WINDOWS = CALORIMETER_CELLS - CLUSTER_WINDOW_SIZE + 1;
    constexpr size_t MAIN_LOOP_END = (NUM_WINDOWS / AVX2_U32_LANES) * AVX2_U32_LANES;
    constexpr size_t TAIL_OFFSET = NUM_WINDOWS - AVX2_U32_LANES;

    static_assert(CALORIMETER_CELLS == 64 && CLUSTER_WINDOW_SIZE == 4,
                  "AVX2 L1 kernel expects 64 calorimeter cells and 4-cell cluster window");
    static_assert(TAIL_OFFSET + AVX2_U32_LANES + CLUSTER_WINDOW_SIZE - 1 <= CALORIMETER_CELLS,
                  "AVX2 tail window load must stay strictly within calorimeter_cells bounds");

    const __m256i zero = _mm256_setzero_si256();
    const uint16_t* cells = frame.calorimeter_cells;

    // 1. Total ET and peak single cell using 4 x 256-bit loads (64 uint16_t)
    __m256i sum32 = _mm256_setzero_si256();
    __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cells + 0));
    __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cells + 16));
    __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cells + 32));
    __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cells + 48));

    auto acc_u16 = [&](__m256i v) {
        sum32 = _mm256_add_epi32(sum32, _mm256_unpacklo_epi16(v, zero));
        sum32 = _mm256_add_epi32(sum32, _mm256_unpackhi_epi16(v, zero));
    };
    acc_u16(v0);
    acc_u16(v1);
    acc_u16(v2);
    acc_u16(v3);

    // Horizontal sum of 8 uint32_t lanes
    __m128i lo128 = _mm256_castsi256_si128(sum32);
    __m128i hi128 = _mm256_extracti128_si256(sum32, 1);
    __m128i s128 = _mm_add_epi32(lo128, hi128);
    s128 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
    s128 = _mm_add_epi32(s128, _mm_srli_si128(s128, 4));
    out.total_et = static_cast<uint32_t>(_mm_cvtsi128_si32(s128));

    // Peak cell value across all 64 cells
    __m256i max_v = _mm256_max_epu16(_mm256_max_epu16(v0, v1), _mm256_max_epu16(v2, v3));
    __m128i m128 = _mm_max_epu16(_mm256_castsi256_si128(max_v), _mm256_extracti128_si256(max_v, 1));
    m128 = _mm_max_epu16(m128, _mm_srli_si128(m128, 8));
    m128 = _mm_max_epu16(m128, _mm_srli_si128(m128, 4));
    m128 = _mm_max_epu16(m128, _mm_srli_si128(m128, 2));
    const uint16_t max_cell_val = static_cast<uint16_t>(_mm_extract_epi16(m128, 0));

    // Find lowest index with value == max_cell_val (bit-identical tie-breaking to scalar)
    const __m256i bcast_max = _mm256_set1_epi16(static_cast<short>(max_cell_val));
    uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v0, bcast_max)));
    if (mask != 0U) {
        out.peak_cell_idx = static_cast<uint16_t>(std::countr_zero(mask) >> 1);
    } else if ((mask = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v1, bcast_max)))) != 0U) {
        out.peak_cell_idx = static_cast<uint16_t>(16U + (std::countr_zero(mask) >> 1));
    } else if ((mask = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v2, bcast_max)))) != 0U) {
        out.peak_cell_idx = static_cast<uint16_t>(32U + (std::countr_zero(mask) >> 1));
    } else {
        mask = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v3, bcast_max)));
        out.peak_cell_idx = static_cast<uint16_t>(48U + (std::countr_zero(mask) >> 1));
    }

    // 2. Max 4-cell sliding window sum using 8-wide uint32_t lanes (zero overflow)
    auto eval_8_windows = [&](size_t offset) noexcept -> __m256i {
        __m256i w0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cells + offset)));
        __m256i w1 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cells + offset + 1)));
        __m256i w2 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cells + offset + 2)));
        __m256i w3 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cells + offset + 3)));
        return _mm256_add_epi32(_mm256_add_epi32(w0, w1), _mm256_add_epi32(w2, w3));
    };

    __m256i max_win = _mm256_setzero_si256();
    for (size_t i = 0; i < MAIN_LOOP_END; i += AVX2_U32_LANES) {
        max_win = _mm256_max_epu32(max_win, eval_8_windows(i));
    }
    if constexpr (MAIN_LOOP_END < NUM_WINDOWS) {
        max_win = _mm256_max_epu32(max_win, eval_8_windows(TAIL_OFFSET));
    }

    __m128i mw128 = _mm_max_epu32(_mm256_castsi256_si128(max_win), _mm256_extracti128_si256(max_win, 1));
    mw128 = _mm_max_epu32(mw128, _mm_srli_si128(mw128, 8));
    mw128 = _mm_max_epu32(mw128, _mm_srli_si128(mw128, 4));
    out.max_cluster4_et = static_cast<uint32_t>(_mm_cvtsi128_si32(mw128));

    // 3. Muon hits (16 x uint16_t = single 256-bit load)
    const __m256i mu = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(frame.muon_hits));
    __m128i mu128 = _mm_max_epu16(_mm256_castsi256_si128(mu), _mm256_extracti128_si256(mu, 1));
    mu128 = _mm_max_epu16(mu128, _mm_srli_si128(mu128, 8));
    mu128 = _mm_max_epu16(mu128, _mm_srli_si128(mu128, 4));
    mu128 = _mm_max_epu16(mu128, _mm_srli_si128(mu128, 2));
    out.max_muon_pt = static_cast<uint16_t>(_mm_extract_epi16(mu128, 0));

    // Unsigned >= comparison: max_epu16(mu, thresh) == mu
    const __m256i thresh_v = _mm256_set1_epi16(static_cast<short>(muon_pt_threshold));
    const __m256i ge_mask = _mm256_cmpeq_epi16(_mm256_max_epu16(mu, thresh_v), mu);
    const uint32_t ge_bits = static_cast<uint32_t>(_mm256_movemask_epi8(ge_mask));
    out.muon_hit_count = static_cast<uint16_t>(std::popcount(ge_bits) >> 1);

    return out;
}
#endif

} // namespace triggerdaq
