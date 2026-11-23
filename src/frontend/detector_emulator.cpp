#include "triggerdaq/frontend/detector_emulator.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

namespace triggerdaq {

namespace {

inline uint64_t splitmix64(uint64_t& x) noexcept {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline uint64_t rotl64(uint64_t x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
}

inline uint64_t steady_now_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

} // namespace

Xoshiro256StarStar::Xoshiro256StarStar(uint64_t seed) noexcept {
    uint64_t st = seed;
    s_[0] = splitmix64(st);
    s_[1] = splitmix64(st);
    s_[2] = splitmix64(st);
    s_[3] = splitmix64(st);
}

uint64_t Xoshiro256StarStar::next() noexcept {
    const uint64_t result = rotl64(s_[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl64(s_[3], 45);
    return result;
}

double Xoshiro256StarStar::next_uniform() noexcept {
    // 53-bit uniform double in [0, 1)
    return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
}

uint32_t Xoshiro256StarStar::next_u32(uint32_t bound) noexcept {
    return static_cast<uint32_t>(next() % bound);
}

void DetectorEmulator::generate_frame(RawEventFrame& frame,
                                      uint64_t event_id,
                                      uint64_t bunch_crossing_id,
                                      uint64_t timestamp_ns,
                                      Xoshiro256StarStar& rng) noexcept {
    std::memset(&frame, 0, sizeof(RawEventFrame));
    frame.event_id = event_id;
    frame.bunch_crossing_id = bunch_crossing_id;
    frame.ingest_timestamp_ns = timestamp_ns;

    const double u = rng.next_uniform();
    TopologyTag tag = TopologyTag::Background;
    if (u < 0.015) {
        tag = TopologyTag::DiMuon;
    } else if (u < 0.035) {
        tag = TopologyTag::LocalizedJet;
    } else if (u < 0.050) {
        tag = TopologyTag::PileupNoise;
    }
    frame.topology_tag = static_cast<uint8_t>(tag);

    // Fast low-energy background calorimeter baseline using cheap bit shifts (zero std::log on hot path)
    for (int i = 0; i < 64; ++i) {
        const uint64_t r = rng.next();
        // Exponential-like tail via minimum of two uniform samples or shift
        const uint16_t base = static_cast<uint16_t>((r & 0x7U) + ((r >> 8) & 0x7U));
        frame.calorimeter_cells[i] = base; // ~0..14 MeV
    }

    switch (tag) {
        case TopologyTag::Background: {
            // Sparse low-momentum muon noise hits
            const uint32_t idx = rng.next_u32(16);
            frame.muon_hits[idx] = static_cast<uint16_t>(rng.next_u32(6));
            break;
        }
        case TopologyTag::DiMuon: {
            // Two correlated high-pT muon hits separated in phi (chambers 4..10 apart)
            const uint32_t ch1 = rng.next_u32(16);
            const uint32_t delta = 4U + rng.next_u32(7);
            const uint32_t ch2 = (ch1 + delta) & 15U;
            frame.muon_hits[ch1] = static_cast<uint16_t>(28U + rng.next_u32(50));
            frame.muon_hits[ch2] = static_cast<uint16_t>(22U + rng.next_u32(45));
            break;
        }
        case TopologyTag::LocalizedJet: {
            // Tightly clustered high-energy burst across 4-6 adjacent calorimeter cells
            const uint32_t start = rng.next_u32(58);
            const uint32_t width = 4U + rng.next_u32(3);
            for (uint32_t j = 0; j < width; ++j) {
                const uint16_t deposit = static_cast<uint16_t>(180U + rng.next_u32(220));
                frame.calorimeter_cells[start + j] =
                    static_cast<uint16_t>(frame.calorimeter_cells[start + j] + deposit);
            }
            // Occasional soft muon in jet
            if ((rng.next() & 3U) == 0U) {
                frame.muon_hits[rng.next_u32(16)] = static_cast<uint16_t>(4U + rng.next_u32(8));
            }
            break;
        }
        case TopologyTag::PileupNoise: {
            // High total energy spread uniformly across all 64 cells (tests topological rejection)
            for (int i = 0; i < 64; ++i) {
                frame.calorimeter_cells[i] = static_cast<uint16_t>(45U + rng.next_u32(30));
            }
            // Soft diffuse muon hits below di-muon discovery threshold
            frame.muon_hits[rng.next_u32(16)] = static_cast<uint16_t>(6U + rng.next_u32(9));
            break;
        }
    }
}

void DetectorEmulator::run(std::atomic<bool>& stop_flag,
                           SlabArena& arena,
                           BoundedChannel<SlotHandle>& l1_queue,
                           EmulatorTelemetry& telemetry,
                           EmulatorMode mode,
                           double target_rate_khz,
                           uint64_t seed) noexcept {
    Xoshiro256StarStar rng(seed);
    uint64_t event_id = 1;
    uint64_t bx_id = 0;

    const double rate_hz = target_rate_khz > 0.001 ? target_rate_khz * 1000.0 : 1000.0;
    const double base_interval_ns = 1e9 / rate_hz;

    uint64_t next_emit_ns = steady_now_ns();
    uint64_t burst_counter = 0;

    while (!stop_flag.load(std::memory_order_relaxed)) {
        uint64_t now_ns = steady_now_ns();

        if (mode != EmulatorMode::MaxThroughputStress) {
            while (now_ns < next_emit_ns && !stop_flag.load(std::memory_order_relaxed)) {
                const uint64_t wait_ns = next_emit_ns - now_ns;
                if (wait_ns > 50000ULL) {
                    std::this_thread::sleep_for(std::chrono::microseconds(25));
                } else {
                    std::this_thread::yield();
                }
                now_ns = steady_now_ns();
            }
            if (stop_flag.load(std::memory_order_relaxed)) {
                break;
            }
        }

        const SlotHandle slot = arena.acquire();
        if (slot == INVALID_SLOT) {
            const uint64_t dt_start = now_ns;
            telemetry.dead_time_events.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
            const uint64_t dt_end = steady_now_ns();
            telemetry.dead_time_ns.fetch_add(dt_end - dt_start, std::memory_order_relaxed);
            continue;
        }

        RawEventFrame* frame = arena.get_frame(slot);
        generate_frame(*frame, event_id++, bx_id++, now_ns, rng);

        const auto tag = static_cast<TopologyTag>(frame->topology_tag);
        telemetry.events_generated.fetch_add(1, std::memory_order_relaxed);
        if (tag == TopologyTag::DiMuon || tag == TopologyTag::LocalizedJet) {
            telemetry.signal_generated.fetch_add(1, std::memory_order_relaxed);
        }

        if (!l1_queue.try_push(slot)) {
            arena.release(slot);
            telemetry.dead_time_events.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }

        if (mode == EmulatorMode::Sustained) {
            next_emit_ns += static_cast<uint64_t>(base_interval_ns);
            if (next_emit_ns + 5'000'000ULL < now_ns) {
                next_emit_ns = now_ns; // Prevent runaway catch-up after OS stalls
            }
        } else if (mode == EmulatorMode::PoissonBurst) {
            // Alternate between nominal Poisson spacing and 4x luminosity bursts
            ++burst_counter;
            const bool in_burst = (burst_counter % 10000ULL) < 2500ULL;
            double u = rng.next_uniform();
            if (u < 1e-6) {
                u = 1e-6;
            }
            const double scale = in_burst ? 0.25 : 1.25;
            const double delta_ns = -std::log(u) * base_interval_ns * scale;
            next_emit_ns += static_cast<uint64_t>(delta_ns);
            if (next_emit_ns + 5'000'000ULL < now_ns) {
                next_emit_ns = now_ns;
            }
        }
    }
}

} // namespace triggerdaq
