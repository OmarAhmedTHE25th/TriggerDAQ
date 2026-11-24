#include "triggerdaq/frontend/detector_emulator.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#include "triggerdaq/core/thread_utils.hpp"
#include "triggerdaq/eb/event_builder.hpp"

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

inline void pace_next_event(EmulatorMode mode,
                            double base_interval_ns,
                            uint64_t now_ns,
                            uint64_t& next_emit_ns,
                            uint64_t& burst_counter,
                            Xoshiro256StarStar& rng) noexcept {
    if (mode == EmulatorMode::Sustained) {
        next_emit_ns += static_cast<uint64_t>(base_interval_ns);
        if (next_emit_ns + 5'000'000ULL < now_ns) {
            next_emit_ns = now_ns;
        }
    } else if (mode == EmulatorMode::PoissonBurst) {
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

inline bool wait_until_emit_or_stop(EmulatorMode mode,
                                    uint64_t next_emit_ns,
                                    std::atomic<bool>& stop_flag,
                                    std::atomic<bool>& paused_flag,
                                    uint64_t& now_ns) noexcept {
    while (paused_flag.load(std::memory_order_acquire) &&
           !stop_flag.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    if (stop_flag.load(std::memory_order_acquire)) {
        return false;
    }

    now_ns = steady_now_ns();
    if (mode != EmulatorMode::MaxThroughputStress) {
        while (now_ns < next_emit_ns &&
               !stop_flag.load(std::memory_order_acquire) &&
               !paused_flag.load(std::memory_order_acquire)) {
            const uint64_t wait_ns = next_emit_ns - now_ns;
            if (wait_ns > 50'000ULL) {
                const uint64_t sleep_us = (wait_ns - 30'000ULL) / 1000ULL;
                std::this_thread::sleep_for(
                    std::chrono::microseconds(sleep_us > 1000ULL ? 1000ULL : sleep_us));
            } else {
                cpu_pause();
            }
            now_ns = steady_now_ns();
        }
        if (stop_flag.load(std::memory_order_acquire)) {
            return false;
        }
    }
    return true;
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
    return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
}

uint32_t Xoshiro256StarStar::next_u32(uint32_t bound) noexcept {
    return static_cast<uint32_t>(next() % bound);
}

void DetectorEmulator::generate_frame(RawEventFrame& frame,
                                      uint64_t event_id,
                                      uint64_t bunch_crossing_id,
                                      uint64_t timestamp_ns,
                                      Xoshiro256StarStar& rng,
                                      uint32_t orbit_id,
                                      uint16_t bcid,
                                      uint32_t lumi_block_id) noexcept {
    std::memset(&frame, 0, sizeof(RawEventFrame));
    frame.event_id = event_id;
    frame.bunch_crossing_id = bunch_crossing_id;
    frame.ingest_timestamp_ns = timestamp_ns;
    frame.orbit_id = orbit_id;
    frame.lumi_block_id = lumi_block_id;
    frame.bcid = bcid;

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

    for (size_t i = 0; i < CALORIMETER_CELLS; ++i) {
        const uint64_t r = rng.next();
        const uint16_t base = static_cast<uint16_t>((r & 0x7U) + ((r >> 8) & 0x7U));
        frame.calorimeter_cells[i] = base;
    }

    switch (tag) {
        case TopologyTag::Background: {
            const uint32_t idx = rng.next_u32(static_cast<uint32_t>(MUON_CHAMBERS));
            frame.muon_hits[idx] = static_cast<uint16_t>(rng.next_u32(6));
            break;
        }
        case TopologyTag::DiMuon: {
            // Correlated high-pT Z-like di-muon pair (M_inv centered around ~65-110 GeV)
            const uint32_t ch1 = rng.next_u32(static_cast<uint32_t>(MUON_CHAMBERS));
            const uint32_t delta = 5U + rng.next_u32(6);
            const uint32_t ch2 = (ch1 + delta) & 15U;
            frame.muon_hits[ch1] = static_cast<uint16_t>(35U + rng.next_u32(45));
            frame.muon_hits[ch2] = static_cast<uint16_t>(30U + rng.next_u32(42));
            break;
        }
        case TopologyTag::LocalizedJet: {
            const uint32_t start = rng.next_u32(58);
            const uint32_t width = 4U + rng.next_u32(3);
            for (uint32_t j = 0; j < width; ++j) {
                const uint16_t deposit = static_cast<uint16_t>(180U + rng.next_u32(220));
                frame.calorimeter_cells[start + j] =
                    static_cast<uint16_t>(frame.calorimeter_cells[start + j] + deposit);
            }
            if ((rng.next() & 3U) == 0U) {
                frame.muon_hits[rng.next_u32(static_cast<uint32_t>(MUON_CHAMBERS))] =
                    static_cast<uint16_t>(4U + rng.next_u32(8));
            }
            break;
        }
        case TopologyTag::PileupNoise: {
            for (size_t i = 0; i < CALORIMETER_CELLS; ++i) {
                frame.calorimeter_cells[i] = static_cast<uint16_t>(45U + rng.next_u32(30));
            }
            frame.muon_hits[rng.next_u32(static_cast<uint32_t>(MUON_CHAMBERS))] =
                static_cast<uint16_t>(6U + rng.next_u32(9));
            break;
        }
    }
}

void DetectorEmulator::run(std::atomic<bool>& stop_flag,
                           std::atomic<bool>& paused_flag,
                           std::atomic<uint32_t>& current_lumi_block,
                           SlabArena& arena,
                           BoundedChannel<SlotHandle>& l1_queue,
                           EmulatorTelemetry& telemetry,
                           EmulatorMode mode,
                           double target_rate_khz,
                           uint64_t seed,
                           const LhcFillingScheme& filling_scheme,
                           bool pin_thread,
                           uint32_t core_idx) noexcept {
    if (pin_thread) {
        pin_current_thread(core_idx);
    }

    Xoshiro256StarStar rng(seed);
    LhcOrbitClock clock{};
    uint64_t event_id = 1;

    const double rate_hz = target_rate_khz > 0.001 ? target_rate_khz * 1000.0 : 1000.0;
    const double base_interval_ns = 1e9 / rate_hz;

    uint64_t next_emit_ns = steady_now_ns();
    uint64_t burst_counter = 0;

    while (!stop_flag.load(std::memory_order_acquire)) {
        uint64_t now_ns = 0;
        if (!wait_until_emit_or_stop(mode, next_emit_ns, stop_flag, paused_flag, now_ns)) {
            break;
        }

        const SlotHandle slot = arena.acquire();
        if (slot == INVALID_SLOT) {
            const uint64_t dt_start = steady_now_ns();
            telemetry.dead_time_events.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
            const uint64_t dt_end = steady_now_ns();
            if (dt_end > dt_start) {
                telemetry.dead_time_ns.fetch_add(dt_end - dt_start, std::memory_order_relaxed);
            }
            continue;
        }

        clock.advance_to_next_filled(filling_scheme);
        const uint64_t global_bx =
            static_cast<uint64_t>(clock.orbit_id) * filling_scheme.bunches_per_orbit + clock.bcid;
        const uint32_t lb_id = current_lumi_block.load(std::memory_order_acquire);

        RawEventFrame* frame = arena.get_frame(slot);
        generate_frame(*frame, event_id++, global_bx, now_ns, rng, clock.orbit_id, clock.bcid, lb_id);

        const auto tag = static_cast<TopologyTag>(frame->topology_tag);
        telemetry.events_generated.fetch_add(1, std::memory_order_relaxed);
        if (tag == TopologyTag::DiMuon || tag == TopologyTag::LocalizedJet) {
            telemetry.signal_generated.fetch_add(1, std::memory_order_relaxed);
        }

        if (!l1_queue.try_push(slot)) {
            arena.release(slot);
            telemetry.dead_time_events.fetch_add(1, std::memory_order_relaxed);
            cpu_pause();
        }

        pace_next_event(mode, base_interval_ns, now_ns, next_emit_ns, burst_counter, rng);
    }
}

void DetectorEmulator::run_split_robs(std::atomic<bool>& stop_flag,
                                      std::atomic<bool>& paused_flag,
                                      std::atomic<uint32_t>& current_lumi_block,
                                      BoundedChannel<CaloRobFragment>& calo_queue,
                                      BoundedChannel<MuonRobFragment>& muon_queue,
                                      EmulatorTelemetry& telemetry,
                                      EmulatorMode mode,
                                      double target_rate_khz,
                                      uint64_t seed,
                                      double muon_drop_rate,
                                      const LhcFillingScheme& filling_scheme,
                                      bool pin_thread,
                                      uint32_t core_idx) noexcept {
    if (pin_thread) {
        pin_current_thread(core_idx);
    }

    Xoshiro256StarStar rng(seed);
    LhcOrbitClock clock{};
    uint64_t event_id = 1;

    const double rate_hz = target_rate_khz > 0.001 ? target_rate_khz * 1000.0 : 1000.0;
    const double base_interval_ns = 1e9 / rate_hz;

    uint64_t next_emit_ns = steady_now_ns();
    uint64_t burst_counter = 0;

    RawEventFrame scratch{};
    while (!stop_flag.load(std::memory_order_acquire)) {
        uint64_t now_ns = 0;
        if (!wait_until_emit_or_stop(mode, next_emit_ns, stop_flag, paused_flag, now_ns)) {
            break;
        }

        clock.advance_to_next_filled(filling_scheme);
        const uint64_t global_bx =
            static_cast<uint64_t>(clock.orbit_id) * filling_scheme.bunches_per_orbit + clock.bcid;
        const uint32_t lb_id = current_lumi_block.load(std::memory_order_acquire);
        const uint64_t eid = event_id++;

        generate_frame(scratch, eid, global_bx, now_ns, rng, clock.orbit_id, clock.bcid, lb_id);

        CaloRobFragment calo{};
        calo.event_id = eid;
        calo.bunch_crossing_id = global_bx;
        calo.ingest_timestamp_ns = now_ns;
        calo.orbit_id = clock.orbit_id;
        calo.lumi_block_id = lb_id;
        calo.bcid = clock.bcid;
        calo.topology_tag = scratch.topology_tag;
        std::memcpy(calo.calorimeter_cells, scratch.calorimeter_cells, sizeof(calo.calorimeter_cells));

        const bool drop_muon = (muon_drop_rate > 0.0) && (rng.next_uniform() < muon_drop_rate);
        if (!calo_queue.try_push(calo)) {
            telemetry.dead_time_events.fetch_add(1, std::memory_order_relaxed);
            cpu_pause();
            continue;
        }

        const auto tag = static_cast<TopologyTag>(scratch.topology_tag);
        telemetry.events_generated.fetch_add(1, std::memory_order_relaxed);
        if (tag == TopologyTag::DiMuon || tag == TopologyTag::LocalizedJet) {
            telemetry.signal_generated.fetch_add(1, std::memory_order_relaxed);
        }

        if (!drop_muon) {
            MuonRobFragment muon{};
            muon.event_id = eid;
            muon.orbit_id = clock.orbit_id;
            muon.bcid = clock.bcid;
            std::memcpy(muon.muon_hits, scratch.muon_hits, sizeof(muon.muon_hits));
            muon_queue.push_blocking(muon);
        }

        pace_next_event(mode, base_interval_ns, now_ns, next_emit_ns, burst_counter, rng);
    }
}

} // namespace triggerdaq
