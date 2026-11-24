#pragma once

#include <atomic>
#include <cstdint>

#include "triggerdaq/core/bounded_channel.hpp"
#include "triggerdaq/core/event_frame.hpp"
#include "triggerdaq/core/slab_arena.hpp"

namespace triggerdaq {

struct CaloRobFragment;
struct MuonRobFragment;

enum class EmulatorMode : uint8_t {
    Sustained = 0,
    PoissonBurst = 1,
    MaxThroughputStress = 2,
};

// Configurable LHC 25ns bunch filling scheme (filled bunch trains + abort gap).
struct LhcFillingScheme {
    uint16_t bunches_per_orbit{LHC_BUNCHES_PER_ORBIT}; // 3564
    uint16_t abort_gap_start{3444};                    // last 120 BCIDs = abort gap
    uint16_t train_length{72};                         // 72 filled bunches per train
    uint16_t train_gap{8};                             // 8 empty bunches between trains

    [[nodiscard]] bool is_in_abort_gap(uint16_t bcid) const noexcept {
        return bcid >= abort_gap_start && bcid < bunches_per_orbit;
    }

    [[nodiscard]] bool is_filled(uint16_t bcid) const noexcept {
        if (bcid >= bunches_per_orbit || bcid >= abort_gap_start) {
            return false;
        }
        const uint16_t period = static_cast<uint16_t>(train_length + train_gap);
        if (period == 0 || train_length == 0) {
            return true;
        }
        return (bcid % period) < train_length;
    }
};

// Tracks LHC (orbit_id, bcid) progression, skipping empty gaps and the abort gap.
struct LhcOrbitClock {
    uint32_t orbit_id{0};
    uint16_t bcid{0};
    bool started{false};

    void advance_to_next_filled(const LhcFillingScheme& scheme) noexcept {
        if (!started) {
            started = true;
            if (scheme.is_filled(bcid)) {
                return;
            }
        }
        for (uint16_t steps = 0; steps <= scheme.bunches_per_orbit; ++steps) {
            ++bcid;
            if (bcid >= scheme.bunches_per_orbit) {
                bcid = 0;
                ++orbit_id;
            }
            if (scheme.is_filled(bcid)) {
                return;
            }
        }
    }
};

// Fast zero-allocation xoshiro256** PRNG
class Xoshiro256StarStar {
public:
    explicit Xoshiro256StarStar(uint64_t seed = 0x9E3779B97F4A7C15ULL) noexcept;

    [[nodiscard]] uint64_t next() noexcept;
    [[nodiscard]] double next_uniform() noexcept; // [0.0, 1.0)
    [[nodiscard]] uint32_t next_u32(uint32_t bound) noexcept;

private:
    uint64_t s_[4]{};
};

struct alignas(64) EmulatorTelemetry {
    std::atomic<uint64_t> events_generated{0};
    std::atomic<uint64_t> signal_generated{0};
    std::atomic<uint64_t> dead_time_events{0};
    std::atomic<uint64_t> dead_time_ns{0};

    void reset() noexcept {
        events_generated.store(0, std::memory_order_relaxed);
        signal_generated.store(0, std::memory_order_relaxed);
        dead_time_events.store(0, std::memory_order_relaxed);
        dead_time_ns.store(0, std::memory_order_relaxed);
    }
};

class DetectorEmulator {
public:
    DetectorEmulator() = default;

    // Populates a single RawEventFrame according to the physics topology mixture.
    static void generate_frame(RawEventFrame& frame,
                               uint64_t event_id,
                               uint64_t bunch_crossing_id,
                               uint64_t timestamp_ns,
                               Xoshiro256StarStar& rng,
                               uint32_t orbit_id = 0,
                               uint16_t bcid = 0,
                               uint32_t lumi_block_id = 1) noexcept;

    // Direct mode: emits unified RawEventFrame SlotHandles into l1_queue.
    void run(std::atomic<bool>& stop_flag,
             std::atomic<bool>& paused_flag,
             std::atomic<uint32_t>& current_lumi_block,
             SlabArena& arena,
             BoundedChannel<SlotHandle>& l1_queue,
             EmulatorTelemetry& telemetry,
             EmulatorMode mode,
             double target_rate_khz,
             uint64_t seed,
             const LhcFillingScheme& filling_scheme = LhcFillingScheme{},
             bool pin_thread = false,
             uint32_t core_idx = 0) noexcept;

    // Event Builder mode: splits output into independent Calo & Muon ROB fragment streams.
    void run_split_robs(std::atomic<bool>& stop_flag,
                        std::atomic<bool>& paused_flag,
                        std::atomic<uint32_t>& current_lumi_block,
                        BoundedChannel<CaloRobFragment>& calo_queue,
                        BoundedChannel<MuonRobFragment>& muon_queue,
                        EmulatorTelemetry& telemetry,
                        EmulatorMode mode,
                        double target_rate_khz,
                        uint64_t seed,
                        double muon_drop_rate,
                        const LhcFillingScheme& filling_scheme = LhcFillingScheme{},
                        bool pin_thread = false,
                        uint32_t core_idx = 0) noexcept;
};

} // namespace triggerdaq
