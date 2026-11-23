#pragma once

#include <atomic>
#include <cstdint>

#include "triggerdaq/core/bounded_channel.hpp"
#include "triggerdaq/core/event_frame.hpp"
#include "triggerdaq/core/slab_arena.hpp"

namespace triggerdaq {

enum class EmulatorMode : uint8_t {
    Sustained = 0,
    PoissonBurst = 1,
    MaxThroughputStress = 2,
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
                               Xoshiro256StarStar& rng) noexcept;

    // Runs the generation loop until stop_flag becomes true.
    void run(std::atomic<bool>& stop_flag,
             SlabArena& arena,
             BoundedChannel<SlotHandle>& l1_queue,
             EmulatorTelemetry& telemetry,
             EmulatorMode mode,
             double target_rate_khz,
             uint64_t seed) noexcept;
};

} // namespace triggerdaq
