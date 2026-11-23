#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

#include "triggerdaq/core/bounded_channel.hpp"
#include "triggerdaq/core/slab_arena.hpp"
#include "triggerdaq/core/telemetry.hpp"
#include "triggerdaq/frontend/detector_emulator.hpp"
#include "triggerdaq/hlt/hlt_farm.hpp"
#include "triggerdaq/l1/l1_trigger.hpp"
#include "triggerdaq/l1/prescaler.hpp"
#include "triggerdaq/storage/event_writer.hpp"

namespace triggerdaq {

struct EngineConfig {
    uint32_t arena_slots{65536};
    uint32_t l1_queue_capacity{16384};
    uint32_t hlt_queue_capacity{16384};
    uint32_t storage_queue_capacity{16384};
    uint32_t hlt_workers{4};
    double target_rate_khz{100.0};
    uint64_t l1_budget_ns{10000};
    EmulatorMode mode{EmulatorMode::Sustained};
    bool pin_threads{false};
    uint64_t seed{42};
    std::string output_path;
};

class TriggerEngine {
public:
    TriggerEngine() = default;
    ~TriggerEngine();

    TriggerEngine(const TriggerEngine&) = delete;
    TriggerEngine& operator=(const TriggerEngine&) = delete;

    // Pre-allocates all memory arenas, channels, and staging buffers.
    void initialize(const EngineConfig& config, const TriggerMenu& menu);

    // Starts Frontend, L1, HLT, and Storage threads.
    void start();

    // Gracefully drains queues and stops all threads.
    void stop();

    // Convenience: start(), sleep for `seconds`, then stop().
    void run_for_seconds(double seconds);

    // Lock-free atomic hot-reload of the active TriggerMenu at runtime.
    void update_menu(const TriggerMenu& menu) noexcept;

    // Lock-free telemetry snapshot (< 1 us, never stalls worker threads).
    [[nodiscard]] TelemetrySnapshot snapshot() const noexcept;

    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    [[nodiscard]] const EngineConfig& config() const noexcept { return config_; }

private:
    EngineConfig config_{};
    bool initialized_{false};
    std::atomic<bool> running_{false};

    // Ordered shutdown flags for clean draining
    std::atomic<bool> stop_frontend_{false};
    std::atomic<bool> stop_l1_{false};
    std::atomic<bool> stop_hlt_{false};
    std::atomic<bool> stop_storage_{false};

    SlabArena arena_;
    BoundedChannel<SlotHandle> l1_queue_;
    BoundedChannel<SlotHandle> hlt_queue_;
    BoundedChannel<SlotHandle> storage_queue_;
    AtomicMenuSlot menu_slot_;

    DetectorEmulator emulator_;
    L1Trigger l1_trigger_;
    HltFarm hlt_farm_;
    EventWriter event_writer_;

    EmulatorTelemetry emulator_telemetry_;
    L1Telemetry l1_telemetry_;
    StorageTelemetry storage_telemetry_;

    std::jthread frontend_thread_;
    std::jthread l1_thread_;
    std::jthread storage_thread_;

    std::chrono::steady_clock::time_point start_time_{};
    std::chrono::steady_clock::time_point end_time_{};
};

} // namespace triggerdaq
