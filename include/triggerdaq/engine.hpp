#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "triggerdaq/core/bounded_channel.hpp"
#include "triggerdaq/core/slab_arena.hpp"
#include "triggerdaq/core/telemetry.hpp"
#include "triggerdaq/eb/event_builder.hpp"
#include "triggerdaq/frontend/detector_emulator.hpp"
#include "triggerdaq/hlt/hlt_farm.hpp"
#include "triggerdaq/l1/l1_trigger.hpp"
#include "triggerdaq/l1/prescaler.hpp"
#include "triggerdaq/storage/event_writer.hpp"

namespace triggerdaq {

enum class RunState : uint8_t {
    Initial = 0,
    Configured = 1,
    Running = 2,
    Paused = 3,
    Stopped = 4,
};

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

    // CERN domain extensions
    LhcFillingScheme filling_scheme{};
    double lumi_block_interval_s{1.0};
    bool enable_zero_suppression{false};
    uint16_t zs_pedestal_threshold{20};
    bool enable_event_builder{false};
    uint64_t eb_timeout_us{500};
    double eb_muon_drop_rate{0.0};
};

class TriggerEngine {
public:
    TriggerEngine() = default;
    ~TriggerEngine();

    TriggerEngine(const TriggerEngine&) = delete;
    TriggerEngine& operator=(const TriggerEngine&) = delete;

    // Run Control FSM transitions
    void configure(const EngineConfig& config, const TriggerMenu& menu);
    void initialize(const EngineConfig& config, const TriggerMenu& menu) { configure(config, menu); }
    void start();
    void pause();
    void resume();
    void stop();
    void reset();

    // Convenience: start(), sleep for `seconds`, then stop().
    void run_for_seconds(double seconds);

    // Lock-free atomic hot-reload of the active TriggerMenu at runtime.
    void update_menu(const TriggerMenu& menu) noexcept;

    // Lock-free telemetry snapshot (< 1 us, never stalls worker threads).
    [[nodiscard]] TelemetrySnapshot snapshot() const noexcept;

    // Returns per-LumiBlock accounting ledger recorded during the run.
    [[nodiscard]] std::vector<LumiBlockRecord> lumi_blocks() const;

    [[nodiscard]] RunState state() const noexcept {
        return state_.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool is_running() const noexcept {
        const RunState st = state_.load(std::memory_order_acquire);
        return st == RunState::Running || st == RunState::Paused;
    }

    [[nodiscard]] const EngineConfig& config() const noexcept { return config_; }

private:
    void lumi_block_loop() noexcept;
    void record_lumi_block(uint32_t lb_id, double start_s, double end_s,
                           uint64_t& prev_gen, uint64_t& prev_l1,
                           uint64_t& prev_hlt, uint64_t& prev_dt);

    EngineConfig config_{};
    std::atomic<RunState> state_{RunState::Initial};
    std::atomic<bool> stop_frontend_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> stop_lumi_timer_{false};
    std::atomic<uint32_t> current_lumi_block_{1};

    SlabArena arena_;
    BoundedChannel<CaloRobFragment> calo_rob_queue_;
    BoundedChannel<MuonRobFragment> muon_rob_queue_;
    BoundedChannel<SlotHandle> l1_queue_;
    BoundedChannel<SlotHandle> hlt_queue_;
    BoundedChannel<SlotHandle> storage_queue_;
    AtomicMenuSlot menu_slot_;

    DetectorEmulator emulator_;
    EventBuilder event_builder_;
    L1Trigger l1_trigger_;
    HltFarm hlt_farm_;
    EventWriter event_writer_;

    EmulatorTelemetry emulator_telemetry_;
    EventBuilderTelemetry eb_telemetry_;
    L1Telemetry l1_telemetry_;
    StorageTelemetry storage_telemetry_;

    std::jthread frontend_thread_;
    std::jthread eb_thread_;
    std::jthread l1_thread_;
    std::jthread storage_thread_;
    std::jthread lumi_thread_;

    mutable std::mutex lumi_mu_;
    std::vector<LumiBlockRecord> lumi_ledger_;

    std::chrono::steady_clock::time_point start_time_{};
    std::chrono::steady_clock::time_point end_time_{};
};

} // namespace triggerdaq
