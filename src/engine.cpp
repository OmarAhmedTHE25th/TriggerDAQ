#include "triggerdaq/engine.hpp"

#include <chrono>
#include <stdexcept>
#include <thread>

namespace triggerdaq {

TriggerEngine::~TriggerEngine() {
    stop();
}

void TriggerEngine::initialize(const EngineConfig& config, const TriggerMenu& menu) {
    if (running_.load(std::memory_order_acquire)) {
        stop();
    }

    config_ = config;
    arena_.initialize(config_.arena_slots);
    l1_queue_.initialize(config_.l1_queue_capacity);
    hlt_queue_.initialize(config_.hlt_queue_capacity);
    storage_queue_.initialize(config_.storage_queue_capacity);
    menu_slot_.initialize(menu);
    hlt_farm_.initialize(config_.hlt_workers, config_.pin_threads);
    event_writer_.initialize(config_.output_path);

    emulator_telemetry_.reset();
    l1_telemetry_.reset();
    storage_telemetry_.reset();

    initialized_ = true;
}

void TriggerEngine::start() {
    if (!initialized_) {
        throw std::runtime_error("TriggerEngine::initialize() must be called before start()");
    }
    if (running_.exchange(true, std::memory_order_acq_rel)) {
        return; // Already running
    }

    stop_frontend_.store(false, std::memory_order_release);
    l1_queue_.reopen();
    hlt_queue_.reopen();
    storage_queue_.reopen();

    emulator_telemetry_.reset();
    l1_telemetry_.reset();
    storage_telemetry_.reset();

    start_time_ = std::chrono::steady_clock::now();
    end_time_ = start_time_;

    // Start downstream stages first so queues are ready to drain immediately
    storage_thread_ = std::jthread([this]() {
        event_writer_.run(arena_, storage_queue_, storage_telemetry_, config_.pin_threads, 2U);
    });

    hlt_farm_.start(arena_, hlt_queue_, storage_queue_, menu_slot_);

    l1_thread_ = std::jthread([this]() {
        l1_trigger_.run(arena_, l1_queue_, hlt_queue_, menu_slot_,
                        l1_telemetry_, config_.l1_budget_ns, config_.pin_threads, 1U);
    });

    frontend_thread_ = std::jthread([this]() {
        emulator_.run(stop_frontend_, arena_, l1_queue_, emulator_telemetry_,
                      config_.mode, config_.target_rate_khz, config_.seed,
                      config_.pin_threads, 0U);
    });
}

void TriggerEngine::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }

    // Step 1: Stop DetectorEmulator and join
    stop_frontend_.store(true, std::memory_order_release);
    if (frontend_thread_.joinable()) {
        frontend_thread_.join();
    }

    // Step 2: Close L1 channel; L1 drains losslessly into HLT, closes hlt_queue_, and joins
    l1_queue_.close();
    if (l1_thread_.joinable()) {
        l1_thread_.join();
    }

    // Step 3: HLT workers drain all queued slots until hlt_queue_ is empty & closed,
    // push losslessly into storage_queue_, and join; then close storage_queue_.
    hlt_farm_.stop();
    storage_queue_.close();

    // Step 4: Storage drains all remaining records, flushes double buffers to disk, and joins.
    if (storage_thread_.joinable()) {
        storage_thread_.join();
    }

    end_time_ = std::chrono::steady_clock::now();
}

void TriggerEngine::run_for_seconds(double seconds) {
    start();
    if (seconds > 0.0) {
        const auto dur = std::chrono::duration<double>(seconds);
        std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::microseconds>(dur));
    }
    stop();
}

void TriggerEngine::update_menu(const TriggerMenu& menu) noexcept {
    menu_slot_.update(menu);
}

TelemetrySnapshot TriggerEngine::snapshot() const noexcept {
    TelemetrySnapshot snap{};
    if (!initialized_) {
        return snap;
    }

    const auto now = running_.load(std::memory_order_acquire)
                         ? std::chrono::steady_clock::now()
                         : end_time_;
    const double elapsed =
        std::chrono::duration<double>(now - start_time_).count();
    snap.elapsed_seconds = elapsed > 0.0 ? elapsed : 0.0;

    snap.events_generated = emulator_telemetry_.events_generated.load(std::memory_order_relaxed);
    snap.signal_generated = emulator_telemetry_.signal_generated.load(std::memory_order_relaxed);
    snap.dead_time_ns = emulator_telemetry_.dead_time_ns.load(std::memory_order_relaxed);

    snap.l1_accepted = l1_telemetry_.l1_accepted.load(std::memory_order_relaxed);
    snap.l1_rejected = l1_telemetry_.l1_rejected.load(std::memory_order_relaxed);
    snap.l1_budget_violations = l1_telemetry_.l1_budget_violations.load(std::memory_order_relaxed);

    uint64_t total_dead =
        emulator_telemetry_.dead_time_events.load(std::memory_order_relaxed) +
        l1_telemetry_.dead_time_events.load(std::memory_order_relaxed);

    LatencyHistogram l1_hist;
    l1_telemetry_.l1_latency_ns.copy_to(l1_hist);
    snap.l1_latency_ns = l1_hist.compute_stats();

    LatencyHistogram hlt_hist;
    hlt_hist.reset();
    LatencyHistogram worker_hist;
    for (uint32_t i = 0; i < hlt_farm_.num_workers(); ++i) {
        const auto& wt = hlt_farm_.worker_telemetry(i);
        snap.hlt_accepted += wt.hlt_accepted.load(std::memory_order_relaxed);
        snap.hlt_rejected += wt.hlt_rejected.load(std::memory_order_relaxed);
        snap.hlt_true_positives += wt.hlt_true_positives.load(std::memory_order_relaxed);
        snap.hlt_false_positives += wt.hlt_false_positives.load(std::memory_order_relaxed);
        total_dead += wt.dead_time_events.load(std::memory_order_relaxed);

        wt.hlt_latency_ns.copy_to(worker_hist);
        hlt_hist.merge_from(worker_hist);
    }

    const LatencyStats hlt_ns = hlt_hist.compute_stats();
    snap.hlt_latency_us.min = hlt_ns.min * 1e-3;
    snap.hlt_latency_us.p50 = hlt_ns.p50 * 1e-3;
    snap.hlt_latency_us.p95 = hlt_ns.p95 * 1e-3;
    snap.hlt_latency_us.p99 = hlt_ns.p99 * 1e-3;
    snap.hlt_latency_us.p999 = hlt_ns.p999 * 1e-3;
    snap.hlt_latency_us.max = hlt_ns.max * 1e-3;

    snap.dead_time_events = total_dead;
    snap.bytes_written = storage_telemetry_.bytes_written.load(std::memory_order_relaxed);
    snap.events_written = storage_telemetry_.events_written.load(std::memory_order_relaxed);
    snap.storage_write_errors = storage_telemetry_.storage_write_errors.load(std::memory_order_relaxed);

    snap.arena_allocated = arena_.allocated_count();
    snap.arena_peak = arena_.peak_occupancy();
    snap.arena_capacity = arena_.capacity();

    snap.l1_queue_fill = l1_queue_.fill_ratio();
    snap.hlt_queue_fill = hlt_queue_.fill_ratio();
    snap.storage_queue_fill = storage_queue_.fill_ratio();

    const TriggerMenu& active_menu = menu_slot_.load();
    snap.num_rules = active_menu.num_rules;
    for (size_t i = 0; i < TriggerMenu::MAX_RULES; ++i) {
        snap.prescale_factors[i] = l1_telemetry_.active_prescales[i].load(std::memory_order_relaxed);
        snap.rule_accept_counts[i] = l1_telemetry_.rule_accept_counts[i].load(std::memory_order_relaxed);
    }

    if (snap.elapsed_seconds > 1e-6) {
        snap.input_rate_hz = static_cast<double>(snap.events_generated) / snap.elapsed_seconds;
        snap.l1_rate_hz = static_cast<double>(snap.l1_accepted) / snap.elapsed_seconds;
        snap.hlt_rate_hz = static_cast<double>(snap.hlt_accepted) / snap.elapsed_seconds;
    }

    const uint64_t total_attempts = snap.events_generated + snap.dead_time_events;
    if (total_attempts > 0) {
        snap.dead_time_fraction =
            static_cast<double>(snap.dead_time_events) / static_cast<double>(total_attempts);
    }
    if (snap.hlt_accepted > 0) {
        snap.signal_purity =
            static_cast<double>(snap.hlt_true_positives) / static_cast<double>(snap.hlt_accepted);
    }
    if (snap.signal_generated > 0) {
        snap.signal_efficiency =
            static_cast<double>(snap.hlt_true_positives) / static_cast<double>(snap.signal_generated);
    }

    return snap;
}

} // namespace triggerdaq
