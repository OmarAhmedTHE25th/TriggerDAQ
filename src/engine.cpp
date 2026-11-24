#include "triggerdaq/engine.hpp"

#include <chrono>
#include <stdexcept>
#include <thread>

#include "triggerdaq/core/thread_utils.hpp"

namespace triggerdaq {

TriggerEngine::~TriggerEngine() {
    const RunState st = state_.load(std::memory_order_acquire);
    if (st == RunState::Running || st == RunState::Paused) {
        stop();
    }
}

void TriggerEngine::configure(const EngineConfig& config, const TriggerMenu& menu) {
    std::lock_guard<std::mutex> fsm_lock(fsm_mu_);
    const RunState st = state_.load(std::memory_order_acquire);
    if (st == RunState::Running || st == RunState::Paused) {
        throw std::logic_error("Cannot configure() while TriggerEngine is Running or Paused");
    }

    config_ = config;
    arena_.initialize(config_.arena_slots);
    if (config_.enable_event_builder) {
        calo_rob_queue_.initialize(config_.l1_queue_capacity);
        muon_rob_queue_.initialize(config_.l1_queue_capacity);
        event_builder_.initialize(EventBuilder::DEFAULT_TABLE_SIZE, config_.eb_timeout_us);
    }
    l1_queue_.initialize(config_.l1_queue_capacity);
    hlt_queue_.initialize(config_.hlt_queue_capacity);
    storage_queue_.initialize(config_.storage_queue_capacity);
    menu_slot_.initialize(menu);

    const uint32_t base_hlt_core = config_.enable_event_builder ? 4U : 3U;
    const bool effective_pin =
        config_.pin_threads && can_pin_without_collision(config_.hlt_workers + base_hlt_core);
    hlt_farm_.initialize(config_.hlt_workers, effective_pin, base_hlt_core);
    event_writer_.initialize(config_.output_path,
                             config_.enable_zero_suppression,
                             config_.zs_pedestal_threshold);

    emulator_telemetry_.reset();
    eb_telemetry_.reset();
    l1_telemetry_.reset();
    storage_telemetry_.reset();
    {
        std::lock_guard<std::mutex> lock(lumi_mu_);
        lumi_ledger_.clear();
    }
    current_lumi_block_.store(1, std::memory_order_release);

    state_.store(RunState::Configured, std::memory_order_release);
}

void TriggerEngine::start() {
    std::lock_guard<std::mutex> fsm_lock(fsm_mu_);
    const RunState st = state_.load(std::memory_order_acquire);
    if (st != RunState::Configured && st != RunState::Stopped) {
        throw std::logic_error("TriggerEngine::start() requires Configured or Stopped state");
    }

    stop_frontend_.store(false, std::memory_order_release);
    paused_.store(false, std::memory_order_release);
    stop_lumi_timer_.store(false, std::memory_order_release);
    current_lumi_block_.store(1, std::memory_order_release);

    if (config_.enable_event_builder) {
        calo_rob_queue_.reopen();
        muon_rob_queue_.reopen();
    }
    l1_queue_.reopen();
    hlt_queue_.reopen();
    storage_queue_.reopen();

    emulator_telemetry_.reset();
    eb_telemetry_.reset();
    l1_telemetry_.reset();
    storage_telemetry_.reset();
    {
        std::lock_guard<std::mutex> lock(lumi_mu_);
        lumi_ledger_.clear();
    }

    start_time_ = std::chrono::steady_clock::now();
    end_time_ = start_time_;
    state_.store(RunState::Running, std::memory_order_release);

    const uint32_t fixed_threads = config_.enable_event_builder ? 4U : 3U;
    const bool effective_pin =
        config_.pin_threads && can_pin_without_collision(config_.hlt_workers + fixed_threads);

    // Start downstream stages first so queues are ready to drain immediately
    storage_thread_ = std::jthread([this, effective_pin]() {
        event_writer_.run(arena_, storage_queue_, storage_telemetry_, effective_pin, 2U);
    });

    hlt_farm_.start(arena_, hlt_queue_, storage_queue_, menu_slot_);

    l1_thread_ = std::jthread([this, effective_pin]() {
        l1_trigger_.run(arena_, l1_queue_, hlt_queue_, menu_slot_,
                        l1_telemetry_, config_.l1_budget_ns, effective_pin, 1U);
    });

    if (config_.enable_event_builder) {
        eb_thread_ = std::jthread([this, effective_pin]() {
            event_builder_.run(arena_, calo_rob_queue_, muon_rob_queue_, l1_queue_,
                               eb_telemetry_, effective_pin, 3U);
        });
        frontend_thread_ = std::jthread([this, effective_pin]() {
            emulator_.run_split_robs(stop_frontend_, paused_, current_lumi_block_,
                                     calo_rob_queue_, muon_rob_queue_, emulator_telemetry_,
                                     config_.mode, config_.target_rate_khz, config_.seed,
                                     config_.eb_muon_drop_rate, config_.filling_scheme,
                                     effective_pin, 0U);
        });
    } else {
        frontend_thread_ = std::jthread([this, effective_pin]() {
            emulator_.run(stop_frontend_, paused_, current_lumi_block_,
                          arena_, l1_queue_, emulator_telemetry_,
                          config_.mode, config_.target_rate_khz, config_.seed,
                          config_.filling_scheme, effective_pin, 0U);
        });
    }

    lumi_thread_ = std::jthread([this]() { lumi_block_loop(); });
}

void TriggerEngine::pause() {
    std::lock_guard<std::mutex> fsm_lock(fsm_mu_);
    RunState expected = RunState::Running;
    if (!state_.compare_exchange_strong(expected, RunState::Paused, std::memory_order_acq_rel)) {
        throw std::logic_error("TriggerEngine::pause() requires Running state");
    }
    paused_.store(true, std::memory_order_release);
}

void TriggerEngine::resume() {
    std::lock_guard<std::mutex> fsm_lock(fsm_mu_);
    RunState expected = RunState::Paused;
    if (!state_.compare_exchange_strong(expected, RunState::Running, std::memory_order_acq_rel)) {
        throw std::logic_error("TriggerEngine::resume() requires Paused state");
    }
    paused_.store(false, std::memory_order_release);
}

void TriggerEngine::stop() {
    std::lock_guard<std::mutex> fsm_lock(fsm_mu_);
    stop_locked();
}

void TriggerEngine::stop_locked() {
    const RunState st = state_.load(std::memory_order_acquire);
    if (st == RunState::Stopped) {
        return;
    }
    if (st != RunState::Running && st != RunState::Paused) {
        throw std::logic_error("TriggerEngine::stop() requires Running or Paused state");
    }

    // Step 1: Stop DetectorEmulator and join
    paused_.store(false, std::memory_order_release);
    stop_frontend_.store(true, std::memory_order_release);
    if (frontend_thread_.joinable()) {
        frontend_thread_.join();
    }

    // Step 1b: If Event Builder is active, close ROB queues so EB drains & closes l1_queue_
    if (config_.enable_event_builder) {
        calo_rob_queue_.close();
        muon_rob_queue_.close();
        if (eb_thread_.joinable()) {
            eb_thread_.join();
        }
    } else {
        l1_queue_.close();
    }

    // Step 2: L1 drains losslessly into HLT, closes hlt_queue_, and joins
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

    // Stop LumiBlock timer and finalize final LumiBlock accounting
    stop_lumi_timer_.store(true, std::memory_order_release);
    if (lumi_thread_.joinable()) {
        lumi_thread_.join();
    }

    end_time_ = std::chrono::steady_clock::now();
    state_.store(RunState::Stopped, std::memory_order_release);
}

void TriggerEngine::reset() {
    std::lock_guard<std::mutex> fsm_lock(fsm_mu_);
    const RunState st = state_.load(std::memory_order_acquire);
    if (st == RunState::Running || st == RunState::Paused) {
        stop_locked();
    }
    emulator_telemetry_.reset();
    eb_telemetry_.reset();
    l1_telemetry_.reset();
    storage_telemetry_.reset();
    {
        std::lock_guard<std::mutex> lock(lumi_mu_);
        lumi_ledger_.clear();
    }
    current_lumi_block_.store(1, std::memory_order_release);
    state_.store(RunState::Initial, std::memory_order_release);
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

void TriggerEngine::record_lumi_block(uint32_t lb_id,
                                      double start_s,
                                      double end_s,
                                      uint64_t& prev_gen,
                                      uint64_t& prev_l1,
                                      uint64_t& prev_hlt,
                                      uint64_t& prev_dt) {
    const TelemetrySnapshot snap = snapshot();
    LumiBlockRecord rec{};
    rec.lumi_block_id = lb_id;
    rec.start_elapsed_s = start_s;
    rec.end_elapsed_s = end_s;
    rec.events_generated = snap.events_generated >= prev_gen ? (snap.events_generated - prev_gen) : 0;
    rec.l1_accepted = snap.l1_accepted >= prev_l1 ? (snap.l1_accepted - prev_l1) : 0;
    rec.hlt_accepted = snap.hlt_accepted >= prev_hlt ? (snap.hlt_accepted - prev_hlt) : 0;
    rec.dead_time_events = snap.dead_time_events >= prev_dt ? (snap.dead_time_events - prev_dt) : 0;

    const uint64_t attempts = rec.events_generated + rec.dead_time_events;
    rec.live_time_fraction =
        attempts > 0 ? (1.0 - static_cast<double>(rec.dead_time_events) / static_cast<double>(attempts)) : 1.0;
    rec.num_rules = snap.num_rules;
    rec.active_prescales = snap.prescale_factors;

    prev_gen = snap.events_generated;
    prev_l1 = snap.l1_accepted;
    prev_hlt = snap.hlt_accepted;
    prev_dt = snap.dead_time_events;

    std::lock_guard<std::mutex> lock(lumi_mu_);
    lumi_ledger_.push_back(rec);
}

void TriggerEngine::lumi_block_loop() noexcept {
    const double interval_s = config_.lumi_block_interval_s > 0.05 ? config_.lumi_block_interval_s : 0.05;
    uint32_t lb_id = 1;
    double lb_start_s = 0.0;
    uint64_t prev_gen = 0;
    uint64_t prev_l1 = 0;
    uint64_t prev_hlt = 0;
    uint64_t prev_dt = 0;

    auto next_tick = start_time_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                       std::chrono::duration<double>(interval_s));

    while (!stop_lumi_timer_.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_tick) {
            const double end_s = std::chrono::duration<double>(now - start_time_).count();
            record_lumi_block(lb_id, lb_start_s, end_s, prev_gen, prev_l1, prev_hlt, prev_dt);
            ++lb_id;
            current_lumi_block_.store(lb_id, std::memory_order_release);
            lb_start_s = end_s;
            next_tick = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                  std::chrono::duration<double>(interval_s));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // Record final partial LumiBlock upon shutdown
    const double final_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();
    record_lumi_block(lb_id, lb_start_s, final_s, prev_gen, prev_l1, prev_hlt, prev_dt);
}

std::vector<LumiBlockRecord> TriggerEngine::lumi_blocks() const {
    std::lock_guard<std::mutex> lock(lumi_mu_);
    return lumi_ledger_;
}

TelemetrySnapshot TriggerEngine::snapshot() const noexcept {
    TelemetrySnapshot snap{};
    const RunState st = state_.load(std::memory_order_acquire);
    if (st == RunState::Initial) {
        return snap;
    }

    const auto now = (st == RunState::Running || st == RunState::Paused)
                         ? std::chrono::steady_clock::now()
                         : end_time_;
    const double elapsed =
        std::chrono::duration<double>(now - start_time_).count();
    snap.elapsed_seconds = elapsed > 0.0 ? elapsed : 0.0;

    snap.events_generated = emulator_telemetry_.events_generated.load(std::memory_order_relaxed);
    snap.signal_generated = emulator_telemetry_.signal_generated.load(std::memory_order_relaxed);
    snap.dead_time_ns = emulator_telemetry_.dead_time_ns.load(std::memory_order_relaxed);

    snap.eb_events_assembled = eb_telemetry_.events_assembled.load(std::memory_order_relaxed);
    snap.eb_partial_events = eb_telemetry_.partial_events.load(std::memory_order_relaxed);
    snap.eb_timeout_events = eb_telemetry_.timeout_events.load(std::memory_order_relaxed);
    snap.eb_late_fragments_dropped =
        eb_telemetry_.late_fragments_dropped.load(std::memory_order_relaxed);

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

        for (size_t b = 0; b < DQM_DIMUON_BINS; ++b) {
            snap.dqm_dimuon_mass_hist[b] += wt.dqm_dimuon_mass_hist[b].load(std::memory_order_relaxed);
        }
        for (size_t b = 0; b < DQM_JET_ET_BINS; ++b) {
            snap.dqm_jet_et_hist[b] += wt.dqm_jet_et_hist[b].load(std::memory_order_relaxed);
        }
        for (size_t b = 0; b < DQM_CALO_CELLS; ++b) {
            snap.dqm_calo_occupancy[b] += wt.dqm_calo_occupancy[b].load(std::memory_order_relaxed);
        }

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
    snap.raw_bytes_uncompressed =
        storage_telemetry_.raw_bytes_uncompressed.load(std::memory_order_relaxed);
    snap.events_written = storage_telemetry_.events_written.load(std::memory_order_relaxed);
    snap.storage_write_errors = storage_telemetry_.storage_write_errors.load(std::memory_order_relaxed);
    snap.compression_ratio =
        snap.bytes_written > 0
            ? (static_cast<double>(snap.raw_bytes_uncompressed) / static_cast<double>(snap.bytes_written))
            : 1.0;

    snap.current_lumi_block = current_lumi_block_.load(std::memory_order_acquire);
    snap.arena_allocated = arena_.allocated_count();
    snap.arena_peak = arena_.peak_occupancy();
    snap.arena_capacity = arena_.capacity();

    snap.l1_queue_fill = l1_queue_.fill_ratio();
    snap.hlt_queue_fill = hlt_queue_.fill_ratio();
    snap.storage_queue_fill = storage_queue_.fill_ratio();

    const uint32_t l1_num_rules = l1_telemetry_.active_menu_num_rules.load(std::memory_order_acquire);
    snap.num_rules = l1_num_rules > 0 ? l1_num_rules : menu_slot_.load().num_rules;
    for (size_t i = 0; i < TriggerMenu::MAX_RULES; ++i) {
        if (i < snap.num_rules) {
            snap.prescale_factors[i] = l1_telemetry_.active_prescales[i].load(std::memory_order_relaxed);
            snap.rule_accept_counts[i] = l1_telemetry_.rule_accept_counts[i].load(std::memory_order_relaxed);
        } else {
            snap.prescale_factors[i] = 1.0f;
            snap.rule_accept_counts[i] = 0;
        }
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
