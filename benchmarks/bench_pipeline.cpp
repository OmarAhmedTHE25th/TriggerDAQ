#include <iomanip>
#include <iostream>
#include <string>

#include "triggerdaq/engine.hpp"
#include "triggerdaq/l1/simd_kernels.hpp"

using namespace triggerdaq;

static void run_bench(const std::string& label,
                      EmulatorMode mode,
                      double target_rate_khz,
                      uint32_t hlt_workers,
                      double duration_sec) {
    EngineConfig cfg{};
    cfg.arena_slots = 65536;
    cfg.l1_queue_capacity = 16384;
    cfg.hlt_queue_capacity = 16384;
    cfg.storage_queue_capacity = 16384;
    cfg.hlt_workers = hlt_workers;
    cfg.target_rate_khz = target_rate_khz;
    cfg.l1_budget_ns = 10000;
    cfg.mode = mode;
    cfg.pin_threads = false;
    cfg.seed = 2026;
    cfg.output_path = ""; // Null sink with full CRC32C framing

    TriggerMenu menu = TriggerMenu::default_nominal_menu();
    TriggerEngine engine;
    engine.initialize(cfg, menu);
    engine.run_for_seconds(duration_sec);

    const TelemetrySnapshot s = engine.snapshot();
    std::cout << "=== " << label << " (" << duration_sec << "s, W=" << hlt_workers << ") ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Generated:      " << s.events_generated
              << " (" << (s.input_rate_hz / 1e3) << " kHz)\n";
    std::cout << "  L1 Accepted:    " << s.l1_accepted
              << " (" << (s.l1_rate_hz / 1e3) << " kHz)\n";
    std::cout << "  HLT Accepted:   " << s.hlt_accepted
              << " (" << (s.hlt_rate_hz / 1e3) << " kHz)\n";
    std::cout << "  Dead-time:      " << s.dead_time_events
              << " (" << (s.dead_time_fraction * 100.0) << "%)\n";
    std::cout << "  L1 Latency ns:  p50=" << s.l1_latency_ns.p50
              << "  p95=" << s.l1_latency_ns.p95
              << "  p99=" << s.l1_latency_ns.p99
              << "  p99.9=" << s.l1_latency_ns.p999
              << "  max=" << s.l1_latency_ns.max << "\n";
    std::cout << "  HLT Latency us: p50=" << s.hlt_latency_us.p50
              << "  p95=" << s.hlt_latency_us.p95
              << "  p99=" << s.hlt_latency_us.p99
              << "  p99.9=" << s.hlt_latency_us.p999
              << "  max=" << s.hlt_latency_us.max << "\n";
    std::cout << "  Signal Purity:  " << (s.signal_purity * 100.0)
              << "%  |  Efficiency: " << (s.signal_efficiency * 100.0) << "%\n";
    std::cout << "  Arena Peak:     " << s.arena_peak << " / " << s.arena_capacity
              << "  |  Leaked: " << s.arena_allocated << "\n";
    std::cout << "  Storage CRC32C: " << (static_cast<double>(s.bytes_written) / (1024.0 * 1024.0))
              << " MiB (" << s.events_written << " events)\n\n";
}

int main() {
    std::cout << "TriggerDAQ C++20 Pipeline Benchmark (AVX2="
              << (has_avx2_kernel() ? "ON" : "OFF") << ")\n\n";

    run_bench("Sustained 200 kHz", EmulatorMode::Sustained, 200.0, 4, 2.0);
    run_bench("PoissonBurst 250 kHz", EmulatorMode::PoissonBurst, 250.0, 4, 2.0);
    run_bench("MaxThroughputStress", EmulatorMode::MaxThroughputStress, 0.0, 4, 3.0);
    return 0;
}
