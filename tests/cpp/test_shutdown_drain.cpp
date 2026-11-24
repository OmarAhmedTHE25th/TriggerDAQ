#include <cmath>
#include <cstdlib>
#include <iostream>

#include "triggerdaq/core/telemetry.hpp"
#include "triggerdaq/engine.hpp"

using namespace triggerdaq;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "CHECK FAILED at " << __FILE__ << ":" << __LINE__ << ": " #cond \
                      << "\n";                                                           \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

int main() {
    // 1. Histogram within-bucket interpolation accuracy & [min, max] clamping
    {
        LatencyHistogram hist;
        hist.reset();
        // Record uniform values 100..200 (101 samples, exact median = 150)
        for (uint64_t v = 100; v <= 200; ++v) {
            hist.record(v);
        }
        const LatencyStats s = hist.compute_stats();
        CHECK(s.min == 100.0);
        CHECK(s.max == 200.0);
        CHECK(s.p50 >= s.min && s.p50 <= s.max);
        CHECK(s.p95 >= s.p50 && s.p95 <= s.max);
        CHECK(s.p99 >= s.p95 && s.p99 <= s.max);
        CHECK(s.p999 >= s.p99 && s.p999 <= s.max);
        // Within-bucket interpolation should place p50 very close to 150 (within +-5)
        CHECK(std::fabs(s.p50 - 150.0) <= 5.0);
    }

    // 2. Lossless ordered shutdown under maximum backpressure (small queues + stress mode + pinning)
    {
        EngineConfig cfg{};
        cfg.arena_slots = 4096;
        cfg.l1_queue_capacity = 512;
        cfg.hlt_queue_capacity = 256;
        cfg.storage_queue_capacity = 256;
        cfg.hlt_workers = 4;
        cfg.mode = EmulatorMode::MaxThroughputStress;
        cfg.pin_threads = true;
        cfg.seed = 2026;
        cfg.output_path = "";

        TriggerMenu menu = TriggerMenu::default_nominal_menu();
        TriggerEngine engine;
        engine.initialize(cfg, menu);

        // Run 3 consecutive start/stop cycles verifying lossless drain and zero leaked slots
        for (int cycle = 0; cycle < 3; ++cycle) {
            engine.run_for_seconds(0.35);
            const TelemetrySnapshot snap = engine.snapshot();

            CHECK(snap.events_generated > 10000);
            CHECK(snap.arena_allocated == 0); // Zero leaked slots
            // Zero dropped post-L1 events!
            CHECK(snap.l1_accepted == snap.hlt_accepted + snap.hlt_rejected);
            // Zero dropped post-HLT events!
            CHECK(snap.hlt_accepted == snap.events_written);
            CHECK(snap.storage_write_errors == 0);
            CHECK(snap.l1_latency_ns.min <= snap.l1_latency_ns.p50);
            CHECK(snap.l1_latency_ns.p999 <= snap.l1_latency_ns.max);
        }
    }

    // 3. Simulated disk I/O error through full TriggerEngine lifecycle
    {
        EngineConfig cfg{};
        cfg.arena_slots = 2048;
        cfg.hlt_workers = 2;
        cfg.target_rate_khz = 100.0;
        cfg.mode = EmulatorMode::Sustained;
        cfg.output_path = "/nonexistent_triggerdaq_dir/fail.daq";

        TriggerEngine engine;
        engine.initialize(cfg, TriggerMenu::default_nominal_menu());
        engine.run_for_seconds(0.2);

        const TelemetrySnapshot snap = engine.snapshot();
        CHECK(snap.arena_allocated == 0);
        CHECK(snap.storage_write_errors > 0);
        CHECK(snap.bytes_written == 0);
    }

    std::cout << "[PASS] test_shutdown_drain (lossless backpressure drain, 3 start/stop cycles, histogram interpolation, I/O error)\n";
    return 0;
}
