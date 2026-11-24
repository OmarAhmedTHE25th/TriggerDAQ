#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

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
    // 1. LHC Orbit & BCID Filling Scheme verification (abort gap + orbit wrap)
    {
        LhcFillingScheme scheme{};
        LhcOrbitClock clock{};
        uint32_t prev_orbit = 0;
        uint16_t prev_bcid = 0;
        bool wrapped = false;

        for (int i = 0; i < 10000; ++i) {
            clock.advance_to_next_filled(scheme);
            CHECK(clock.bcid < LHC_BUNCHES_PER_ORBIT);
            CHECK(!scheme.is_in_abort_gap(clock.bcid));
            CHECK(scheme.is_filled(clock.bcid));
            if (i > 0) {
                if (clock.orbit_id > prev_orbit) {
                    CHECK(clock.orbit_id == prev_orbit + 1);
                    CHECK(clock.bcid < prev_bcid);
                    wrapped = true;
                } else {
                    CHECK(clock.orbit_id == prev_orbit);
                    CHECK(clock.bcid > prev_bcid);
                }
            }
            prev_orbit = clock.orbit_id;
            prev_bcid = clock.bcid;
        }
        CHECK(wrapped);
        CHECK(clock.orbit_id >= 3);
    }

    // 2. Run Control FSM state transition validation + LumiBlock progression
    {
        TriggerEngine engine;
        CHECK(engine.state() == RunState::Initial);

        // Invalid: start() before configure()
        bool threw = false;
        try {
            engine.start();
        } catch (const std::logic_error&) {
            threw = true;
        }
        CHECK(threw);

        // Invalid: pause() before running
        threw = false;
        try {
            engine.pause();
        } catch (const std::logic_error&) {
            threw = true;
        }
        CHECK(threw);

        EngineConfig cfg{};
        cfg.arena_slots = 8192;
        cfg.hlt_workers = 2;
        cfg.target_rate_khz = 80.0;
        cfg.mode = EmulatorMode::Sustained;
        cfg.lumi_block_interval_s = 0.15;

        engine.configure(cfg, TriggerMenu::default_nominal_menu());
        CHECK(engine.state() == RunState::Configured);

        engine.start();
        CHECK(engine.state() == RunState::Running);
        std::this_thread::sleep_for(std::chrono::milliseconds(180));

        engine.pause();
        CHECK(engine.state() == RunState::Paused);
        std::this_thread::sleep_for(std::chrono::milliseconds(60));

        engine.resume();
        CHECK(engine.state() == RunState::Running);
        std::this_thread::sleep_for(std::chrono::milliseconds(180));

        engine.stop();
        CHECK(engine.state() == RunState::Stopped);

        // Invalid: pause() when Stopped
        threw = false;
        try {
            engine.pause();
        } catch (const std::logic_error&) {
            threw = true;
        }
        CHECK(threw);

        const auto lbs = engine.lumi_blocks();
        CHECK(lbs.size() >= 2);
        for (size_t i = 0; i < lbs.size(); ++i) {
            CHECK(lbs[i].lumi_block_id == static_cast<uint32_t>(i + 1));
        }

        const TelemetrySnapshot snap = engine.snapshot();
        CHECK(snap.arena_allocated == 0);
        // Verify DQM histogram sum == hlt_accepted
        uint64_t dimuon_sum = 0;
        uint64_t jet_sum = 0;
        for (auto c : snap.dqm_dimuon_mass_hist) dimuon_sum += c;
        for (auto c : snap.dqm_jet_et_hist) jet_sum += c;
        CHECK(dimuon_sum == snap.hlt_accepted);
        CHECK(jet_sum == snap.hlt_accepted);

        engine.reset();
        CHECK(engine.state() == RunState::Initial);
    }

    // 3. Zero-Suppression (ZS) round-trip + compression ratio + bit-flip detection
    {
        const std::string zs_path = "/tmp/triggerdaq_test_zs.daq";
        EngineConfig cfg{};
        cfg.arena_slots = 8192;
        cfg.hlt_workers = 2;
        cfg.target_rate_khz = 120.0;
        cfg.mode = EmulatorMode::Sustained;
        cfg.enable_zero_suppression = true;
        cfg.zs_pedestal_threshold = 20;
        cfg.output_path = zs_path;

        TriggerEngine engine;
        engine.configure(cfg, TriggerMenu::default_nominal_menu());
        engine.run_for_seconds(0.35);

        const TelemetrySnapshot snap = engine.snapshot();
        CHECK(snap.arena_allocated == 0);
        CHECK(snap.events_written > 50);
        CHECK(snap.compression_ratio > 1.35);

        DaqVerificationResult res = verify_daq_file(zs_path);
        CHECK(res.header_valid);
        CHECK(res.zero_suppressed);
        CHECK(res.total_records == snap.events_written);
        CHECK(res.valid_records == snap.events_written);
        CHECK(res.corrupt_records == 0);

        // Corrupt a single bit inside record #2 payload and verify CRC32C catches it
        {
            std::fstream fs(zs_path, std::ios::binary | std::ios::in | std::ios::out);
            CHECK(fs.is_open());
            fs.seekg(static_cast<std::streamoff>(sizeof(DaqFileHeader) + 20));
            char b = 0;
            fs.read(&b, 1);
            b ^= 0x04;
            fs.seekp(static_cast<std::streamoff>(sizeof(DaqFileHeader) + 20));
            fs.write(&b, 1);
            fs.flush();
        }
        DaqVerificationResult corrupt_res = verify_daq_file(zs_path);
        CHECK(corrupt_res.header_valid);
        CHECK(corrupt_res.corrupt_records >= 1);
        std::remove(zs_path.c_str());
    }

    // 4. Event Builder 1% Muon ROB fault-injection test (zero leaked SlabArena slots)
    {
        EngineConfig cfg{};
        cfg.arena_slots = 8192;
        cfg.l1_queue_capacity = 2048;
        cfg.hlt_queue_capacity = 2048;
        cfg.storage_queue_capacity = 2048;
        cfg.hlt_workers = 2;
        cfg.target_rate_khz = 80.0;
        cfg.mode = EmulatorMode::Sustained;
        cfg.enable_event_builder = true;
        cfg.eb_timeout_us = 250;
        cfg.eb_muon_drop_rate = 0.01; // Drop 1% of Muon ROB fragments
        cfg.seed = 2026;

        TriggerEngine engine;
        engine.configure(cfg, TriggerMenu::default_nominal_menu());
        engine.run_for_seconds(0.45);

        const TelemetrySnapshot snap = engine.snapshot();
        CHECK(snap.arena_allocated == 0); // Zero leaked slots even under 1% ROB drops!
        CHECK(snap.eb_events_assembled > 5000);
        CHECK(snap.eb_partial_events > 20);
        CHECK(snap.eb_timeout_events > 0);
        CHECK(snap.l1_accepted == snap.hlt_accepted + snap.hlt_rejected);
        CHECK(snap.hlt_accepted == snap.events_written);
    }

    std::cout << "[PASS] test_cern_upgrades (LHC BCID/orbit, FSM, LumiBlocks, DQM, ZS codec, Event Builder 1% fault-injection)\n";
    return 0;
}
