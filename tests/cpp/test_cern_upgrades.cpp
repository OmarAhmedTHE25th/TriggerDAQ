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

        // Truncated ZS file: append a valid header + 3 trailing bytes of payload (must NOT double-count)
        {
            const std::string trunc_path = "/tmp/triggerdaq_test_zs_trunc.daq";
            std::ofstream ofs(trunc_path, std::ios::binary | std::ios::trunc);
            DaqFileHeader fh{};
            std::memcpy(fh.magic, DAQ_MAGIC, sizeof(DAQ_MAGIC));
            fh.schema_version = DAQ_SCHEMA_VERSION;
            fh.record_payload_size = DAQ_RECORD_PAYLOAD_SIZE;
            fh.flags = DAQ_FLAG_ZERO_SUPPRESSED;
            ofs.write(reinterpret_cast<const char*>(&fh), sizeof(fh));

            DaqRecordHeader rh{};
            rh.length = static_cast<uint32_t>(ZS_BASE_PAYLOAD_SIZE);
            rh.crc32c = 0;
            ofs.write(reinterpret_cast<const char*>(&rh), sizeof(rh));
            const char partial[3] = {1, 2, 3};
            ofs.write(partial, 3);
            ofs.close();

            DaqVerificationResult trunc_res = verify_daq_file(trunc_path);
            CHECK(trunc_res.header_valid);
            CHECK(trunc_res.zero_suppressed);
            CHECK(trunc_res.total_records == 1);
            CHECK(trunc_res.valid_records == 0);
            CHECK(trunc_res.corrupt_records == 1);
            std::remove(trunc_path.c_str());
        }
        std::remove(zs_path.c_str());
    }

    // 4. Event Builder late-arriving fragment after timeout (must drop without ghost duplicate)
    {
        SlabArena arena;
        arena.initialize(64);
        BoundedChannel<CaloRobFragment> calo_q(16);
        BoundedChannel<MuonRobFragment> muon_q(16);
        BoundedChannel<SlotHandle> l1_q(16);
        EventBuilder eb;
        eb.initialize(64, 500); // 500 us timeout
        EventBuilderTelemetry eb_tel{};

        std::jthread eb_thr([&]() {
            eb.run(arena, calo_q, muon_q, l1_q, eb_tel);
        });

        // Push Calo fragment for event_id=42, wait past timeout, then push late Muon fragment
        CaloRobFragment calo{};
        calo.event_id = 42;
        calo.ingest_timestamp_ns = 123456;
        calo.lumi_block_id = 1;
        CHECK(calo_q.try_push(calo));

        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        MuonRobFragment late_muon{};
        late_muon.event_id = 42;
        late_muon.muon_hits[0] = 50;
        CHECK(muon_q.try_push(late_muon));

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        calo_q.close();
        muon_q.close();
        eb_thr.join();

        // Drain emitted slot(s) from l1_q
        uint32_t popped = 0;
        SlotHandle slot = INVALID_SLOT;
        while (l1_q.try_pop(slot)) {
            RawEventFrame* f = arena.get_frame(slot);
            CHECK(f->event_id == 42);
            CHECK((f->status_flags & FRAME_STATUS_PARTIAL_EB) != 0);
            arena.release(slot);
            ++popped;
        }

        CHECK(popped == 1); // Exactly ONE event emitted (no ghost duplicate!)
        CHECK(eb_tel.events_assembled.load() == 1);
        CHECK(eb_tel.partial_events.load() == 1);
        CHECK(eb_tel.late_fragments_dropped.load() == 1);
        CHECK(arena.allocated_count() == 0);
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
