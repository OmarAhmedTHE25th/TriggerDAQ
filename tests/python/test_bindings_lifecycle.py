"""Tests for nanobind TriggerEngine lifecycle, GIL-free execution, and telemetry snapshots."""

import threading
import time

import pytest
import triggerdaq as td


def test_engine_lifecycle_and_gil_release():
    cfg = td.EngineConfig()
    cfg.arena_slots = 16384
    cfg.hlt_workers = 2
    cfg.target_rate_khz = 100.0
    cfg.mode = td.EmulatorMode.Sustained
    cfg.lumi_block_interval_s = 0.2
    cfg.seed = 101

    menu = td.physics_nominal().compile()
    engine = td.TriggerEngine()
    assert engine.state() == td.RunState.Initial
    assert not engine.is_running()

    # Invalid FSM transitions before configure()
    with pytest.raises(RuntimeError):
        engine.start()
    with pytest.raises(RuntimeError):
        engine.pause()

    engine.configure(cfg, menu)
    assert engine.state() == td.RunState.Configured

    # Verify GIL is released during run_for_seconds by running a concurrent Python thread
    py_ticks = 0
    stop_ticker = threading.Event()

    def ticker():
        nonlocal py_ticks
        while not stop_ticker.is_set():
            py_ticks += 1
            time.sleep(0.005)

    t = threading.Thread(target=ticker)
    t.start()
    try:
        engine.run_for_seconds(0.45)
    finally:
        stop_ticker.set()
        t.join()

    assert py_ticks >= 10, f"Expected Python thread to run freely while C++ executes (ticks={py_ticks})"
    assert not engine.is_running()
    assert engine.state() == td.RunState.Stopped

    with pytest.raises(RuntimeError):
        engine.pause()

    snap = engine.snapshot()
    assert snap.events_generated > 5000
    assert snap.l1_accepted > 0
    assert snap.hlt_accepted > 0
    assert snap.arena_allocated == 0, "Zero leaked SlabArena slots after graceful stop"
    assert snap.signal_purity > 0.90, f"Expected high HLT signal purity, got {snap.signal_purity:.3f}"
    assert snap.signal_efficiency > 0.85, f"Expected high HLT signal efficiency, got {snap.signal_efficiency:.3f}"

    # Verify DQM histograms sum to hlt_accepted and exhibit a clear Di-Muon mass peak (>40 GeV)
    dimuon_hist = list(snap.dqm_dimuon_mass_hist)
    jet_hist = list(snap.dqm_jet_et_hist)
    assert sum(dimuon_hist) == snap.hlt_accepted
    assert sum(jet_hist) == snap.hlt_accepted
    high_mass_counts = sum(dimuon_hist[8:26])  # 40..130 GeV range
    assert high_mass_counts > 0, "Expected clear Di-Muon peak in 40-130 GeV bins"

    lbs = engine.lumi_blocks()
    assert len(lbs) >= 2
    assert [lb.lumi_block_id for lb in lbs] == list(range(1, len(lbs) + 1))

    d = snap.to_dict()
    assert d["events_generated"] == snap.events_generated
    assert "l1_latency_ns" in d and "hlt_latency_us" in d
    assert "dqm_dimuon_mass_hist" in d


def test_lock_free_telemetry_polling():
    cfg = td.EngineConfig()
    cfg.arena_slots = 16384
    cfg.hlt_workers = 2
    cfg.mode = td.EmulatorMode.MaxThroughputStress
    cfg.seed = 202

    engine = td.TriggerEngine()
    engine.initialize(cfg, td.physics_nominal().compile())
    engine.start()
    assert engine.is_running()

    prev = 0
    for _ in range(10):
        time.sleep(0.02)
        s = engine.snapshot()
        assert s.events_generated >= prev
        prev = s.events_generated

    engine.stop()
    assert not engine.is_running()
    final = engine.snapshot()
    assert final.events_generated > 0
    assert final.arena_allocated == 0
