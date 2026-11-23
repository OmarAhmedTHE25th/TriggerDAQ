"""Tests for nanobind TriggerEngine lifecycle, GIL-free execution, and telemetry snapshots."""

import threading
import time

import triggerdaq as td


def test_engine_lifecycle_and_gil_release():
    cfg = td.EngineConfig()
    cfg.arena_slots = 16384
    cfg.hlt_workers = 2
    cfg.target_rate_khz = 100.0
    cfg.mode = td.EmulatorMode.Sustained
    cfg.seed = 101

    menu = td.physics_nominal().compile()
    engine = td.TriggerEngine()
    engine.initialize(cfg, menu)
    assert not engine.is_running()

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
        engine.run_for_seconds(0.4)
    finally:
        stop_ticker.set()
        t.join()

    assert py_ticks >= 10, f"Expected Python thread to run freely while C++ executes (ticks={py_ticks})"
    assert not engine.is_running()

    snap = engine.snapshot()
    assert snap.events_generated > 5000
    assert snap.l1_accepted > 0
    assert snap.hlt_accepted > 0
    assert snap.arena_allocated == 0, "Zero leaked SlabArena slots after graceful stop"
    assert snap.signal_purity > 0.90, f"Expected high HLT signal purity, got {snap.signal_purity:.3f}"
    assert snap.signal_efficiency > 0.85, f"Expected high HLT signal efficiency, got {snap.signal_efficiency:.3f}"

    d = snap.to_dict()
    assert d["events_generated"] == snap.events_generated
    assert "l1_latency_ns" in d and "hlt_latency_us" in d


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
