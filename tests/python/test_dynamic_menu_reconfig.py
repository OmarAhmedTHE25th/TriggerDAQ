"""Tests for zero-downtime atomic runtime TriggerMenu reconfiguration."""

import time

import triggerdaq as td


def test_runtime_menu_hot_reload():
    cfg = td.EngineConfig()
    cfg.arena_slots = 32768
    cfg.hlt_workers = 2
    cfg.target_rate_khz = 150.0
    cfg.mode = td.EmulatorMode.Sustained
    cfg.seed = 303

    # Start with nominal menu (accepts DiMuon, Jet, TotalET)
    nominal = td.physics_nominal(version=1)
    engine = td.TriggerEngine()
    engine.initialize(cfg, nominal.compile())
    engine.start()

    time.sleep(0.25)
    snap_before = engine.snapshot()
    assert snap_before.l1_accepted > 0
    assert snap_before.hlt_accepted > 0

    # Hot-reload to a strict menu where all rules are disabled -> L1 accepts must freeze
    strict = td.TriggerMenuSpec(
        name="all_disabled",
        muon_pt_threshold=12,
        hlt_score_threshold=999,
        version=2,
        rules=[
            td.TriggerRuleSpec(
                name="DisabledRule",
                condition=td.RuleCondition.TotalET,
                priority=td.RulePriority.Low,
                threshold=999999,
                enabled=False,
            )
        ],
    )
    engine.update_menu(strict.compile())
    time.sleep(0.05)  # Let in-flight frames drain past L1

    snap_mid = engine.snapshot()
    time.sleep(0.20)
    snap_after = engine.snapshot()

    engine.stop()

    assert snap_after.events_generated > snap_mid.events_generated + 1000
    assert snap_after.l1_accepted == snap_mid.l1_accepted, (
        "Expected L1 accepts to stop after hot-reloading disabled menu"
    )
    assert engine.snapshot().arena_allocated == 0


def test_rapid_menu_swap_stress():
    """500 rapid update_menu() swaps under MaxThroughputStress to verify race-free AtomicMenuSlot."""
    cfg = td.EngineConfig()
    cfg.arena_slots = 32768
    cfg.hlt_workers = 4
    cfg.mode = td.EmulatorMode.MaxThroughputStress
    cfg.seed = 909

    m1 = td.physics_nominal(version=1).compile()
    m2 = td.heavy_ion_high_pileup(version=2).compile()

    engine = td.TriggerEngine()
    engine.initialize(cfg, m1)
    engine.start()

    for i in range(500):
        menu = td.physics_nominal(version=i + 3).compile() if (i % 2 == 0) else td.heavy_ion_high_pileup(version=i + 3).compile()
        engine.update_menu(menu)
        if i % 50 == 0:
            s = engine.snapshot()
            assert s.num_rules in (3, 4)

    engine.stop()
    final = engine.snapshot()
    assert final.events_generated > 0
    assert final.arena_allocated == 0

