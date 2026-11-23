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
