"""Declarative Python Trigger Menu dataclasses and physics presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ._triggerdaq_ext import (
    RuleCondition,
    RulePriority,
    TriggerMenu,
    TriggerRule,
)


@dataclass
class TriggerRuleSpec:
    name: str
    condition: RuleCondition
    priority: RulePriority = RulePriority.Normal
    threshold: int = 100
    secondary_threshold: int = 0
    base_prescale: float = 1.0
    max_prescale: float = 100.0
    enabled: bool = True

    def to_cpp(self) -> TriggerRule:
        rule = TriggerRule()
        rule.name = self.name
        rule.condition = self.condition
        rule.priority = self.priority
        rule.threshold = int(self.threshold)
        rule.secondary_threshold = int(self.secondary_threshold)
        rule.base_prescale = float(self.base_prescale)
        rule.max_prescale = float(self.max_prescale)
        rule.enabled = bool(self.enabled)
        return rule


@dataclass
class TriggerMenuSpec:
    name: str
    rules: List[TriggerRuleSpec] = field(default_factory=list)
    muon_pt_threshold: int = 12
    hlt_score_threshold: int = 500
    version: int = 1

    def compile(self) -> TriggerMenu:
        if len(self.rules) > 8:
            raise ValueError(f"TriggerMenu supports at most 8 rules, got {len(self.rules)}")
        menu = TriggerMenu()
        menu.muon_pt_threshold = int(self.muon_pt_threshold)
        menu.hlt_score_threshold = int(self.hlt_score_threshold)
        menu.version = int(self.version)
        for spec in self.rules:
            if not menu.add_rule(spec.to_cpp()):
                raise RuntimeError(f"Failed to add trigger rule '{spec.name}'")
        return menu


def physics_nominal(version: int = 1) -> TriggerMenuSpec:
    """Balanced pp collision menu protecting DiMuon discovery at 1.0x prescale."""
    return TriggerMenuSpec(
        name="physics_nominal",
        muon_pt_threshold=12,
        hlt_score_threshold=500,
        version=version,
        rules=[
            TriggerRuleSpec(
                name="L1_DiMu12",
                condition=RuleCondition.DiMuon,
                priority=RulePriority.Critical,
                threshold=18,
                secondary_threshold=2,
                base_prescale=1.0,
                max_prescale=1.0,
            ),
            TriggerRuleSpec(
                name="L1_SingleMu25",
                condition=RuleCondition.SingleMuon,
                priority=RulePriority.High,
                threshold=25,
                base_prescale=1.0,
                max_prescale=10.0,
            ),
            TriggerRuleSpec(
                name="L1_JetCluster500",
                condition=RuleCondition.JetCluster,
                priority=RulePriority.Normal,
                threshold=500,
                base_prescale=1.0,
                max_prescale=25.0,
            ),
            TriggerRuleSpec(
                name="L1_TotalET1200",
                condition=RuleCondition.TotalET,
                priority=RulePriority.Low,
                threshold=1200,
                base_prescale=1.0,
                max_prescale=100.0,
            ),
        ],
    )


def heavy_ion_high_pileup(version: int = 1) -> TriggerMenuSpec:
    """High-pileup menu with raised TotalET threshold & tighter HLT score cut."""
    return TriggerMenuSpec(
        name="heavy_ion_high_pileup",
        muon_pt_threshold=15,
        hlt_score_threshold=650,
        version=version,
        rules=[
            TriggerRuleSpec(
                name="L1_DiMu15_HI",
                condition=RuleCondition.DiMuon,
                priority=RulePriority.Critical,
                threshold=22,
                secondary_threshold=2,
                base_prescale=1.0,
                max_prescale=1.0,
            ),
            TriggerRuleSpec(
                name="L1_JetCluster750",
                condition=RuleCondition.JetCluster,
                priority=RulePriority.High,
                threshold=750,
                base_prescale=1.0,
                max_prescale=15.0,
            ),
            TriggerRuleSpec(
                name="L1_TotalET2800",
                condition=RuleCondition.TotalET,
                priority=RulePriority.Low,
                threshold=2800,
                base_prescale=2.0,
                max_prescale=200.0,
            ),
        ],
    )


def luminosity_stress(version: int = 1) -> TriggerMenuSpec:
    """Aggressive low-threshold stress menu to exercise PI backpressure & HLT farm."""
    return TriggerMenuSpec(
        name="luminosity_stress",
        muon_pt_threshold=8,
        hlt_score_threshold=450,
        version=version,
        rules=[
            TriggerRuleSpec(
                name="L1_DiMu8_Crit",
                condition=RuleCondition.DiMuon,
                priority=RulePriority.Critical,
                threshold=16,
                secondary_threshold=2,
                base_prescale=1.0,
                max_prescale=1.0,
            ),
            TriggerRuleSpec(
                name="L1_SingleMu12",
                condition=RuleCondition.SingleMuon,
                priority=RulePriority.High,
                threshold=12,
                base_prescale=1.0,
                max_prescale=20.0,
            ),
            TriggerRuleSpec(
                name="L1_JetCluster250",
                condition=RuleCondition.JetCluster,
                priority=RulePriority.Normal,
                threshold=250,
                base_prescale=1.0,
                max_prescale=50.0,
            ),
            TriggerRuleSpec(
                name="L1_TotalET450",
                condition=RuleCondition.TotalET,
                priority=RulePriority.Low,
                threshold=450,
                base_prescale=1.0,
                max_prescale=200.0,
            ),
        ],
    )


MENU_PRESETS = {
    "physics_nominal": physics_nominal,
    "heavy_ion_high_pileup": heavy_ion_high_pileup,
    "luminosity_stress": luminosity_stress,
}


def get_preset_menu(name: str, version: int = 1) -> TriggerMenuSpec:
    if name not in MENU_PRESETS:
        raise ValueError(f"Unknown menu preset '{name}'. Choose from: {list(MENU_PRESETS.keys())}")
    return MENU_PRESETS[name](version=version)
