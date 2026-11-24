"""TriggerDAQ — Hybrid C++20 / Python Two-Stage Detector Trigger & DAQ Engine."""

from ._triggerdaq_ext import (
    EmulatorMode,
    EngineConfig,
    LatencyStats,
    RuleCondition,
    RulePriority,
    TelemetrySnapshot,
    TopologyTag,
    TriggerEngine,
    TriggerMenu,
    TriggerRule,
    get_record_layout,
    has_avx2_kernel,
    verify_and_map_daq,
)
from .menu import (
    MENU_PRESETS,
    TriggerMenuSpec,
    TriggerRuleSpec,
    get_preset_menu,
    heavy_ion_high_pileup,
    luminosity_stress,
    physics_nominal,
)
from .reader import DaqFileReport, DaqReader, RECORD_DTYPE

__all__ = [
    "EmulatorMode",
    "EngineConfig",
    "LatencyStats",
    "RuleCondition",
    "RulePriority",
    "TelemetrySnapshot",
    "TopologyTag",
    "TriggerEngine",
    "TriggerMenu",
    "TriggerRule",
    "get_record_layout",
    "has_avx2_kernel",
    "verify_and_map_daq",
    "TriggerRuleSpec",
    "TriggerMenuSpec",
    "MENU_PRESETS",
    "get_preset_menu",
    "physics_nominal",
    "heavy_ion_high_pileup",
    "luminosity_stress",
    "DaqFileReport",
    "DaqReader",
    "RECORD_DTYPE",
]
