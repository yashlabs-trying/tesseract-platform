"""Adaptive runtime controller (Milestone 3).

Consumes memory pressure, queue depth, TTFT, ITL, throughput, demand, profile
load cost and quality certificates; emits a recommended precision profile with
projected memory/cost impact, a switching plan and a rollback plan. Enforces
hysteresis, minimum dwell time, cooldowns and rate limits. Operates
recommendation-only until validated, and never performs unsafe in-place weight
replacement during active inference.
"""

from __future__ import annotations

from adaptive.controller.controller import Controller
from adaptive.controller.simulation import run_simulation
from adaptive.models.controller import (
    ControlDecision,
    ControllerAction,
    ControllerPolicy,
    ProfileOption,
    QualityCertificate,
    RollbackPlan,
    SimulationResult,
    SwitchPlan,
    SwitchStep,
    TelemetrySnapshot,
)

__all__ = [
    "ControlDecision",
    "Controller",
    "ControllerAction",
    "ControllerPolicy",
    "ProfileOption",
    "QualityCertificate",
    "RollbackPlan",
    "SimulationResult",
    "SwitchPlan",
    "SwitchStep",
    "TelemetrySnapshot",
    "run_simulation",
]
