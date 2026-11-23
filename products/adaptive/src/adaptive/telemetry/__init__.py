"""Runtime telemetry ingestion (Milestone 3).

Measured telemetry (from a real runtime) and simulated telemetry (mock GPU
telemetry for Mac M5 development) are always labelled with an
:class:`~adaptive.models.enums.EvidenceType` and never conflated.
"""

from __future__ import annotations

from adaptive.models.controller import TelemetrySnapshot
from adaptive.telemetry.sources import (
    ConstantTelemetry,
    ScriptedTelemetry,
    TelemetrySource,
    memory_pressure_scenario,
    steady_scenario,
    traffic_burst_scenario,
)

__all__ = [
    "ConstantTelemetry",
    "ScriptedTelemetry",
    "TelemetrySnapshot",
    "TelemetrySource",
    "memory_pressure_scenario",
    "steady_scenario",
    "traffic_burst_scenario",
]
