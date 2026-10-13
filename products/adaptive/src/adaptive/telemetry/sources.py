"""Telemetry sources (Milestone 3).

Measured telemetry and simulated/mock telemetry are kept distinct via
:class:`~adaptive.models.enums.EvidenceType`; the controller never conflates
them. Scenarios provide deterministic telemetry for simulation and tests.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from adaptive.models.controller import TelemetrySnapshot
from adaptive.models.enums import EvidenceType


class TelemetrySource(Protocol):
    """Anything that can yield telemetry snapshots."""

    def snapshot(self) -> TelemetrySnapshot: ...


class ConstantTelemetry:
    """A source that always returns the same snapshot."""

    def __init__(self, snapshot: TelemetrySnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> TelemetrySnapshot:
        return self._snapshot


class ScriptedTelemetry:
    """A deterministic sequence of snapshots (a simulated timeline)."""

    def __init__(self, snapshots: Sequence[TelemetrySnapshot]) -> None:
        if not snapshots:
            raise ValueError("scripted telemetry requires at least one snapshot")
        self._snapshots = list(snapshots)
        self._index = 0

    def snapshot(self) -> TelemetrySnapshot:
        current = self._snapshots[min(self._index, len(self._snapshots) - 1)]
        self._index += 1
        return current

    def __iter__(self) -> Iterator[TelemetrySnapshot]:
        return iter(self._snapshots)


def steady_scenario(
    *, steps: int = 6, step_s: float = 60.0, pressure: float = 0.5, tps: float = 200.0
) -> list[TelemetrySnapshot]:
    """A stable, healthy timeline."""
    return [
        TelemetrySnapshot(
            timestamp_s=index * step_s,
            memory_pressure=pressure,
            memory_used_gb=pressure * 80.0,
            memory_total_gb=80.0,
            queue_depth=1,
            ttft_ms=120.0,
            itl_ms=20.0,
            tokens_per_second=tps,
            model_demand=10.0,
            evidence=EvidenceType.synthetic,
        )
        for index in range(steps)
    ]


def memory_pressure_scenario(
    *, step_s: float = 60.0, baseline: float = 0.5, peak: float = 0.97
) -> list[TelemetrySnapshot]:
    """A timeline that ramps into memory pressure and then recovers."""
    pressures = [baseline, baseline, 0.75, peak, peak, 0.7, baseline, baseline]
    return [
        TelemetrySnapshot(
            timestamp_s=index * step_s,
            memory_pressure=pressure,
            memory_used_gb=pressure * 80.0,
            memory_total_gb=80.0,
            queue_depth=2 if pressure < 0.9 else 12,
            ttft_ms=120.0 if pressure < 0.9 else 900.0,
            itl_ms=20.0 if pressure < 0.9 else 80.0,
            tokens_per_second=200.0 if pressure < 0.9 else 60.0,
            model_demand=10.0 if pressure < 0.9 else 40.0,
            evidence=EvidenceType.synthetic,
        )
        for index, pressure in enumerate(pressures)
    ]


def traffic_burst_scenario(
    *, step_s: float = 30.0, burst_demand: float = 60.0
) -> list[TelemetrySnapshot]:
    """A timeline with a sudden request burst that raises TTFT/ITL."""
    demands = [5.0, 5.0, burst_demand, burst_demand, burst_demand, 5.0, 5.0]
    snapshots: list[TelemetrySnapshot] = []
    for index, demand in enumerate(demands):
        burst = demand >= burst_demand
        snapshots.append(
            TelemetrySnapshot(
                timestamp_s=index * step_s,
                memory_pressure=0.92 if burst else 0.5,
                memory_used_gb=(0.92 if burst else 0.5) * 80.0,
                memory_total_gb=80.0,
                queue_depth=40 if burst else 2,
                ttft_ms=1200.0 if burst else 120.0,
                itl_ms=95.0 if burst else 20.0,
                tokens_per_second=70.0 if burst else 200.0,
                model_demand=demand,
                evidence=EvidenceType.synthetic,
            )
        )
    return snapshots
