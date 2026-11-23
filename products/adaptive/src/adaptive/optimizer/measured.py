"""Measured cost model (Milestone 5).

Wires measured benchmark data (imported from Foundry) into the optimizer and
controller. Profiles with a measurement use the measured cost; profiles without
one fall back to the analytic estimate, clearly marked as estimated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from adaptive.integration.foundry import BenchmarkMeasurement, measurement_to_cost
from adaptive.models.architecture import ModelArchitecture
from adaptive.models.cost import InferenceCost
from adaptive.models.enums import EvidenceType
from adaptive.models.quantization import PrecisionProfile
from adaptive.optimizer.cost import AnalyticCostModel


def measured_costs_from_measurements(
    measurements: Iterable[BenchmarkMeasurement],
) -> dict[str, InferenceCost]:
    """Build a ``profile_id -> InferenceCost`` map from measured benchmarks."""
    costs: dict[str, InferenceCost] = {}
    for measurement in measurements:
        costs[measurement.profile_id] = measurement_to_cost(measurement)
    return costs


class MeasuredCostModel:
    """Cost model preferring measured costs, with an analytic fallback."""

    def __init__(
        self,
        costs: Mapping[str, InferenceCost],
        fallback: AnalyticCostModel | None = None,
    ) -> None:
        self.costs = dict(costs)
        self.fallback = fallback

    def estimate(
        self, architecture: ModelArchitecture, profile: PrecisionProfile
    ) -> InferenceCost:
        measured = self.costs.get(profile.profile_id)
        if measured is not None:
            return measured
        if self.fallback is not None:
            return self.fallback.estimate(architecture, profile)
        analytic = AnalyticCostModel().estimate(architecture, profile)
        return analytic.model_copy(
            update={
                "evidence": EvidenceType.estimated,
                "estimated_cost_per_million_tokens": analytic.lower_cost,
            }
        )


__all__ = ["MeasuredCostModel", "measured_costs_from_measurements"]
