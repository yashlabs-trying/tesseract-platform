"""Precision optimization (Milestone 2).

Greedy layer upgrades and a beam-search alternative minimize estimated serving
cost subject to quality, latency, memory, throughput and compatibility
constraints. Tracks the Pareto frontier and compares against uniform 4-bit and
8-bit baselines. Lower precision is not assumed to be faster; latency is
derived from bytes moved by the analytic cost model.
"""

from __future__ import annotations

from adaptive.models.optimization import (
    CandidateEvaluation,
    OptimizationConstraints,
    OptimizationResult,
)
from adaptive.optimizer.cost import AnalyticCostModel, HardwareProfile
from adaptive.optimizer.engine import optimize
from adaptive.optimizer.pareto import dominates, is_antichain, pareto_frontier
from adaptive.optimizer.quality_estimator import SensitivityQualityEstimator, bit_penalty

__all__ = [
    "AnalyticCostModel",
    "CandidateEvaluation",
    "HardwareProfile",
    "OptimizationConstraints",
    "OptimizationResult",
    "SensitivityQualityEstimator",
    "bit_penalty",
    "dominates",
    "is_antichain",
    "optimize",
    "pareto_frontier",
]
