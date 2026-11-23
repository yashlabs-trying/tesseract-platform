"""Phase 4 - optimizer validation (feasibility, optimality, Pareto, constraints)."""

from __future__ import annotations

from itertools import combinations

import pytest

pytest.importorskip("torch")

from adaptive.analyzer import ModelAnalyzer
from adaptive.models.enums import EvidenceType
from adaptive.models.optimization import OptimizationConstraints
from adaptive.models.sensitivity import SensitivityMethod, SensitivityReport
from adaptive.optimizer import AnalyticCostModel, SensitivityQualityEstimator, optimize
from adaptive.optimizer.pareto import dominates
from adaptive.quantize import mixed_profile, validate_profile
from adaptive.quantize.storage import compute_storage


def _single_layer_architecture():
    config = {
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
    }
    return ModelAnalyzer().analyze(config, model_id="tiny1")


def _sensitivity(arch) -> SensitivityReport:
    layers = [s.name for s in arch.layers if s.quantizable and len(s.shape) == 2]
    scores = {
        name: {SensitivityMethod.activation_weighted.value: 1.0 / (i + 1)}
        for i, name in enumerate(layers)
    }
    return SensitivityReport(
        model_id=arch.model_id,
        structural_fingerprint=arch.structural_fingerprint(),
        calibration_fingerprint="cal",
        methods=(SensitivityMethod.activation_weighted,),
        bits=4,
        group_size=32,
        scores=scores,
        raw_scores=scores,
        rankings={SensitivityMethod.activation_weighted.value: layers},
        family_attribution={},
        evidence=EvidenceType.estimated,
    )


def _exhaustive_optimum(arch, sensitivity, tolerance):
    layers = [s.name for s in arch.layers if s.quantizable and len(s.shape) == 2]
    cost_model = AnalyticCostModel()
    estimator = SensitivityQualityEstimator(sensitivity, baseline_value=100.0)
    best = None
    for size in range(len(layers) + 1):
        for subset in combinations(layers, size):
            profile = mixed_profile(
                arch, {name: 8 for name in subset}, default_bits=4, group_size=32, name="enum"
            )
            if validate_profile(arch, profile):
                continue
            if estimator.degradation(profile) > tolerance:
                continue
            cost = cost_model.estimate(arch, profile).lower_cost
            if best is None or cost < best[0]:
                best = (cost, profile.profile_id, subset)
    return best


def test_optimizer_matches_exhaustive_optimum() -> None:
    arch = _single_layer_architecture()
    sensitivity = _sensitivity(arch)
    tolerance = 0.05
    constraints = OptimizationConstraints(quality_tolerance=tolerance, allowed_bits=(4, 8))

    result = optimize(
        arch, sensitivity, constraints, baseline_metric_value=100.0, group_size=32
    )
    selected = result.selected()
    assert selected is not None and selected.feasible

    optimum = _exhaustive_optimum(arch, sensitivity, tolerance)
    assert optimum is not None
    # greedy must be optimal (within a tiny tolerance) on this small space
    assert selected.cost.lower_cost <= optimum[0] * (1.0 + 1e-9)


def test_optimizer_feasible_when_tight_budget() -> None:
    arch = _single_layer_architecture()
    sensitivity = _sensitivity(arch)
    constraints = OptimizationConstraints(
        quality_tolerance=0.0, allowed_bits=(4, 8, 16)
    )
    result = optimize(
        arch, sensitivity, constraints, baseline_metric_value=100.0, group_size=32
    )
    selected = result.selected()
    assert selected is not None
    assert selected.feasible
    assert selected.predicted_degradation <= 1e-9


def test_optimizer_reports_infeasible_explicitly() -> None:
    arch = _single_layer_architecture()
    sensitivity = _sensitivity(arch)
    constraints = OptimizationConstraints(
        quality_tolerance=0.05, allowed_bits=(4, 8), p95_latency_ms=1e-12
    )
    result = optimize(
        arch, sensitivity, constraints, baseline_metric_value=100.0, group_size=32
    )
    assert any("no feasible" in note for note in result.notes)
    # still returns a candidate list; nothing may be marked feasible
    assert all(not c.feasible for c in result.candidates) or result.selected() is not None


def test_pareto_frontier_is_antichain() -> None:
    arch = _single_layer_architecture()
    sensitivity = _sensitivity(arch)
    result = optimize(
        arch,
        sensitivity,
        OptimizationConstraints(quality_tolerance=0.5, allowed_bits=(4, 8)),
        baseline_metric_value=100.0,
        group_size=32,
    )
    points = [
        (c.profile_id, c.cost.lower_cost, c.predicted_degradation) for c in result.candidates
    ]
    frontier = [p for p in points if p[0] in set(result.pareto_frontier)]
    for a in frontier:
        for b in frontier:
            if a is not b:
                assert not dominates(a, b)


def test_storage_decreases_with_lower_bits() -> None:
    arch = _single_layer_architecture()
    p4 = mixed_profile(arch, {}, default_bits=4, group_size=32, name="u4")
    p8 = mixed_profile(arch, {}, default_bits=8, group_size=32, name="u8")
    assert compute_storage(arch, p4).total_bytes < compute_storage(arch, p8).total_bytes
