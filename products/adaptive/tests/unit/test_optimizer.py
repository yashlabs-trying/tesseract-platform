"""M2: constrained precision optimization and Pareto frontier."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from adaptive.errors import ConfigurationError
from adaptive.models.enums import RuntimeBackend
from adaptive.models.optimization import OptimizationConstraints
from adaptive.optimizer import optimize
from adaptive.optimizer.pareto import dominates, is_antichain, pareto_frontier
from adaptive.sensitivity import analyze_sensitivity


def test_dominates() -> None:
    a = ("a", 1.0, 1.0)
    b = ("b", 2.0, 2.0)
    c = ("c", 1.0, 3.0)
    assert dominates(a, b)
    assert not dominates(b, a)
    assert dominates(a, c)
    assert not dominates(c, a)


def test_pareto_frontier_known_set() -> None:
    points = [
        ("a", 1.0, 5.0),
        ("b", 2.0, 3.0),
        ("c", 3.0, 4.0),  # dominated by b
        ("d", 4.0, 1.0),
        ("e", 5.0, 6.0),  # dominated by a/b
    ]
    frontier = pareto_frontier(points)
    assert frontier == ["a", "b", "d"]
    frontier_points = [p for p in points if p[0] in frontier]
    assert is_antichain(frontier_points)


def test_pareto_frontier_is_antichain_property() -> None:
    points = [("x", 1.0, 1.0), ("y", 1.0, 1.0), ("z", 0.5, 2.0)]
    frontier = pareto_frontier(points)
    frontier_points = [p for p in points if p[0] in frontier]
    assert is_antichain(frontier_points)


def _sensitivity(arch, model, dataset):
    return analyze_sensitivity(
        arch,
        model,
        dataset.sequences,
        bits=4,
        group_size=16,
        max_batches=2,
        calibration_fingerprint=dataset.fingerprint(),
    )


def _optimize(arch, sensitivity, **kwargs):
    constraints = kwargs.pop("constraints", None) or OptimizationConstraints(
        quality_tolerance=kwargs.pop("tolerance", 0.5),
        allowed_bits=(4, 8, 16),
    )
    return optimize(
        arch,
        sensitivity,
        constraints,
        baseline_metric_value=100.0,
        group_size=16,
        **kwargs,
    )


def test_baselines_present(tiny_architecture, tiny_model, calibration_dataset) -> None:
    sensitivity = _sensitivity(tiny_architecture, tiny_model, calibration_dataset)
    result = _optimize(tiny_architecture, sensitivity)
    assert result.uniform_4_profile_id is not None
    assert result.uniform_8_profile_id is not None
    assert result.pareto_frontier


def test_large_tolerance_selects_cheapest(
    tiny_architecture, tiny_model, calibration_dataset
) -> None:
    sensitivity = _sensitivity(tiny_architecture, tiny_model, calibration_dataset)
    result = _optimize(tiny_architecture, sensitivity, tolerance=1.0)
    assert result.selected_profile_id == result.uniform_4_profile_id


def test_zero_tolerance_requires_baseline(
    tiny_architecture, tiny_model, calibration_dataset
) -> None:
    sensitivity = _sensitivity(tiny_architecture, tiny_model, calibration_dataset)
    result = _optimize(tiny_architecture, sensitivity, tolerance=0.0)
    selected = result.selected()
    assert selected is not None
    assert selected.feasible
    assert selected.predicted_degradation <= 1e-9
    # the least sensitive layer may legitimately remain at 4-bit
    assert 16 in selected.profile.distinct_bits()
    assert selected.profile.distinct_bits() <= {4, 16}


def test_runtime_rejection(tiny_architecture, tiny_model, calibration_dataset) -> None:
    sensitivity = _sensitivity(tiny_architecture, tiny_model, calibration_dataset)
    constraints = OptimizationConstraints(runtime=RuntimeBackend.mps, allowed_bits=(4,))
    with pytest.raises(ConfigurationError):
        optimize(tiny_architecture, sensitivity, constraints, group_size=16)


def test_latency_infeasible_falls_back_to_baseline(
    tiny_architecture, tiny_model, calibration_dataset
) -> None:
    sensitivity = _sensitivity(tiny_architecture, tiny_model, calibration_dataset)
    constraints = OptimizationConstraints(
        quality_tolerance=0.5, allowed_bits=(4, 8, 16), p95_latency_ms=1e-9
    )
    result = optimize(
        tiny_architecture, sensitivity, constraints, baseline_metric_value=100.0, group_size=16
    )
    assert result.selected() is not None
    assert any("no feasible" in note for note in result.notes)
    selected = result.selected()
    assert selected is not None and selected.profile.name.startswith("uniform-16")


def test_candidates_record_violations(tiny_architecture, tiny_model, calibration_dataset) -> None:
    sensitivity = _sensitivity(tiny_architecture, tiny_model, calibration_dataset)
    constraints = OptimizationConstraints(
        quality_tolerance=0.0, allowed_bits=(4, 8, 16)
    )
    result = optimize(
        tiny_architecture, sensitivity, constraints, baseline_metric_value=100.0, group_size=16
    )
    uniform4 = result.candidate(result.uniform_4_profile_id)  # type: ignore[arg-type]
    assert not uniform4.feasible
    assert uniform4.violations


def test_deterministic(tiny_architecture, tiny_model, calibration_dataset) -> None:
    sensitivity = _sensitivity(tiny_architecture, tiny_model, calibration_dataset)
    first = _optimize(tiny_architecture, sensitivity)
    second = _optimize(tiny_architecture, sensitivity)
    assert first.selected_profile_id == second.selected_profile_id
    assert first.pareto_frontier == second.pareto_frontier


def test_beam_algorithm_runs(tiny_architecture, tiny_model, calibration_dataset) -> None:
    sensitivity = _sensitivity(tiny_architecture, tiny_model, calibration_dataset)
    result = _optimize(tiny_architecture, sensitivity, algorithm="beam")
    assert result.algorithm == "beam"
    assert result.selected() is not None
