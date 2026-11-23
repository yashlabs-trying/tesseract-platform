"""End-to-end Mac pipeline: M1 analyze -> M2 optimize -> M3 controller -> M4 adapters."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from adaptive.controller import ControllerPolicy, ProfileOption, run_simulation
from adaptive.integration import (
    IgniteLoadRequest,
    InProcessIgnite,
    PlacementCandidate,
    PlacementConstraints,
    PlacementRequest,
    export_profile_comparison,
    recommend_precision_for_placement,
    report_memory_requirements,
)
from adaptive.models.optimization import OptimizationConstraints
from adaptive.models.quality import QualityMetric
from adaptive.optimizer import optimize
from adaptive.quality import run_quality_suite
from adaptive.quantize import uniform_profile
from adaptive.sensitivity import analyze_sensitivity
from adaptive.telemetry import memory_pressure_scenario


def test_end_to_end_mac(
    tiny_architecture, tiny_model, calibration_dataset, evaluation_dataset
) -> None:
    report = analyze_sensitivity(
        tiny_architecture,
        tiny_model,
        calibration_dataset.sequences,
        bits=4,
        group_size=16,
        calibration_fingerprint=calibration_dataset.fingerprint(),
        max_batches=2,
    )

    baseline_quality = run_quality_suite(
        tiny_architecture,
        tiny_model,
        uniform_profile(tiny_architecture, 16, group_size=16),
        calibration=calibration_dataset,
        evaluation=evaluation_dataset,
        suite=(QualityMetric.perplexity,),
        max_batches=2,
    )

    result = optimize(
        tiny_architecture,
        report,
        OptimizationConstraints(quality_tolerance=0.5, allowed_bits=(4, 8, 16)),
        baseline_metric_value=baseline_quality.metric(QualityMetric.perplexity).value,
        group_size=16,
    )
    assert result.selected() is not None

    # M4 - Foundry export
    comparison = export_profile_comparison(result)
    assert comparison["profiles"]

    # M4 - Fusion placement from optimizer candidates
    placement_candidates = tuple(
        PlacementCandidate(
            profile_id=candidate.profile_id,
            precision_bits=max(candidate.profile.distinct_bits()),
            weight_bytes=candidate.storage.total_bytes,
            estimated_latency_ms=candidate.cost.estimated_latency_ms,
            estimated_throughput_tps=candidate.cost.estimated_throughput_tps,
        )
        for candidate in result.candidates
    )
    placement = recommend_precision_for_placement(
        PlacementRequest(
            constraints=PlacementConstraints(
                deployment_id="d1",
                model_id=tiny_architecture.model_id,
                gpu_count=1,
                vram_per_gpu_gb=24.0,
                headroom_fraction=0.5,
            ),
            candidates=placement_candidates,
        )
    )
    assert placement.feasible
    assert placement.recommended_profile_id is not None

    # M4 - Fusion memory requirement for the selected profile
    selected = result.selected()
    assert selected is not None
    requirement = report_memory_requirements(tiny_architecture, selected.profile)
    assert requirement.per_replica_gb > 0.0

    # M3 - controller options from the optimizer's Pareto candidates
    options = [
        ProfileOption(
            profile_id=candidate.profile_id,
            precision_bits=max(candidate.profile.distinct_bits()),
            estimated_cost=candidate.cost.lower_cost,
            weight_bytes=candidate.storage.total_bytes,
            quality_degradation=candidate.predicted_degradation,
        )
        for candidate in result.candidates
    ]
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0, quality_tolerance=0.5)
    simulation = run_simulation(
        policy, options, memory_pressure_scenario(), automatic=True
    )
    assert simulation.switch_count >= 0

    # M4 - Ignite loads the selected profile (prevalidated)
    ignite = InProcessIgnite(validated_profiles={selected.profile_id})
    load = ignite.load_profile(
        IgniteLoadRequest(
            deployment_id="d1",
            model_id=tiny_architecture.model_id,
            profile_id=selected.profile_id,
            replica_count=2,
        )
    )
    assert load.accepted
    assert load.transition is not None and load.transition.in_place is False
