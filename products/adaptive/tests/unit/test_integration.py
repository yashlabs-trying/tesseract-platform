"""M4: Foundry, Fusion and Ignite integration contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive.errors import ConfigurationError
from adaptive.integration import (
    IgniteLoadRequest,
    InProcessIgnite,
    PlacementCandidate,
    PlacementConstraints,
    PlacementRequest,
    export_profile_comparison,
    import_performance_profile,
    measurement_to_cost,
    plan_replica_transition,
    project_capacity,
    recommend_precision_for_placement,
    report_memory_requirements,
)
from adaptive.models.controller import QualityCertificate
from adaptive.models.cost import InferenceCost
from adaptive.models.enums import EvidenceType, QuantMethod, RuntimeBackend
from adaptive.models.optimization import (
    CandidateEvaluation,
    OptimizationConstraints,
    OptimizationResult,
)
from adaptive.models.quantization import PrecisionProfile, int_config
from adaptive.models.storage import StorageBreakdown

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "foundry"
    / "performance-profile-example.json"
)


def _profile(profile_id: str, bits: int, name: str) -> PrecisionProfile:
    return PrecisionProfile(
        profile_id=profile_id,
        name=name,
        default_config=int_config(bits),
        runtime=RuntimeBackend.cuda,
        method=QuantMethod.gptq,
    )


def _candidate(profile_id: str, bits: int, cost: float) -> CandidateEvaluation:
    return CandidateEvaluation(
        profile=_profile(profile_id, bits, f"uniform-{bits}"),
        storage=StorageBreakdown(total_bytes=int(cost * 1e9), baseline_bytes=8_000_000_000),
        cost=InferenceCost(lower_cost=cost, evidence=EvidenceType.measured),
        predicted_quality=1.0,
        predicted_degradation=0.01 * bits,
        feasible=True,
    )


def _optimization_result() -> OptimizationResult:
    return OptimizationResult(
        model_id="m",
        structural_fingerprint="fp",
        constraints=OptimizationConstraints(runtime=RuntimeBackend.cuda, method=QuantMethod.gptq),
        candidates=(_candidate("p4", 4, 1.0), _candidate("p8", 8, 2.0)),
        pareto_frontier=("p4", "p8"),
        selected_profile_id="p4",
        uniform_4_profile_id="p4",
        uniform_8_profile_id="p8",
    )


# ---- Foundry ------------------------------------------------------------


def test_foundry_import_measurement() -> None:
    document = json.loads(FIXTURE.read_text())
    measurement = import_performance_profile(document)
    assert measurement.profile_id == "foundry-profile-qwen2-0.5b-int4"
    assert measurement.model_id == "Qwen/Qwen2.5-0.5B"
    assert measurement.ttft_p95_ms == 71.0
    assert measurement.throughput_tps == 1850.0
    assert measurement.evidence is EvidenceType.measured
    assert measurement.provenance.source == "foundry"


def test_foundry_rejects_wrong_schema() -> None:
    with pytest.raises(ConfigurationError):
        import_performance_profile({"schema_version": "nope", "profile_id": "x"})


def test_foundry_rejects_missing_fields() -> None:
    with pytest.raises(ConfigurationError):
        import_performance_profile(
            {"schema_version": "tesseract.performance-profile.v1", "profile_id": "x"}
        )


def test_foundry_measurement_to_cost() -> None:
    document = json.loads(FIXTURE.read_text())
    cost = measurement_to_cost(import_performance_profile(document))
    assert cost.evidence is EvidenceType.measured
    assert cost.estimated_latency_ms == 71.0
    assert cost.estimated_throughput_tps == 1850.0


def test_foundry_export_comparison() -> None:
    document = export_profile_comparison(_optimization_result())
    assert document["schema_version"] == "tesseract.adaptive.profile-comparison.v1"
    assert len(document["profiles"]) == 2
    for profile in document["profiles"]:
        assert 0.0 <= profile["confidence"] <= 1.0
        assert profile["provenance"]["source"] == "adaptive"


def test_foundry_throughput_field_naming() -> None:
    # canonical Foundry field
    canonical = {
        "schema_version": "tesseract.performance-profile.v1",
        "profile_id": "p",
        "model_id": "m",
        "evidence_type": "measured",
        "output_tokens_per_second": 1234.0,
    }
    assert import_performance_profile(canonical).throughput_tps == 1234.0
    # legacy alias still accepted
    legacy = dict(canonical)
    del legacy["output_tokens_per_second"]
    legacy["throughput_tps"] = 999.0
    assert import_performance_profile(legacy).throughput_tps == 999.0


# ---- Fusion -------------------------------------------------------------


def test_fusion_memory_requirements(tiny_architecture) -> None:
    from adaptive.quantize import uniform_profile

    profile = uniform_profile(tiny_architecture, 4, group_size=16)
    requirement = report_memory_requirements(tiny_architecture, profile)
    assert requirement.weight_bytes > 0
    assert requirement.per_replica_gb > 0.0
    assert requirement.evidence is EvidenceType.estimated


def test_fusion_project_capacity() -> None:
    assert project_capacity(1_000_000_000, 24.0, 0.1) == 21
    assert project_capacity(0, 24.0, 0.1) == 0


def test_fusion_placement_recommendation() -> None:
    constraints = PlacementConstraints(
        deployment_id="d1",
        model_id="m",
        gpu_count=1,
        vram_per_gpu_gb=24.0,
        headroom_fraction=0.1,
        latency_slo_ms=50.0,
    )
    candidates = (
        PlacementCandidate(
            profile_id="p16",
            precision_bits=16,
            weight_bytes=2_000_000_000,
            estimated_latency_ms=80.0,
        ),
        PlacementCandidate(
            profile_id="p8",
            precision_bits=8,
            weight_bytes=1_000_000_000,
            estimated_latency_ms=40.0,
        ),
        PlacementCandidate(
            profile_id="p4",
            precision_bits=4,
            weight_bytes=500_000_000,
            estimated_latency_ms=25.0,
        ),
    )
    response = recommend_precision_for_placement(
        PlacementRequest(constraints=constraints, candidates=candidates)
    )
    assert response.feasible
    # 16-bit violates the latency SLO; 8-bit is the highest precision that fits
    assert response.recommended_profile_id == "p8"
    assert response.replicas_per_gpu == 21
    assert response.capacity_improvement > 1.0


def test_fusion_placement_infeasible() -> None:
    constraints = PlacementConstraints(
        deployment_id="d1",
        model_id="m",
        gpu_count=1,
        vram_per_gpu_gb=1.0,
        headroom_fraction=0.1,
        latency_slo_ms=10.0,
    )
    candidates = (
        PlacementCandidate(profile_id="p4", precision_bits=4, weight_bytes=2_000_000_000),
    )
    response = recommend_precision_for_placement(
        PlacementRequest(constraints=constraints, candidates=candidates)
    )
    assert not response.feasible
    assert response.recommended_profile_id is None


# ---- Ignite -------------------------------------------------------------


def test_ignite_loads_prevalidated_profile() -> None:
    ignite = InProcessIgnite(validated_profiles={"p4"})
    response = ignite.load_profile(
        IgniteLoadRequest(deployment_id="d1", model_id="m", profile_id="p4", replica_count=3)
    )
    assert response.accepted
    assert response.transition is not None
    assert response.transition.in_place is False
    assert response.transition.max_unavailable == 0
    assert response.readiness_signal == "ready"
    assert response.rollback_signal == "rollback"


def test_ignite_rejects_unvalidated_profile() -> None:
    ignite = InProcessIgnite()
    response = ignite.load_profile(
        IgniteLoadRequest(deployment_id="d1", model_id="m", profile_id="p4")
    )
    assert not response.accepted
    assert response.transition is None


def test_ignite_accepts_certified_profile() -> None:
    ignite = InProcessIgnite()
    certificate = QualityCertificate(
        profile_id="p8", metric="perplexity", degradation=0.01, validated=True
    )
    response = ignite.load_profile(
        IgniteLoadRequest(
            deployment_id="d1", model_id="m", profile_id="p8", certificate=certificate
        )
    )
    assert response.accepted


def test_ignite_transition_plan_is_safe() -> None:
    transition = plan_replica_transition("p16", "p4", replica_count=4)
    assert transition.in_place is False
    assert transition.max_unavailable == 0
    assert transition.steps[0].value == "drain"
    assert transition.steps[-1].value == "verify"
