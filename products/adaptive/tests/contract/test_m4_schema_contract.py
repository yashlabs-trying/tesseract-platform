"""Contract tests for M4 integration schemas."""

from __future__ import annotations

import pytest

jsonschema = pytest.importorskip("jsonschema")

from adaptive.integration import (
    IgniteLoadRequest,
    InProcessIgnite,
    PlacementCandidate,
    PlacementConstraints,
    PlacementRequest,
    export_profile_comparison,
    load_document,
    placement_document,
    recommend_precision_for_placement,
)
from adaptive.models.cost import InferenceCost
from adaptive.models.enums import EvidenceType, QuantMethod, RuntimeBackend
from adaptive.models.optimization import (
    CandidateEvaluation,
    OptimizationConstraints,
    OptimizationResult,
)
from adaptive.models.quantization import PrecisionProfile, int_config
from adaptive.models.storage import StorageBreakdown
from adaptive.schemas import load_schema


def _validate(document: dict, schema_name: str) -> None:
    jsonschema.validate(instance=document, schema=dict(load_schema(schema_name)))


def _result() -> OptimizationResult:
    profile = PrecisionProfile(
        profile_id="p4",
        name="uniform-4",
        default_config=int_config(4),
        runtime=RuntimeBackend.cuda,
        method=QuantMethod.gptq,
    )
    candidate = CandidateEvaluation(
        profile=profile,
        storage=StorageBreakdown(total_bytes=4_000_000_000, baseline_bytes=16_000_000_000),
        cost=InferenceCost(lower_cost=1.0, evidence=EvidenceType.estimated),
        predicted_quality=1.0,
        predicted_degradation=0.05,
        feasible=True,
    )
    return OptimizationResult(
        model_id="m",
        structural_fingerprint="fp",
        constraints=OptimizationConstraints(runtime=RuntimeBackend.cuda, method=QuantMethod.gptq),
        candidates=(candidate,),
        pareto_frontier=("p4",),
        selected_profile_id="p4",
    )


def test_profile_comparison_contract() -> None:
    _validate(export_profile_comparison(_result()), "profile-comparison-v1.json")


def test_placement_recommendation_contract() -> None:
    response = recommend_precision_for_placement(
        PlacementRequest(
            constraints=PlacementConstraints(
                deployment_id="d1", model_id="m", gpu_count=1, vram_per_gpu_gb=24.0
            ),
            candidates=(
                PlacementCandidate(profile_id="p4", precision_bits=4, weight_bytes=1_000_000_000),
            ),
        )
    )
    _validate(placement_document(response), "placement-recommendation-v1.json")


def test_ignite_load_contract() -> None:
    ignite = InProcessIgnite(validated_profiles={"p4"})
    response = ignite.load_profile(
        IgniteLoadRequest(deployment_id="d1", model_id="m", profile_id="p4")
    )
    _validate(load_document(response), "ignite-profile-load-v1.json")


def test_m4_schema_ids() -> None:
    for name, expected in (
        ("profile-comparison-v1.json", "tesseract.adaptive.profile-comparison.v1"),
        ("placement-recommendation-v1.json", "tesseract.adaptive.placement-recommendation.v1"),
        ("ignite-profile-load-v1.json", "tesseract.adaptive.ignite-profile-load.v1"),
    ):
        assert load_schema(name)["$id"] == expected
