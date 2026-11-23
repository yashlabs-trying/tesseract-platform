"""Fusion integration adapter (Milestone 4).

Reports profile memory requirements, recommends precision under placement
constraints, and projects capacity improvements. Fusion remains independently
usable; this adapter is a pure contract translation with no network dependency.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

from adaptive.models.architecture import ModelArchitecture
from adaptive.models.enums import EvidenceType
from adaptive.models.quantization import PrecisionProfile
from adaptive.quantize.storage import compute_storage

PLACEMENT_RECOMMENDATION_SCHEMA = "tesseract.adaptive.placement-recommendation.v1"


class MemoryRequirement(BaseModel):
    """Serialized memory footprint of one profile replica."""

    model_config = {"frozen": True}

    profile_id: str
    weight_bytes: int
    overhead_bytes: int
    per_replica_gb: float
    min_vram_gb: float
    evidence: EvidenceType = EvidenceType.estimated


class PlacementCandidate(BaseModel):
    """A profile option offered to Fusion for placement."""

    model_config = {"frozen": True}

    profile_id: str
    precision_bits: int
    weight_bytes: int
    estimated_latency_ms: float = 0.0
    estimated_throughput_tps: float = 0.0
    quality_degradation: float = 0.0


class PlacementConstraints(BaseModel):
    """Placement constraints supplied by Fusion."""

    model_config = {"frozen": True}

    deployment_id: str
    model_id: str
    gpu_count: int = Field(ge=1)
    vram_per_gpu_gb: float = Field(gt=0.0)
    headroom_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    latency_slo_ms: float | None = None
    min_throughput_tps: float = 0.0


class PlacementRequest(BaseModel):
    """A placement recommendation request."""

    model_config = {"frozen": True}

    constraints: PlacementConstraints
    candidates: tuple[PlacementCandidate, ...]


class PlacementResponse(BaseModel):
    """A placement recommendation with projected capacity improvement."""

    model_config = {"frozen": True}

    deployment_id: str
    model_id: str
    recommended_profile_id: str | None
    replicas_per_gpu: int = 0
    projected_capacity: int = 0
    baseline_capacity: int = 0
    capacity_improvement: float = 0.0
    feasible: bool = False
    reason: str = ""
    evidence: EvidenceType = EvidenceType.estimated


def report_memory_requirements(
    architecture: ModelArchitecture, profile: PrecisionProfile
) -> MemoryRequirement:
    """Serialized memory footprint of ``profile`` for ``architecture``."""
    storage = compute_storage(architecture, profile)
    return MemoryRequirement(
        profile_id=profile.profile_id,
        weight_bytes=storage.parameter_bytes,
        overhead_bytes=storage.overhead_bytes,
        per_replica_gb=storage.total_bytes / 1e9,
        min_vram_gb=storage.total_bytes / 1e9,
    )


def project_capacity(weight_bytes: int, vram_gb: float, headroom_fraction: float = 0.1) -> int:
    """Number of replicas that fit in ``vram_gb`` with headroom."""
    usable = vram_gb * 1e9 * (1.0 - headroom_fraction)
    if weight_bytes <= 0:
        return 0
    return max(int(math.floor(usable / weight_bytes)), 0)


def recommend_precision_for_placement(request: PlacementRequest) -> PlacementResponse:
    """Pick the highest-precision profile that fits and meets SLOs."""
    constraints = request.constraints
    usable_per_gpu = constraints.vram_per_gpu_gb * (1.0 - constraints.headroom_fraction)

    feasible: list[tuple[PlacementCandidate, int]] = []
    for candidate in request.candidates:
        replicas = project_capacity(
            candidate.weight_bytes, constraints.vram_per_gpu_gb, constraints.headroom_fraction
        )
        if replicas < 1:
            continue
        if (
            constraints.latency_slo_ms is not None
            and candidate.estimated_latency_ms > constraints.latency_slo_ms
        ):
            continue
        if (
            constraints.min_throughput_tps
            and candidate.estimated_throughput_tps < constraints.min_throughput_tps
        ):
            continue
        feasible.append((candidate, replicas))

    if not feasible:
        return PlacementResponse(
            deployment_id=constraints.deployment_id,
            model_id=constraints.model_id,
            recommended_profile_id=None,
            feasible=False,
            reason="no candidate fits the placement constraints",
        )

    # highest precision (best quality) that fits; tie-break on capacity
    chosen, replicas = max(
        feasible, key=lambda item: (item[0].precision_bits, item[1])
    )
    total_capacity = replicas * constraints.gpu_count

    baseline_bytes = max(candidate.weight_bytes for candidate in request.candidates)
    baseline_replicas = project_capacity(
        baseline_bytes, constraints.vram_per_gpu_gb, constraints.headroom_fraction
    )
    baseline_capacity = max(baseline_replicas * constraints.gpu_count, 1)
    improvement = total_capacity / baseline_capacity

    return PlacementResponse(
        deployment_id=constraints.deployment_id,
        model_id=constraints.model_id,
        recommended_profile_id=chosen.profile_id,
        replicas_per_gpu=replicas,
        projected_capacity=total_capacity,
        baseline_capacity=baseline_capacity,
        capacity_improvement=improvement,
        feasible=True,
        reason=(
            f"{chosen.precision_bits}-bit profile fits {usable_per_gpu:.1f} GB/GPU "
            f"at {replicas} replica(s) per GPU"
        ),
    )


def candidates_from_options(
    options: Sequence[tuple[str, int, int, float, float, float]]
) -> tuple[PlacementCandidate, ...]:
    """Build placement candidates from ``(id, bits, bytes, latency, throughput, deg)``."""
    return tuple(
        PlacementCandidate(
            profile_id=profile_id,
            precision_bits=bits,
            weight_bytes=weight_bytes,
            estimated_latency_ms=latency,
            estimated_throughput_tps=throughput,
            quality_degradation=degradation,
        )
        for profile_id, bits, weight_bytes, latency, throughput, degradation in options
    )


def placement_document(response: PlacementResponse) -> dict[str, object]:
    """Serialize a placement response with a versioned schema id."""
    return {"schema_version": PLACEMENT_RECOMMENDATION_SCHEMA, **response.model_dump(mode="json")}


__all__ = [
    "PLACEMENT_RECOMMENDATION_SCHEMA",
    "MemoryRequirement",
    "PlacementCandidate",
    "PlacementConstraints",
    "PlacementRequest",
    "PlacementResponse",
    "candidates_from_options",
    "placement_document",
    "project_capacity",
    "recommend_precision_for_placement",
    "report_memory_requirements",
]
