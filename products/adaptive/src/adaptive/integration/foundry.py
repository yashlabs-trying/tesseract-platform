"""Foundry integration adapter (Milestone 4).

Imports benchmark measurements in the shared ``tesseract.performance-profile.v1``
schema and exports Adaptive profile comparisons with explicit confidence and
provenance. Foundry remains independently usable; this adapter is a pure
contract translation with no network dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from adaptive.errors import ConfigurationError
from adaptive.models.cost import InferenceCost
from adaptive.models.enums import EvidenceType
from adaptive.models.optimization import OptimizationResult

FOUNDRY_PERFORMANCE_SCHEMA = "tesseract.performance-profile.v1"
PROFILE_COMPARISON_SCHEMA = "tesseract.adaptive.profile-comparison.v1"


class Provenance(BaseModel):
    """Where a measurement or export came from."""

    model_config = {"frozen": True}

    source: str
    tool: str = ""
    tool_version: str = ""
    evidence: EvidenceType = EvidenceType.estimated


class BenchmarkMeasurement(BaseModel):
    """A normalized benchmark measurement imported from Foundry."""

    model_config = {"frozen": True}

    profile_id: str
    model_id: str
    model_revision: str | None = None
    candidate_id: str | None = None
    hardware_gpu_model: str = ""
    hardware_gpu_count: int = 1
    hardware_vram_gb: float = 0.0
    engine: str = ""
    engine_version: str = ""
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    throughput_tps: float | None = None
    workload_concurrency: int = 1
    evidence: EvidenceType = EvidenceType.measured
    provenance: Provenance


def _evidence(value: str) -> EvidenceType:
    try:
        return EvidenceType(value)
    except ValueError as exc:
        raise ConfigurationError(f"unknown evidence_type: {value!r}") from exc


def import_performance_profile(document: Mapping[str, Any]) -> BenchmarkMeasurement:
    """Validate and import a ``tesseract.performance-profile.v1`` document."""
    schema = document.get("schema_version")
    if schema != FOUNDRY_PERFORMANCE_SCHEMA:
        raise ConfigurationError(
            f"expected {FOUNDRY_PERFORMANCE_SCHEMA}, got {schema!r}"
        )
    for required in ("profile_id", "model_id", "evidence_type"):
        if required not in document:
            raise ConfigurationError(f"performance profile missing {required!r}")

    def optional_float(key: str) -> float | None:
        value = document.get(key)
        return float(value) if value is not None else None

    return BenchmarkMeasurement(
        profile_id=str(document["profile_id"]),
        model_id=str(document["model_id"]),
        model_revision=_opt_str(document.get("model_revision")),
        candidate_id=_opt_str(document.get("candidate_id")),
        hardware_gpu_model=str(document.get("hardware_gpu_model", "")),
        hardware_gpu_count=int(document.get("hardware_gpu_count", 1) or 1),
        hardware_vram_gb=float(document.get("hardware_vram_gb", 0.0) or 0.0),
        engine=str(document.get("engine", "")),
        engine_version=str(document.get("engine_version", "")),
        ttft_p50_ms=optional_float("ttft_p50_ms"),
        ttft_p95_ms=optional_float("ttft_p95_ms"),
        itl_p50_ms=optional_float("itl_p50_ms"),
        itl_p95_ms=optional_float("itl_p95_ms"),
        throughput_tps=optional_float("throughput_tps"),
        workload_concurrency=int(document.get("workload_concurrency", 1) or 1),
        evidence=_evidence(str(document["evidence_type"])),
        provenance=Provenance(
            source="foundry",
            tool=str(document.get("benchmark_tool", "")),
            tool_version=str(document.get("benchmark_version", "")),
            evidence=_evidence(str(document["evidence_type"])),
        ),
    )


def measurement_to_cost(measurement: BenchmarkMeasurement) -> InferenceCost:
    """Convert a measured benchmark into an :class:`InferenceCost`."""
    latency_ms = measurement.ttft_p95_ms or measurement.ttft_p50_ms or 0.0
    throughput = measurement.throughput_tps or 0.0
    return InferenceCost(
        estimated_latency_ms=latency_ms,
        estimated_throughput_tps=throughput,
        estimated_cost_per_million_tokens=(
            latency_ms / 1000.0 / throughput if throughput > 0 else 0.0
        ),
        lower_cost=latency_ms,
        evidence=measurement.evidence,
    )


def _confidence(evidence: EvidenceType) -> float:
    return {
        EvidenceType.synthetic: 0.1,
        EvidenceType.estimated: 0.3,
        EvidenceType.measured: 0.8,
        EvidenceType.validated: 1.0,
    }[evidence]


def export_profile_comparison(result: OptimizationResult) -> dict[str, Any]:
    """Export a versioned profile comparison with confidence and provenance."""
    profiles = []
    for candidate in result.candidates:
        evidence = candidate.cost.evidence
        profiles.append(
            {
                "profile_id": candidate.profile_id,
                "name": candidate.profile.name,
                "bits": sorted(candidate.profile.distinct_bits()),
                "total_bytes": candidate.storage.total_bytes,
                "lower_cost": candidate.cost.lower_cost,
                "estimated_latency_ms": candidate.cost.estimated_latency_ms,
                "estimated_throughput_tps": candidate.cost.estimated_throughput_tps,
                "predicted_quality": candidate.predicted_quality,
                "predicted_degradation": candidate.predicted_degradation,
                "feasible": candidate.feasible,
                "confidence": _confidence(evidence),
                "provenance": {
                    "source": "adaptive",
                    "evidence": evidence.value,
                },
            }
        )
    return {
        "schema_version": PROFILE_COMPARISON_SCHEMA,
        "model_id": result.model_id,
        "structural_fingerprint": result.structural_fingerprint,
        "algorithm": result.algorithm,
        "selected_profile_id": result.selected_profile_id,
        "pareto_frontier": list(result.pareto_frontier),
        "profiles": profiles,
    }


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "FOUNDRY_PERFORMANCE_SCHEMA",
    "PROFILE_COMPARISON_SCHEMA",
    "BenchmarkMeasurement",
    "Provenance",
    "export_profile_comparison",
    "import_performance_profile",
    "measurement_to_cost",
]
