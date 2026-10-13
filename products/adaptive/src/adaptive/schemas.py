"""Versioned JSON document serialization for Tesseract Adaptive artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from adaptive.models.architecture import ModelArchitecture
from adaptive.models.optimization import OptimizationResult
from adaptive.models.quality import QualityReport
from adaptive.models.quantization import PrecisionProfile
from adaptive.models.sensitivity import SensitivityReport
from adaptive.models.storage import StorageBreakdown

ARCHITECTURE_SCHEMA = "tesseract.adaptive.model-architecture.v1"
PROFILE_SCHEMA = "tesseract.adaptive.precision-profile.v1"
STORAGE_SCHEMA = "tesseract.adaptive.storage-breakdown.v1"
SENSITIVITY_SCHEMA = "tesseract.adaptive.sensitivity-report.v1"
QUALITY_SCHEMA = "tesseract.adaptive.quality-report.v1"
OPTIMIZATION_SCHEMA = "tesseract.adaptive.optimization-result.v1"
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas" / "tesseract"


def architecture_document(architecture: ModelArchitecture) -> dict[str, Any]:
    """Serialize an architecture to a versioned JSON document."""
    return {
        "schema_version": ARCHITECTURE_SCHEMA,
        "model_id": architecture.model_id,
        "model_type": architecture.model_type,
        "family": architecture.family.value,
        "revision": architecture.revision,
        "hidden_size": architecture.hidden_size,
        "intermediate_size": architecture.intermediate_size,
        "num_layers": architecture.num_layers,
        "num_attention_heads": architecture.num_attention_heads,
        "num_key_value_heads": architecture.num_key_value_heads,
        "head_dim": architecture.head_dim,
        "vocab_size": architecture.vocab_size,
        "max_position_embeddings": architecture.max_position_embeddings,
        "tie_word_embeddings": architecture.tie_word_embeddings,
        "baseline_dtype": architecture.baseline_dtype.value,
        "total_parameters": architecture.total_parameters,
        "quantizable_parameters": architecture.quantizable_parameters,
        "baseline_weight_bytes": architecture.baseline_weight_bytes,
        "structural_fingerprint": architecture.structural_fingerprint(),
        "notes": list(architecture.notes),
    }


def profile_document(profile: PrecisionProfile) -> dict[str, Any]:
    """Serialize a precision profile to a versioned JSON document."""
    return {
        "schema_version": PROFILE_SCHEMA,
        "profile_id": profile.profile_id,
        "name": profile.name,
        "description": profile.description,
        "runtime": profile.runtime.value,
        "method": profile.method.value,
        "deployable": profile.deployable,
        "evidence": profile.evidence.value,
        "default_config": profile.default_config.model_dump(mode="json"),
        "assignments": {
            name: cfg.model_dump(mode="json")
            for name, cfg in sorted(profile.assignments.items())
        },
        "notes": list(profile.notes),
    }


def storage_document(
    architecture: ModelArchitecture,
    profile: PrecisionProfile,
    breakdown: StorageBreakdown,
) -> dict[str, Any]:
    """Serialize a storage breakdown to a versioned JSON document."""
    return {
        "schema_version": STORAGE_SCHEMA,
        "model_id": architecture.model_id,
        "structural_fingerprint": architecture.structural_fingerprint(),
        "profile_id": profile.profile_id,
        "baseline_bytes": breakdown.baseline_bytes,
        "quantized_bytes": breakdown.total_bytes,
        "compression_ratio": breakdown.compression_ratio,
        "breakdown": breakdown.model_dump(mode="json"),
    }


def sensitivity_document(report: SensitivityReport) -> dict[str, Any]:
    """Serialize a sensitivity report to a versioned JSON document."""
    return {
        "schema_version": SENSITIVITY_SCHEMA,
        "model_id": report.model_id,
        "structural_fingerprint": report.structural_fingerprint,
        "calibration_fingerprint": report.calibration_fingerprint,
        "methods": [method.value for method in report.methods],
        "bits": report.bits,
        "group_size": report.group_size,
        "scores": report.scores,
        "raw_scores": report.raw_scores,
        "rankings": report.rankings,
        "family_attribution": report.family_attribution,
        "evidence": report.evidence.value,
        "notes": list(report.notes),
    }


def quality_document(report: QualityReport) -> dict[str, Any]:
    """Serialize a quality report to a versioned JSON document."""
    return {
        "schema_version": QUALITY_SCHEMA,
        "model_id": report.model_id,
        "structural_fingerprint": report.structural_fingerprint,
        "profile_id": report.profile_id,
        "manifest": report.manifest.model_dump(mode="json"),
        "metrics": {
            name: result.model_dump(mode="json") for name, result in report.metrics.items()
        },
        "evidence": report.evidence.value,
        "notes": list(report.notes),
    }


def optimization_document(result: OptimizationResult) -> dict[str, Any]:
    """Serialize an optimization result to a versioned JSON document."""
    return {
        "schema_version": OPTIMIZATION_SCHEMA,
        "model_id": result.model_id,
        "structural_fingerprint": result.structural_fingerprint,
        "algorithm": result.algorithm,
        "constraints": result.constraints.model_dump(mode="json"),
        "selected_profile_id": result.selected_profile_id,
        "pareto_frontier": list(result.pareto_frontier),
        "uniform_4_profile_id": result.uniform_4_profile_id,
        "uniform_8_profile_id": result.uniform_8_profile_id,
        "candidates": [
            {
                "profile_id": candidate.profile_id,
                "name": candidate.profile.name,
                "bits": sorted(candidate.profile.distinct_bits()),
                "total_bytes": candidate.storage.total_bytes,
                "lower_cost": candidate.cost.lower_cost,
                "estimated_latency_ms": candidate.cost.estimated_latency_ms,
                "predicted_quality": candidate.predicted_quality,
                "predicted_degradation": candidate.predicted_degradation,
                "feasible": candidate.feasible,
                "violations": list(candidate.violations),
            }
            for candidate in result.candidates
        ],
        "evidence": result.evidence.value,
        "notes": list(result.notes),
    }


def load_schema(name: str) -> Mapping[str, Any]:
    """Load a versioned JSON schema by file name from ``schemas/tesseract``."""
    path = SCHEMA_DIR / name
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"schema {name} is not a JSON object")
    return data


def write_document(document: Mapping[str, Any], path: Path | str) -> None:
    """Atomically write a JSON document (write temp, then replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
    tmp.replace(target)
