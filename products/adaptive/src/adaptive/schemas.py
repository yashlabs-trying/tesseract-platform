"""Versioned JSON document serialization for Tesseract Adaptive artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from adaptive.models.architecture import ModelArchitecture
from adaptive.models.quantization import PrecisionProfile
from adaptive.models.storage import StorageBreakdown

ARCHITECTURE_SCHEMA = "tesseract.adaptive.model-architecture.v1"
PROFILE_SCHEMA = "tesseract.adaptive.precision-profile.v1"
STORAGE_SCHEMA = "tesseract.adaptive.storage-breakdown.v1"
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
