"""Contract loader and validator.

Contracts live under ``contracts/`` at the repository root and are the single
source of truth for cross-product data exchange. ``load_schema`` returns the raw
JSON Schema; ``validate`` returns a list of human-readable errors (empty when
valid).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft7Validator

_SCHEMA_FILES = {
    "workload": "workload.schema.json",
    "telemetry": "telemetry.schema.json",
    "reservation": "reservation.schema.json",
    "deployment": "deployment.schema.json",
    "lifecycle": "lifecycle.schema.json",
    "precision-profile": "precision_profile.schema.json",
    "rollback": "rollback.schema.json",
    # Consumed contract (owned by Foundry, vendored here for validation).
    "foundry-performance-profile": "foundry-performance-profile.schema.json",
}


def contracts_dir() -> Path:
    override = os.environ.get("TESSERACT_CONTRACTS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "contracts"


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    if name not in _SCHEMA_FILES:
        raise KeyError(f"unknown contract {name!r}; known: {sorted(_SCHEMA_FILES)}")
    path = contracts_dir() / _SCHEMA_FILES[name]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=None)
def _validator(name: str) -> Draft7Validator:
    return Draft7Validator(load_schema(name))


def validate(name: str, document: Mapping[str, Any]) -> list[str]:
    """Validate ``document`` against contract ``name``. Returns error strings."""
    errors = sorted(_validator(name).iter_errors(document), key=lambda e: list(e.path))
    return [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]


def validate_or_raise(name: str, document: Mapping[str, Any]) -> None:
    errors = validate(name, document)
    if errors:
        raise ValueError(f"{name} contract violation: " + "; ".join(errors))


__all__ = [
    "contracts_dir",
    "load_schema",
    "validate",
    "validate_or_raise",
]
