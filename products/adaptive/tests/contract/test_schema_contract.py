"""Contract tests: emitted documents validate against versioned JSON schemas."""

from __future__ import annotations

from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from adaptive.analyzer import ModelAnalyzer
from adaptive.models.enums import Granularity
from adaptive.quantize import compute_storage, uniform_profile
from adaptive.schemas import (
    architecture_document,
    load_schema,
    profile_document,
    storage_document,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "models"


def _validate(document: dict, schema_name: str) -> None:
    schema = load_schema(schema_name)
    jsonschema.validate(instance=document, schema=dict(schema))


@pytest.mark.parametrize("config_name", ["llama-3.2-1b.json", "qwen2.5-0.5b.json"])
def test_architecture_document_contract(config_name: str) -> None:
    arch = ModelAnalyzer().analyze_file(FIXTURES / config_name, model_id=config_name)
    _validate(architecture_document(arch), "model-architecture-v1.json")


def test_profile_document_contract() -> None:
    arch = ModelAnalyzer().analyze_file(FIXTURES / "llama-3.2-1b.json", model_id="m")
    profile = uniform_profile(arch, 4, granularity=Granularity.per_group, group_size=128)
    _validate(profile_document(profile), "precision-profile-v1.json")


def test_storage_document_contract() -> None:
    arch = ModelAnalyzer().analyze_file(FIXTURES / "llama-3.2-1b.json", model_id="m")
    profile = uniform_profile(arch, 8, granularity=Granularity.per_channel)
    breakdown = compute_storage(arch, profile)
    _validate(storage_document(arch, profile, breakdown), "storage-breakdown-v1.json")


def test_schema_versions_are_immutable_ids() -> None:
    for name, expected in (
        ("model-architecture-v1.json", "tesseract.adaptive.model-architecture.v1"),
        ("precision-profile-v1.json", "tesseract.adaptive.precision-profile.v1"),
        ("storage-breakdown-v1.json", "tesseract.adaptive.storage-breakdown.v1"),
    ):
        schema = load_schema(name)
        assert schema["$id"] == expected
