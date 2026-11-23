"""Contract tests for M5 schemas: benchmark report, deployment plan, GPU manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from adaptive.backends import get_backend, plan_document
from adaptive.benchmark import build_benchmark_report, load_manifest
from adaptive.integration import import_performance_profile
from adaptive.models.enums import QuantMethod, RuntimeBackend
from adaptive.quantize import uniform_profile
from adaptive.schemas import load_schema

ROOT = Path(__file__).resolve().parents[2]


def _validate(document: dict, schema_name: str) -> None:
    jsonschema.validate(instance=document, schema=dict(load_schema(schema_name)))


def test_benchmark_report_contract() -> None:
    measurement = import_performance_profile(
        json.loads((ROOT / "fixtures" / "foundry" / "performance-profile-example.json").read_text())
    )
    report = build_benchmark_report([measurement])
    _validate(report.model_dump(mode="json"), "benchmark-report-v1.json")


def test_deployment_plan_contract(tiny_architecture) -> None:
    backend = get_backend(QuantMethod.gptq)
    profile = uniform_profile(
        tiny_architecture, 4, runtime=RuntimeBackend.cuda, method=QuantMethod.gptq
    )
    _validate(plan_document(backend.plan(tiny_architecture, profile)), "deployment-plan-v1.json")


def test_gpu_manifest_contract() -> None:
    manifest = load_manifest(ROOT / "gpu" / "manifests" / "qwen2.5-0.5b-int4-gptq.yaml")
    _validate(manifest.model_dump(mode="json"), "gpu-test-manifest-v1.json")


def test_m5_schema_ids() -> None:
    for name, expected in (
        ("benchmark-report-v1.json", "tesseract.adaptive.benchmark-report.v1"),
        ("deployment-plan-v1.json", "tesseract.adaptive.deployment-plan.v1"),
        ("gpu-test-manifest-v1.json", "tesseract.adaptive.gpu-test-manifest.v1"),
    ):
        assert load_schema(name)["$id"] == expected
