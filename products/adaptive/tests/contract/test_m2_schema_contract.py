"""Contract tests for M2 versioned JSON schemas."""

from __future__ import annotations

import pytest

jsonschema = pytest.importorskip("jsonschema")
pytest.importorskip("torch")

from adaptive.models.optimization import OptimizationConstraints
from adaptive.models.quality import QualityMetric
from adaptive.optimizer import optimize
from adaptive.quality import run_quality_suite
from adaptive.quantize import uniform_profile
from adaptive.schemas import (
    load_schema,
    optimization_document,
    quality_document,
    sensitivity_document,
)
from adaptive.sensitivity import SensitivityMethod, analyze_sensitivity


def _validate(document: dict, schema_name: str) -> None:
    jsonschema.validate(instance=document, schema=dict(load_schema(schema_name)))


def test_sensitivity_document_contract(
    tiny_architecture, tiny_model, calibration_dataset
) -> None:
    report = analyze_sensitivity(
        tiny_architecture,
        tiny_model,
        calibration_dataset.sequences,
        methods=(SensitivityMethod.weight_error, SensitivityMethod.activation_weighted),
        bits=4,
        group_size=16,
        calibration_fingerprint=calibration_dataset.fingerprint(),
        max_batches=2,
    )
    _validate(sensitivity_document(report), "sensitivity-report-v1.json")


def test_quality_document_contract(
    tiny_architecture, tiny_model, calibration_dataset, evaluation_dataset
) -> None:
    profile = uniform_profile(tiny_architecture, 4, group_size=16)
    report = run_quality_suite(
        tiny_architecture,
        tiny_model,
        profile,
        calibration=calibration_dataset,
        evaluation=evaluation_dataset,
        suite=(QualityMetric.perplexity, QualityMetric.output_divergence),
        max_batches=2,
    )
    _validate(quality_document(report), "quality-report-v1.json")


def test_optimization_document_contract(
    tiny_architecture, tiny_model, calibration_dataset
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
    result = optimize(
        tiny_architecture,
        report,
        OptimizationConstraints(quality_tolerance=0.5, allowed_bits=(4, 8, 16)),
        baseline_metric_value=100.0,
        group_size=16,
    )
    _validate(optimization_document(result), "optimization-result-v1.json")


def test_m2_schema_ids() -> None:
    for name, expected in (
        ("sensitivity-report-v1.json", "tesseract.adaptive.sensitivity-report.v1"),
        ("quality-report-v1.json", "tesseract.adaptive.quality-report.v1"),
        ("optimization-result-v1.json", "tesseract.adaptive.optimization-result.v1"),
    ):
        assert load_schema(name)["$id"] == expected
