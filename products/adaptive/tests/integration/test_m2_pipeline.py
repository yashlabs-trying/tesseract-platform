"""M2 end-to-end: sensitivity -> quality -> optimization."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from adaptive.errors import ConfigurationError, QualityGateError
from adaptive.models.integrity import ArtifactManifest
from adaptive.models.optimization import OptimizationConstraints
from adaptive.models.quality import QualityMetric, QualityThreshold
from adaptive.optimizer import optimize
from adaptive.quality import enforce_thresholds, run_quality_suite
from adaptive.quantize import uniform_profile
from adaptive.schemas import (
    optimization_document,
    quality_document,
    sensitivity_document,
    write_document,
)
from adaptive.sensitivity import SensitivityMethod, analyze_sensitivity


def test_full_m2_pipeline(
    tiny_architecture, tiny_model, calibration_dataset, evaluation_dataset, tmp_path: Path
) -> None:
    report = analyze_sensitivity(
        tiny_architecture,
        tiny_model,
        calibration_dataset.sequences,
        methods=(
            SensitivityMethod.weight_error,
            SensitivityMethod.activation_weighted,
            SensitivityMethod.output_divergence,
        ),
        bits=4,
        group_size=16,
        calibration_fingerprint=calibration_dataset.fingerprint(),
        max_batches=2,
    )
    assert report.top_k(SensitivityMethod.activation_weighted, 1)

    uniform4 = uniform_profile(tiny_architecture, 4, group_size=16)
    baseline = run_quality_suite(
        tiny_architecture,
        tiny_model,
        uniform_profile(tiny_architecture, 16, group_size=16),
        calibration=calibration_dataset,
        evaluation=evaluation_dataset,
        suite=(QualityMetric.perplexity,),
        max_batches=2,
    )
    quality = run_quality_suite(
        tiny_architecture,
        tiny_model,
        uniform4,
        calibration=calibration_dataset,
        evaluation=evaluation_dataset,
        suite=(QualityMetric.perplexity, QualityMetric.output_divergence),
        max_batches=2,
    )
    assert quality.metric(QualityMetric.perplexity).baseline_value is not None
    assert quality.metric(QualityMetric.perplexity).value >= 0.0

    enforce_thresholds(
        quality,
        [
            QualityThreshold(
                metric=QualityMetric.perplexity, max_relative_degradation=1.0
            )
        ],
    )

    constraints = OptimizationConstraints(quality_tolerance=0.5, allowed_bits=(4, 8, 16))
    result = optimize(
        tiny_architecture,
        report,
        constraints,
        baseline_metric_value=baseline.metric(QualityMetric.perplexity).value,
        group_size=16,
    )
    assert result.selected() is not None
    assert result.pareto_frontier

    out = tmp_path / "m2"
    write_document(sensitivity_document(report), out / "sensitivity.json")
    write_document(quality_document(quality), out / "quality.json")
    write_document(optimization_document(result), out / "optimization.json")
    for name in ("sensitivity.json", "quality.json", "optimization.json"):
        ArtifactManifest.from_file(out / name).verify(out)


def test_quality_suite_rejects_shared_dataset(
    tiny_architecture, tiny_model, calibration_dataset
) -> None:
    profile = uniform_profile(tiny_architecture, 8, group_size=16)
    with pytest.raises(ConfigurationError):
        run_quality_suite(
            tiny_architecture,
            tiny_model,
            profile,
            calibration=calibration_dataset,
            evaluation=calibration_dataset,
        )


def test_quality_gate_failure_raises(
    tiny_architecture, tiny_model, calibration_dataset, evaluation_dataset
) -> None:
    profile = uniform_profile(tiny_architecture, 4, group_size=16)
    quality = run_quality_suite(
        tiny_architecture,
        tiny_model,
        profile,
        calibration=calibration_dataset,
        evaluation=evaluation_dataset,
        suite=(QualityMetric.perplexity,),
        max_batches=2,
    )
    # a zero tolerance cannot be met by any lossy profile unless identical
    if quality.metric(QualityMetric.perplexity).relative_degradation() > 0:
        with pytest.raises(QualityGateError):
            enforce_thresholds(
                quality,
                [
                    QualityThreshold(
                        metric=QualityMetric.perplexity, max_relative_degradation=0.0
                    )
                ],
            )
