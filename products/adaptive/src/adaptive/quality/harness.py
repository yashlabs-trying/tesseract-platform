"""Quality evaluation harness (Milestone 2)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from adaptive import __version__
from adaptive.errors import QualityGateError
from adaptive.models.architecture import ModelArchitecture
from adaptive.models.enums import EvidenceType
from adaptive.models.quality import (
    ExperimentManifest,
    MetricResult,
    QualityMetric,
    QualityReport,
    QualityThreshold,
)
from adaptive.models.quantization import PrecisionProfile
from adaptive.quality import metrics as _metrics
from adaptive.quality.datasets import TokenDataset, ensure_disjoint
from adaptive.quantize.model import quantize_model

DEFAULT_SUITE: tuple[QualityMetric, ...] = (
    QualityMetric.perplexity,
    QualityMetric.output_divergence,
)


def build_manifest(
    architecture: ModelArchitecture,
    calibration: TokenDataset,
    evaluation: TokenDataset,
    *,
    seed: int = 0,
    code_version: str | None = None,
    settings: dict[str, str | int | float | bool] | None = None,
) -> ExperimentManifest:
    return ExperimentManifest(
        model_id=architecture.model_id,
        model_revision=architecture.revision,
        calibration_dataset=calibration.name,
        evaluation_dataset=evaluation.name,
        calibration_fingerprint=calibration.fingerprint(),
        evaluation_fingerprint=evaluation.fingerprint(),
        seed=seed,
        code_version=code_version or __version__,
        settings=settings or {},
    )


def _next_token_predictor(model: nn.Module) -> Callable[[Sequence[int]], int]:
    def predict(prompt: Sequence[int]) -> int:
        tokens = torch.tensor([list(prompt)], dtype=torch.long)
        with torch.no_grad():
            logits = model(tokens)
        return int(logits[0, -1, :].argmax().item())

    return predict


def run_quality_suite(
    architecture: ModelArchitecture,
    model: nn.Module,
    profile: PrecisionProfile,
    *,
    calibration: TokenDataset,
    evaluation: TokenDataset,
    suite: Sequence[QualityMetric] = DEFAULT_SUITE,
    seed: int = 0,
    code_version: str | None = None,
    task_examples: Sequence[tuple[Sequence[int], int]] | None = None,
    generate_fn: Callable[[str], str] | None = None,
    prompts: Sequence[str] | None = None,
    batch_size: int = 2,
    max_batches: int | None = 8,
) -> QualityReport:
    """Evaluate ``profile`` against the full-precision model on ``evaluation``.

    ``calibration`` and ``evaluation`` must be disjoint; this is enforced here.
    """
    ensure_disjoint(calibration, evaluation)
    candidate = quantize_model(architecture, profile, model)

    def reference_fn(batch: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = model(batch)
        return result

    def candidate_fn(batch: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = candidate(batch)
        return result

    results: dict[str, MetricResult] = {}
    notes: list[str] = []

    if QualityMetric.perplexity in suite:
        baseline = _metrics.perplexity(
            reference_fn, evaluation, batch_size=batch_size, max_batches=max_batches
        )
        measured = _metrics.perplexity(
            candidate_fn,
            evaluation,
            batch_size=batch_size,
            max_batches=max_batches,
            baseline_value=baseline.value,
        )
        results[measured.metric.value] = measured

    if QualityMetric.long_context in suite:
        baseline_long = _metrics.long_context_perplexity(
            reference_fn, evaluation, batch_size=batch_size
        )
        measured_long = _metrics.long_context_perplexity(
            candidate_fn,
            evaluation,
            batch_size=batch_size,
            baseline_value=baseline_long.value,
        )
        results[measured_long.metric.value] = measured_long

    if QualityMetric.output_divergence in suite:
        results[QualityMetric.output_divergence.value] = _metrics.output_divergence(
            reference_fn, candidate_fn, evaluation, batch_size=batch_size, max_batches=max_batches
        )

    if QualityMetric.calibration_reconstruction in suite:
        results[QualityMetric.calibration_reconstruction.value] = (
            _metrics.calibration_reconstruction(architecture, profile, candidate)
        )
        notes.append(
            "calibration_reconstruction is a diagnostic, not proof of model quality"
        )

    if QualityMetric.task_accuracy in suite:
        if task_examples is None:
            raise ValueError("task_accuracy requires task_examples")
        baseline_acc = _metrics.task_accuracy(_next_token_predictor(model), task_examples)
        measured_acc = _metrics.task_accuracy(
            _next_token_predictor(candidate), task_examples, baseline_value=baseline_acc.value
        )
        results[measured_acc.metric.value] = measured_acc

    if QualityMetric.structured_output in suite:
        if generate_fn is None or prompts is None:
            raise ValueError("structured_output requires generate_fn and prompts")
        results[QualityMetric.structured_output.value] = _metrics.structured_output(
            generate_fn, prompts
        )

    manifest = build_manifest(
        architecture,
        calibration,
        evaluation,
        seed=seed,
        code_version=code_version,
        settings={"batch_size": batch_size, "max_batches": max_batches if max_batches else 0},
    )
    return QualityReport(
        model_id=architecture.model_id,
        structural_fingerprint=architecture.structural_fingerprint(),
        profile_id=profile.profile_id,
        manifest=manifest,
        metrics=results,
        evidence=EvidenceType.measured,
        notes=tuple(notes),
    )


def enforce_thresholds(
    report: QualityReport, thresholds: Sequence[QualityThreshold]
) -> None:
    """Raise :class:`QualityGateError` if any threshold is violated."""
    violations: list[str] = []
    for threshold in thresholds:
        key = threshold.metric.value
        if key not in report.metrics:
            violations.append(f"{key}: metric not present in report")
            continue
        result = report.metrics[key]
        if (
            threshold.max_relative_degradation is not None
            and result.relative_degradation() > threshold.max_relative_degradation
        ):
            violations.append(
                f"{key}: degradation {result.relative_degradation():.4f} > "
                f"{threshold.max_relative_degradation:.4f}"
            )
        if threshold.min_value is not None and result.value < threshold.min_value:
            violations.append(f"{key}: value {result.value:.4f} < {threshold.min_value:.4f}")
        if threshold.max_value is not None and result.value > threshold.max_value:
            violations.append(f"{key}: value {result.value:.4f} > {threshold.max_value:.4f}")
    if violations:
        raise QualityGateError("quality gate failed: " + "; ".join(violations))
