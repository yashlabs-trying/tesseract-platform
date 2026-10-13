"""M2: quality evaluation engine."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adaptive.errors import ConfigurationError, QualityGateError
from adaptive.models.quality import (
    ExperimentManifest,
    MetricResult,
    QualityMetric,
    QualityReport,
    QualityThreshold,
    metric_lower_is_better,
)
from adaptive.quality import (
    DatasetSplit,
    TokenDataset,
    ensure_disjoint,
    make_synthetic_dataset,
)
from adaptive.quality.harness import enforce_thresholds
from adaptive.quality.metrics import (
    output_divergence,
    perplexity,
    structured_output,
    task_accuracy,
    top1_agreement,
)


def _report(metrics: dict[str, MetricResult]) -> QualityReport:
    manifest = ExperimentManifest(
        model_id="tiny",
        calibration_dataset="cal",
        evaluation_dataset="eval",
        calibration_fingerprint="a",
        evaluation_fingerprint="b",
    )
    return QualityReport(
        model_id="tiny",
        structural_fingerprint="fp",
        profile_id="p",
        manifest=manifest,
        metrics=metrics,
    )


def test_dataset_isolation_rejects_overlap() -> None:
    calibration = make_synthetic_dataset(
        "d", DatasetSplit.calibration, vocab_size=16, num_sequences=2, seq_len=8, seed=0
    )
    overlapping = TokenDataset(
        name="d-eval", split=DatasetSplit.evaluation, sequences=calibration.sequences, seed=0
    )
    with pytest.raises(ConfigurationError):
        ensure_disjoint(calibration, overlapping)


def test_dataset_isolation_rejects_wrong_split() -> None:
    calibration = make_synthetic_dataset(
        "d", DatasetSplit.calibration, vocab_size=16, num_sequences=2, seq_len=8, seed=0
    )
    with pytest.raises(ConfigurationError):
        ensure_disjoint(calibration, calibration)


def test_dataset_isolation_accepts_disjoint() -> None:
    calibration = make_synthetic_dataset(
        "d", DatasetSplit.calibration, vocab_size=16, num_sequences=2, seq_len=8, seed=0
    )
    evaluation = make_synthetic_dataset(
        "d", DatasetSplit.evaluation, vocab_size=16, num_sequences=2, seq_len=8, seed=1
    )
    ensure_disjoint(calibration, evaluation)


def test_perplexity_uniform_logits() -> None:
    vocab = 32

    def uniform(batch: torch.Tensor) -> torch.Tensor:
        return torch.zeros(batch.shape[0], batch.shape[1], vocab)

    dataset = make_synthetic_dataset(
        "d", DatasetSplit.evaluation, vocab_size=vocab, num_sequences=2, seq_len=8, seed=0
    )
    result = perplexity(uniform, dataset)
    assert result.value == pytest.approx(vocab, rel=1e-3)


def test_output_divergence_identical_models_zero() -> None:
    dataset = make_synthetic_dataset(
        "d", DatasetSplit.evaluation, vocab_size=16, num_sequences=2, seq_len=8, seed=0
    )

    def fn(batch: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator().manual_seed(3)
        return torch.randn(batch.shape[0], batch.shape[1], 16, generator=generator)

    result = output_divergence(fn, fn, dataset)
    assert result.value == pytest.approx(0.0, abs=1e-9)


def test_top1_agreement() -> None:
    dataset = make_synthetic_dataset(
        "d", DatasetSplit.evaluation, vocab_size=16, num_sequences=2, seq_len=8, seed=0
    )
    reference = torch.randn(2, 8, 16, generator=torch.Generator().manual_seed(0))
    candidate = reference.clone()
    candidate[0, 0, :] = -reference[0, 0, :]

    def ref_fn(batch: torch.Tensor) -> torch.Tensor:
        return reference[: batch.shape[0]]

    def cand_fn(batch: torch.Tensor) -> torch.Tensor:
        return candidate[: batch.shape[0]]

    result = top1_agreement(ref_fn, cand_fn, dataset)
    assert 0.0 <= result.value <= 1.0


def test_threshold_enforcement_passes_and_fails() -> None:
    metrics = {
        QualityMetric.perplexity.value: MetricResult(
            metric=QualityMetric.perplexity, value=10.5, baseline_value=10.0
        )
    }
    report = _report(metrics)
    enforce_thresholds(
        report, [QualityThreshold(metric=QualityMetric.perplexity, max_relative_degradation=0.1)]
    )
    with pytest.raises(QualityGateError):
        enforce_thresholds(
            report,
            [QualityThreshold(metric=QualityMetric.perplexity, max_relative_degradation=0.01)],
        )


def test_min_value_threshold() -> None:
    metrics = {
        QualityMetric.task_accuracy.value: MetricResult(
            metric=QualityMetric.task_accuracy, value=0.4, baseline_value=0.8
        )
    }
    report = _report(metrics)
    with pytest.raises(QualityGateError):
        enforce_thresholds(
            report, [QualityThreshold(metric=QualityMetric.task_accuracy, min_value=0.5)]
        )


def test_metric_direction() -> None:
    assert metric_lower_is_better(QualityMetric.perplexity)
    assert not metric_lower_is_better(QualityMetric.task_accuracy)
    lower = MetricResult(metric=QualityMetric.perplexity, value=11.0, baseline_value=10.0)
    assert lower.relative_degradation() == pytest.approx(0.1)
    higher = MetricResult(metric=QualityMetric.task_accuracy, value=0.9, baseline_value=0.8)
    assert higher.relative_degradation() == pytest.approx(0.0)
    worse = MetricResult(metric=QualityMetric.task_accuracy, value=0.6, baseline_value=0.8)
    assert worse.relative_degradation() == pytest.approx(0.25)


def test_manifest_fingerprint_depends_on_seed() -> None:
    base = dict(
        model_id="m",
        calibration_dataset="c",
        evaluation_dataset="e",
        calibration_fingerprint="a",
        evaluation_fingerprint="b",
    )
    first = ExperimentManifest(**base, seed=0)
    second = ExperimentManifest(**base, seed=1)
    assert first.fingerprint() != second.fingerprint()
    assert first.fingerprint() == ExperimentManifest(**base, seed=0).fingerprint()


def test_task_accuracy_metric() -> None:
    examples = [([1, 2, 3], 7), ([4, 5], 9)]
    result = task_accuracy(lambda prompt: prompt[-1] + 4, examples)
    assert result.value == pytest.approx(1.0)


def test_structured_output_metric() -> None:
    outputs = {"good": '{"a": 1}', "bad": "not json"}
    result = structured_output(lambda prompt: outputs[prompt], ["good", "bad"])
    assert result.value == pytest.approx(0.5)
