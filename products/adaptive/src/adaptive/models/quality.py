"""Quality evaluation domain models."""

from __future__ import annotations

import hashlib
import json
from enum import Enum

from pydantic import BaseModel, Field

from adaptive.models.enums import EvidenceType


class QualityMetric(str, Enum):
    """Configurable quality metrics."""

    perplexity = "perplexity"
    calibration_reconstruction = "calibration_reconstruction"
    output_divergence = "output_divergence"
    task_accuracy = "task_accuracy"
    long_context = "long_context"
    structured_output = "structured_output"


_LOWER_IS_BETTER = {
    QualityMetric.perplexity: True,
    QualityMetric.calibration_reconstruction: True,
    QualityMetric.output_divergence: True,
    QualityMetric.task_accuracy: False,
    QualityMetric.long_context: True,
    QualityMetric.structured_output: False,
}


def metric_lower_is_better(metric: QualityMetric) -> bool:
    """True when a smaller value of ``metric`` is better."""
    return _LOWER_IS_BETTER[metric]


class MetricResult(BaseModel):
    """A single measured metric value with provenance."""

    model_config = {"frozen": True}

    metric: QualityMetric
    value: float
    num_samples: int = 0
    baseline_value: float | None = None
    evidence: EvidenceType = EvidenceType.measured

    @property
    def lower_is_better(self) -> bool:
        return _LOWER_IS_BETTER[self.metric]

    def relative_degradation(self) -> float:
        """Relative worsening versus ``baseline_value`` (0.0 when not worse).

        For lower-is-better metrics degradation is ``(value - base) / |base|``;
        for higher-is-better metrics it is ``(base - value) / |base|``.
        """
        if self.baseline_value is None or self.baseline_value == 0:
            return 0.0
        if self.lower_is_better:
            delta = (self.value - self.baseline_value) / abs(self.baseline_value)
        else:
            delta = (self.baseline_value - self.value) / abs(self.baseline_value)
        return max(delta, 0.0)


class QualityThreshold(BaseModel):
    """A constraint on one quality metric."""

    model_config = {"frozen": True}

    metric: QualityMetric
    max_relative_degradation: float | None = None
    min_value: float | None = None
    max_value: float | None = None


class ExperimentManifest(BaseModel):
    """Pinned settings for a reproducible evaluation."""

    model_config = {"frozen": True}

    model_id: str
    model_revision: str | None = None
    tokenizer: str = "reference"
    calibration_dataset: str
    evaluation_dataset: str
    calibration_fingerprint: str
    evaluation_fingerprint: str
    seed: int = 0
    code_version: str = "0.1.0"
    settings: dict[str, str | int | float | bool] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class QualityReport(BaseModel):
    """Measured quality of one profile against a baseline."""

    model_config = {"frozen": True}

    model_id: str
    structural_fingerprint: str
    profile_id: str
    manifest: ExperimentManifest
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    evidence: EvidenceType = EvidenceType.measured
    notes: tuple[str, ...] = Field(default_factory=tuple)

    def metric(self, metric: QualityMetric | str) -> MetricResult:
        key = metric.value if isinstance(metric, QualityMetric) else metric
        if key not in self.metrics:
            raise KeyError(key)
        return self.metrics[key]

    def degradation(self, metric: QualityMetric | str) -> float:
        return self.metric(metric).relative_degradation()
