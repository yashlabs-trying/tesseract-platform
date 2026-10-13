"""Sensitivity-driven quality degradation estimator (Milestone 2).

This is a *model*, not a measurement. It maps per-layer sensitivity scores and
bit widths to a predicted relative degradation of a reference quality metric.
Its outputs are labelled ``estimated`` and must be replaced by measured quality
reports before a profile is promoted to production.
"""

from __future__ import annotations

from adaptive.models.enums import EvidenceType
from adaptive.models.quality import QualityMetric, metric_lower_is_better
from adaptive.models.quantization import PrecisionProfile
from adaptive.models.sensitivity import SensitivityMethod, SensitivityReport

DEFAULT_PENALTIES: dict[int, float] = {4: 1.0, 8: 0.12, 16: 0.0}


def bit_penalty(bits: int, penalties: dict[int, float] | None = None) -> float:
    """Relative degradation weight for a bit width (higher = more degradation)."""
    table = penalties or DEFAULT_PENALTIES
    if bits in table:
        return table[bits]
    ordered = sorted(table.items())
    for threshold, penalty in ordered:
        if bits <= threshold:
            return penalty
    return ordered[-1][1]


class SensitivityQualityEstimator:
    """Predict quality degradation from layer sensitivity and bit widths."""

    def __init__(
        self,
        sensitivity: SensitivityReport,
        *,
        baseline_value: float,
        metric: QualityMetric = QualityMetric.perplexity,
        method: SensitivityMethod = SensitivityMethod.activation_weighted,
        scale: float = 0.1,
        penalties: dict[int, float] | None = None,
    ) -> None:
        self.sensitivity = sensitivity
        self.baseline_value = baseline_value
        self.metric = metric
        self.method = method
        self.scale = scale
        self.penalties = penalties or dict(DEFAULT_PENALTIES)
        self.evidence = EvidenceType.estimated

    def degradation(self, profile: PrecisionProfile) -> float:
        """Predicted relative degradation (>= 0) for ``profile``."""
        scores = self.sensitivity.scores
        if not scores:
            return 0.0
        total = 0.0
        for layer_name, by_method in scores.items():
            score = by_method.get(self.method.value, 0.0)
            config = profile.assignments.get(layer_name, profile.default_config)
            total += score * bit_penalty(config.bits, self.penalties)
        return self.scale * total / max(len(scores), 1)

    def predict(self, profile: PrecisionProfile) -> float:
        """Predicted value of the reference metric under ``profile``."""
        degradation = self.degradation(profile)
        if metric_lower_is_better(self.metric):
            return self.baseline_value * (1.0 + degradation)
        return self.baseline_value * (1.0 - min(degradation, 1.0))
