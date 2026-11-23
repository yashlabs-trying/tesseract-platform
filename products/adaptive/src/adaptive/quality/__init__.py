"""Quality evaluation (Milestone 2).

Configurable metrics: perplexity, calibration reconstruction, reference-vs-
quantized output divergence, task accuracy, long-context quality and structured
output correctness. Calibration and final evaluation datasets are kept
separate and the separation is enforced structurally. Reconstruction error is a
diagnostic only; it is never accepted as proof of model quality.
"""

from __future__ import annotations

from adaptive.models.quality import (
    ExperimentManifest,
    MetricResult,
    QualityMetric,
    QualityReport,
    QualityThreshold,
)
from adaptive.quality.datasets import (
    DatasetSplit,
    TokenDataset,
    ensure_disjoint,
    make_long_context_dataset,
    make_synthetic_dataset,
)
from adaptive.quality.harness import (
    DEFAULT_SUITE,
    build_manifest,
    enforce_thresholds,
    run_quality_suite,
)

__all__ = [
    "DEFAULT_SUITE",
    "DatasetSplit",
    "ExperimentManifest",
    "MetricResult",
    "QualityMetric",
    "QualityReport",
    "QualityThreshold",
    "TokenDataset",
    "build_manifest",
    "enforce_thresholds",
    "ensure_disjoint",
    "make_long_context_dataset",
    "make_synthetic_dataset",
    "run_quality_suite",
]
