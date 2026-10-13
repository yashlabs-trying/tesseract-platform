"""Domain models for Tesseract Adaptive."""

from __future__ import annotations

from adaptive.models.architecture import LayerSpec, ModelArchitecture
from adaptive.models.cost import InferenceCost
from adaptive.models.enums import (
    DType,
    EvidenceType,
    Granularity,
    LayerFamily,
    LayerRole,
    ModelFamily,
    QuantMethod,
    QuantScheme,
    RuntimeBackend,
    dtype_bytes,
    dtype_from_torch_name,
)
from adaptive.models.hardware import HardwareSpec
from adaptive.models.integrity import ArtifactManifest, sha256_bytes, sha256_file
from adaptive.models.optimization import (
    CandidateEvaluation,
    OptimizationConstraints,
    OptimizationResult,
)
from adaptive.models.quality import (
    ExperimentManifest,
    MetricResult,
    QualityMetric,
    QualityReport,
    QualityThreshold,
)
from adaptive.models.quantization import (
    PrecisionProfile,
    QuantConfig,
    baseline_config,
    int_config,
)
from adaptive.models.sensitivity import (
    LayerSensitivity,
    SensitivityMethod,
    SensitivityReport,
)
from adaptive.models.storage import StorageBreakdown

__all__ = [
    "ArtifactManifest",
    "CandidateEvaluation",
    "DType",
    "EvidenceType",
    "ExperimentManifest",
    "Granularity",
    "HardwareSpec",
    "InferenceCost",
    "LayerFamily",
    "LayerRole",
    "LayerSensitivity",
    "LayerSpec",
    "MetricResult",
    "ModelArchitecture",
    "ModelFamily",
    "OptimizationConstraints",
    "OptimizationResult",
    "PrecisionProfile",
    "QualityMetric",
    "QualityReport",
    "QualityThreshold",
    "QuantConfig",
    "QuantMethod",
    "QuantScheme",
    "RuntimeBackend",
    "SensitivityMethod",
    "SensitivityReport",
    "StorageBreakdown",
    "baseline_config",
    "dtype_bytes",
    "dtype_from_torch_name",
    "int_config",
    "sha256_bytes",
    "sha256_file",
]
