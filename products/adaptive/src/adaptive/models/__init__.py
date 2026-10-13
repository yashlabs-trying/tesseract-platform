"""Domain models for Tesseract Adaptive."""

from __future__ import annotations

from adaptive.models.architecture import LayerSpec, ModelArchitecture
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
from adaptive.models.quantization import (
    PrecisionProfile,
    QuantConfig,
    baseline_config,
    int_config,
)
from adaptive.models.storage import StorageBreakdown

__all__ = [
    "ArtifactManifest",
    "DType",
    "EvidenceType",
    "Granularity",
    "HardwareSpec",
    "LayerFamily",
    "LayerRole",
    "LayerSpec",
    "ModelArchitecture",
    "ModelFamily",
    "PrecisionProfile",
    "QuantConfig",
    "QuantMethod",
    "QuantScheme",
    "RuntimeBackend",
    "StorageBreakdown",
    "baseline_config",
    "dtype_bytes",
    "dtype_from_torch_name",
    "int_config",
    "sha256_bytes",
    "sha256_file",
]
