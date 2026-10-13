"""GPU test manifests (Milestone 5).

Manifests pin exactly what must be measured on a GPU host: model, revision,
precision profile, hardware, workload, datasets and acceptance thresholds. They
are validated on any host; only execution requires a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from adaptive.errors import ConfigurationError

GPU_TEST_MANIFEST_SCHEMA = "tesseract.adaptive.gpu-test-manifest.v1"


class GpuProfileSpec(BaseModel):
    model_config = {"frozen": True}

    bits: int
    method: str
    group_size: int | None = None
    scheme: str = "symmetric"
    granularity: str = "per_group"


class GpuHardwareSpec(BaseModel):
    model_config = {"frozen": True}

    gpu_model: str
    gpu_count: int = Field(default=1, ge=1)
    vram_gb: float = Field(gt=0.0)


class GpuWorkloadSpec(BaseModel):
    model_config = {"frozen": True}

    concurrency: int = Field(default=1, ge=1)
    input_tokens: int = Field(default=512, ge=1)
    output_tokens: int = Field(default=128, ge=1)


class GpuThresholds(BaseModel):
    model_config = {"frozen": True}

    ttft_p95_ms: float = Field(gt=0.0)
    itl_p95_ms: float = Field(gt=0.0)
    min_throughput_tps: float = Field(default=0.0, ge=0.0)
    max_perplexity_degradation: float = Field(default=0.01, ge=0.0)


class GpuTestManifest(BaseModel):
    """A pinned GPU validation manifest."""

    model_config = {"frozen": True}

    schema_version: str = GPU_TEST_MANIFEST_SCHEMA
    name: str
    model_id: str
    model_revision: str | None = None
    profile: GpuProfileSpec
    hardware: GpuHardwareSpec
    workload: GpuWorkloadSpec
    thresholds: GpuThresholds
    calibration_dataset: str
    evaluation_dataset: str
    seed: int = 0


def load_manifest(path: Path | str) -> GpuTestManifest:
    """Load and validate a single GPU test manifest (YAML or JSON)."""
    target = Path(path)
    if not target.exists():
        raise ConfigurationError(f"manifest not found: {target}")
    text = target.read_text()
    if target.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigurationError(f"manifest root must be an object: {target}")
    if data.get("schema_version", GPU_TEST_MANIFEST_SCHEMA) != GPU_TEST_MANIFEST_SCHEMA:
        raise ConfigurationError(
            f"unexpected manifest schema: {data.get('schema_version')!r}"
        )
    return GpuTestManifest.model_validate(data)


def load_manifests(directory: Path | str) -> list[GpuTestManifest]:
    """Load every ``*.yaml``/``*.yml`` manifest in a directory."""
    root = Path(directory)
    if not root.exists():
        raise ConfigurationError(f"manifest directory not found: {root}")
    manifests = [
        load_manifest(path)
        for path in sorted(root.iterdir())
        if path.suffix in (".yaml", ".yml")
    ]
    if not manifests:
        raise ConfigurationError(f"no manifests found in {root}")
    return manifests


__all__ = [
    "GPU_TEST_MANIFEST_SCHEMA",
    "GpuHardwareSpec",
    "GpuProfileSpec",
    "GpuTestManifest",
    "GpuThresholds",
    "GpuWorkloadSpec",
    "load_manifest",
    "load_manifests",
]
