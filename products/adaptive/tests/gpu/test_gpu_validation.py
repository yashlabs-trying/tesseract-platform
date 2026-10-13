"""GPU-only validation tests (Milestone 5).

These tests are marked ``gpu`` and require a CUDA device. They are skipped on
the Mac/CPU development host. They load the pinned GPU test manifests and, when
a GPU is present, assert that the required production backend is available.

Actual benchmark numbers must be produced on a GPU host via
``scripts/benchmark_gpu.sh`` and imported through Foundry; nothing here
fabricates results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive.backends import available_backends, cuda_available, cuda_device_name
from adaptive.benchmark import load_manifests

pytestmark = pytest.mark.gpu

MANIFEST_DIR = Path(__file__).resolve().parents[2] / "gpu" / "manifests"


def test_gpu_manifests_are_well_formed() -> None:
    manifests = load_manifests(MANIFEST_DIR)
    assert manifests
    for manifest in manifests:
        assert manifest.thresholds.ttft_p95_ms > 0.0
        assert manifest.hardware.gpu_count >= 1


@pytest.mark.skipif(not cuda_available(), reason="requires a CUDA device")
def test_required_backends_available_on_gpu() -> None:
    manifests = load_manifests(MANIFEST_DIR)
    available = {backend.method.value for backend in available_backends()}
    missing = sorted(
        {manifest.profile.method for manifest in manifests} - available
    )
    assert not missing, f"missing production backends on GPU host: {missing}"


@pytest.mark.skipif(not cuda_available(), reason="requires a CUDA device")
def test_cuda_device_reported() -> None:
    assert cuda_device_name() is not None
