"""M5: production backend adapter framework."""

from __future__ import annotations

import pytest

from adaptive.backends import (
    BackendAdapter,
    available_backends,
    backends_for_runtime,
    cuda_available,
    default_backends,
    get_backend,
    plan_document,
)
from adaptive.errors import BackendUnavailableError, UnsupportedQuantizationError
from adaptive.models.enums import QuantMethod, RuntimeBackend
from adaptive.quantize import uniform_profile


def test_registry_lists_backends() -> None:
    assert len(default_backends()) == 5
    assert {backend.method for backend in default_backends()} == {
        QuantMethod.gptq,
        QuantMethod.awq,
        QuantMethod.bitsandbytes_int8,
        QuantMethod.bitsandbytes_nf4,
        QuantMethod.fp8,
    }


def test_get_backend() -> None:
    assert get_backend(QuantMethod.gptq).name == "gptq"
    with pytest.raises(UnsupportedQuantizationError):
        get_backend(QuantMethod.reference)


def test_backends_for_runtime() -> None:
    assert len(backends_for_runtime(RuntimeBackend.cuda)) == 5
    assert backends_for_runtime(RuntimeBackend.mps) == []


def test_availability_reflects_environment() -> None:
    if not cuda_available():
        assert available_backends() == []
        assert all(not backend.available() for backend in default_backends())


def test_requirements_summary() -> None:
    backend = get_backend(QuantMethod.gptq)
    summary = backend.requirements().summary()
    assert "CUDA" in summary
    assert "auto_gptq" in summary


def test_supports_bit_widths(tiny_architecture) -> None:
    gptq = get_backend(QuantMethod.gptq)
    awq = get_backend(QuantMethod.awq)
    profile4 = uniform_profile(
        tiny_architecture, 4, runtime=RuntimeBackend.cuda, method=QuantMethod.gptq
    )
    profile8 = uniform_profile(
        tiny_architecture, 8, runtime=RuntimeBackend.cuda, method=QuantMethod.gptq
    )
    assert gptq.supports(profile4)
    assert gptq.supports(profile8)
    assert awq.supports(profile4)
    assert not awq.supports(profile8)


def test_plan_is_generated_without_gpu(tiny_architecture) -> None:
    backend = get_backend(QuantMethod.gptq)
    profile = uniform_profile(
        tiny_architecture, 4, runtime=RuntimeBackend.cuda, method=QuantMethod.gptq, group_size=128
    )
    plan = backend.plan(tiny_architecture, profile)
    assert plan.method is QuantMethod.gptq
    assert plan.backend is RuntimeBackend.cuda
    assert plan.calibration_required is True
    assert plan.calibration_samples == 128
    assert plan.expected_weight_bytes > 0
    assert plan.executable_here is False
    assert plan.artifacts


def test_prepare_raises_without_gpu(tiny_architecture) -> None:
    if cuda_available():
        pytest.skip("CUDA present; unavailability path not exercised")
    backend = get_backend(QuantMethod.gptq)
    profile = uniform_profile(
        tiny_architecture, 4, runtime=RuntimeBackend.cuda, method=QuantMethod.gptq
    )
    with pytest.raises(BackendUnavailableError):
        backend.prepare(object(), profile)


def test_prepare_rejects_unsupported_profile(tiny_architecture) -> None:
    awq = get_backend(QuantMethod.awq)
    profile8 = uniform_profile(
        tiny_architecture, 8, runtime=RuntimeBackend.cuda, method=QuantMethod.gptq
    )
    with pytest.raises(UnsupportedQuantizationError):
        awq.prepare(object(), profile8)


def test_plan_document_has_schema_version(tiny_architecture) -> None:
    backend = get_backend(QuantMethod.fp8)
    profile = uniform_profile(
        tiny_architecture, 8, runtime=RuntimeBackend.cuda, method=QuantMethod.fp8
    )
    document = plan_document(backend.plan(tiny_architecture, profile))
    assert document["schema_version"] == "tesseract.adaptive.deployment-plan.v1"


def test_backend_adapter_is_abstract() -> None:
    assert isinstance(get_backend(QuantMethod.gptq), BackendAdapter)
