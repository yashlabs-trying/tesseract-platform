"""Concrete production backend adapters (Milestone 5)."""

from __future__ import annotations

from adaptive.backends.base import BackendAdapter, BackendRequirements
from adaptive.errors import UnsupportedQuantizationError
from adaptive.models.enums import QuantMethod, RuntimeBackend


class GptqBackend(BackendAdapter):
    name = "gptq"
    method = QuantMethod.gptq
    runtime = RuntimeBackend.cuda
    supported_bits = (4, 8)
    requires_calibration = True
    default_calibration_samples = 128
    artifact_suffix = "gptq.safetensors"

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(
            requires_cuda=True,
            requires_any_packages=("auto_gptq", "gptqmodel"),
            notes=("GPTQ requires a calibration set to compute Hessian statistics",),
        )


class AwqBackend(BackendAdapter):
    name = "awq"
    method = QuantMethod.awq
    runtime = RuntimeBackend.cuda
    supported_bits = (4,)
    requires_calibration = True
    default_calibration_samples = 128
    artifact_suffix = "awq.safetensors"

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(
            requires_cuda=True,
            requires_any_packages=("awq", "autoawq"),
            notes=("AWQ requires activation calibration",),
        )


class BitsAndBytesInt8Backend(BackendAdapter):
    name = "bitsandbytes-int8"
    method = QuantMethod.bitsandbytes_int8
    runtime = RuntimeBackend.cuda
    supported_bits = (8,)
    requires_calibration = False
    artifact_suffix = "bnb-int8"

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(
            requires_cuda=True,
            requires_packages=("bitsandbytes",),
        )


class BitsAndBytesNf4Backend(BackendAdapter):
    name = "bitsandbytes-nf4"
    method = QuantMethod.bitsandbytes_nf4
    runtime = RuntimeBackend.cuda
    supported_bits = (4,)
    requires_calibration = False
    artifact_suffix = "bnb-nf4"

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(
            requires_cuda=True,
            requires_packages=("bitsandbytes",),
        )


class Fp8Backend(BackendAdapter):
    name = "fp8"
    method = QuantMethod.fp8
    runtime = RuntimeBackend.cuda
    supported_bits = (8,)
    requires_calibration = False
    artifact_suffix = "fp8.safetensors"

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(
            requires_cuda=True,
            min_compute_capability="8.9",
            notes=("FP8 requires Hopper-class (compute capability >= 8.9) hardware",),
        )


def default_backends() -> tuple[BackendAdapter, ...]:
    """Return the built-in production backend set with execution hooks."""
    from adaptive.backends.executors import default_executor

    return (
        GptqBackend(executor=default_executor(QuantMethod.gptq)),
        AwqBackend(executor=default_executor(QuantMethod.awq)),
        BitsAndBytesInt8Backend(executor=default_executor(QuantMethod.bitsandbytes_int8)),
        BitsAndBytesNf4Backend(executor=default_executor(QuantMethod.bitsandbytes_nf4)),
        Fp8Backend(executor=default_executor(QuantMethod.fp8)),
    )


def get_backend(method: QuantMethod) -> BackendAdapter:
    """Return the backend implementing ``method``."""
    for backend in default_backends():
        if backend.method is method:
            return backend
    raise UnsupportedQuantizationError(f"no production backend for method {method.value}")


def backends_for_runtime(runtime: RuntimeBackend) -> list[BackendAdapter]:
    return [backend for backend in default_backends() if backend.runtime is runtime]


def available_backends() -> list[BackendAdapter]:
    """Backends that can execute in the current environment."""
    return [backend for backend in default_backends() if backend.available()]


__all__ = [
    "AwqBackend",
    "BitsAndBytesInt8Backend",
    "BitsAndBytesNf4Backend",
    "Fp8Backend",
    "GptqBackend",
    "available_backends",
    "backends_for_runtime",
    "default_backends",
    "get_backend",
]
