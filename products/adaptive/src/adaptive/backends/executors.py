"""In-process backend execution hooks (Milestone 5).

These hooks perform the actual quantization when the required libraries and a
CUDA device are present. On a CPU/Mac host the backend's availability check
fails first, so ``prepare()`` never reaches an executor.

Only bitsandbytes and FP8 have real in-process executors. GPTQ and AWQ are
delegated to their library pipelines (which require calibration data and a GPU);
their executors raise an explicit, actionable error rather than pretending to
run. No executor silently falls back to full precision.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from adaptive.errors import BackendUnavailableError, UnsupportedQuantizationError
from adaptive.models.enums import QuantMethod
from adaptive.models.quantization import PrecisionProfile

Executor = Callable[[Any, PrecisionProfile], Any]


def _replace_linear(model: Any, factory: Callable[[Any], Any]) -> int:
    """Replace every ``nn.Linear`` child with ``factory(module)``; return count."""
    import torch

    replaced = 0

    def recurse(parent: Any) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                new_module = factory(child)
                setattr(parent, name, new_module)
                replaced += 1
            else:
                recurse(child)

    recurse(model)
    return replaced


def fp8_executor(model: Any, profile: PrecisionProfile) -> Any:
    """Cast linear weights to FP8 (E4M3) storage in place."""
    import torch

    target = torch.float8_e4m3fn
    converted = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = module.weight.data.to(target)
                converted += 1
    if converted == 0:
        raise UnsupportedQuantizationError("fp8 executor found no linear layers to convert")
    return model


def bitsandbytes_executor(model: Any, profile: PrecisionProfile) -> Any:
    """Replace linear layers with bitsandbytes INT8 / NF4 layers."""
    try:
        import bitsandbytes as bnb  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on environment
        raise BackendUnavailableError("bitsandbytes is not installed") from exc

    bits = profile.default_config.bits

    def factory(module: Any) -> Any:
        if bits == 4:
            return bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                quant_type="nf4",
                compute_dtype=None,
            )
        return bnb.nn.Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            has_fp16_weights=False,
        )

    replaced = _replace_linear(model, factory)
    if replaced == 0:
        raise UnsupportedQuantizationError("bitsandbytes executor found no linear layers")
    return model


def _delegated(name: str, instructions: str) -> Executor:
    def executor(model: Any, profile: PrecisionProfile) -> Any:
        raise BackendUnavailableError(
            f"{name} execution is delegated to its library pipeline on a GPU host; "
            f"{instructions}"
        )

    return executor


def default_executor(method: QuantMethod) -> Executor | None:
    """Return the in-process executor for a method, if one exists."""
    if method is QuantMethod.fp8:
        return fp8_executor
    if method is QuantMethod.bitsandbytes_int8 or method is QuantMethod.bitsandbytes_nf4:
        return bitsandbytes_executor
    if method is QuantMethod.gptq:
        return _delegated(
            "GPTQ",
            "use gptqmodel/auto-gptq with the pinned calibration set "
            "(see scripts/benchmark_gpu.sh)",
        )
    if method is QuantMethod.awq:
        return _delegated(
            "AWQ",
            "use autoawq with the pinned calibration set (see scripts/benchmark_gpu.sh)",
        )
    return None


__all__ = [
    "Executor",
    "bitsandbytes_executor",
    "default_executor",
    "fp8_executor",
]
