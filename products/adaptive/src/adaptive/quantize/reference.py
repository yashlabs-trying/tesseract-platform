"""Deterministic reference quantizer built on PyTorch ops.

This implementation is intentionally simple and portable: it runs on CPU and
Apple MPS, uses no custom kernels, and exists to (a) validate the profile and
storage machinery, and (b) provide a ground-truth reference against which
optimized backend adapters (GPTQ/AWQ/bitsandbytes) can be compared.

It is *not* a production kernel and must not be presented as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adaptive.errors import ConfigurationError, UnsupportedQuantizationError
from adaptive.models.enums import DType, Granularity, QuantScheme
from adaptive.models.quantization import QuantConfig

try:  # pragma: no cover - exercised indirectly
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

_EPS = 1e-12


def _require_torch() -> None:
    if torch is None:
        raise UnsupportedQuantizationError(
            "the reference quantizer requires PyTorch; install with: pip install torch"
        )


_TORCH_DTYPES = {
    DType.fp32: "float32",
    DType.fp16: "float16",
    DType.bf16: "bfloat16",
    DType.int8: "int8",
    DType.uint8: "uint8",
}

_TORCH_TO_DTYPE = {
    "torch.float32": DType.fp32,
    "torch.float16": DType.fp16,
    "torch.bfloat16": DType.bf16,
    "torch.int8": DType.int8,
    "torch.uint8": DType.uint8,
}


def torch_dtype(dtype: DType) -> torch.dtype:
    """Return the torch dtype corresponding to a domain :class:`DType`."""
    _require_torch()
    name = _TORCH_DTYPES.get(dtype)
    if name is None:
        raise UnsupportedQuantizationError(f"no torch dtype for {dtype.value}")
    result: torch.dtype = getattr(torch, name)
    return result


def _dtype_of(tensor: torch.Tensor) -> DType:
    return _TORCH_TO_DTYPE.get(str(tensor.dtype), DType.fp16)


@dataclass
class QuantizedTensor:
    """Result of quantizing a single weight tensor."""

    data: torch.Tensor
    """Integer codes; int8 for symmetric, uint8 for asymmetric schemes."""

    scales: torch.Tensor
    """Per-group scale factors (float32)."""

    zero_points: torch.Tensor | None
    """Per-group zero points (int), or ``None`` for symmetric schemes."""

    config: QuantConfig
    original_shape: tuple[int, ...]
    original_dtype: DType
    padded_in_features: int = 0
    packed: torch.Tensor | None = field(default=None)
    """Optional packed uint8 payload for 4-bit storage."""


def _group_tensor(t: torch.Tensor, config: QuantConfig) -> tuple[torch.Tensor, int, int]:
    """Reshape ``t`` into ``(num_groups, group_size)`` plus padding metadata."""
    _require_torch()
    if config.granularity is Granularity.per_tensor:
        return t.reshape(1, -1), t.shape[-1], 0
    if t.dim() != 2:
        raise ConfigurationError(
            f"{config.granularity.value} quantization requires a 2D weight, got {tuple(t.shape)}"
        )
    out_features, in_features = t.shape
    if config.granularity is Granularity.per_channel:
        return t.reshape(out_features, in_features), in_features, 0
    group_size = config.group_size
    assert group_size is not None  # validated by QuantConfig
    padding = (-in_features) % group_size
    if padding:
        t = torch.nn.functional.pad(t, (0, padding))
    num_groups = (in_features + padding) // group_size
    grouped = t.reshape(out_features, num_groups, group_size).reshape(
        out_features * num_groups, group_size
    )
    return grouped, in_features, padding


def quantize_tensor(t: torch.Tensor, config: QuantConfig) -> QuantizedTensor:
    """Quantize ``t`` according to ``config`` using an affine scheme."""
    _require_torch()
    if config.is_float:
        raise ConfigurationError("quantize_tensor requires an integer config")

    work = t.to(torch.float32)
    grouped, _in_features, padding = _group_tensor(work, config)

    if config.scheme is QuantScheme.symmetric:
        qmax = 2 ** (config.bits - 1) - 1
        qmin = -(2 ** (config.bits - 1))
        absmax = grouped.abs().amax(dim=1)
        scale = (absmax / qmax).clamp_min(_EPS)
        codes = torch.round(grouped / scale.unsqueeze(1))
        codes = codes.clamp(qmin, qmax)
        zero_points = None
        container = torch.int8
    else:
        qmax = 2**config.bits - 1
        qmin = 0
        minimum = grouped.amin(dim=1)
        maximum = grouped.amax(dim=1)
        scale = ((maximum - minimum) / (qmax - qmin)).clamp_min(_EPS)
        zero = torch.round(qmin - minimum / scale).clamp(qmin, qmax)
        codes = torch.round(grouped / scale.unsqueeze(1)) + zero.unsqueeze(1)
        codes = codes.clamp(qmin, qmax)
        zero_points = zero.to(torch.int32)
        container = torch.uint8 if config.bits == 8 else torch.int8

    return QuantizedTensor(
        data=codes.to(container),
        scales=scale.to(torch.float32),
        zero_points=zero_points,
        config=config,
        original_shape=tuple(t.shape),
        original_dtype=_dtype_of(t),
        padded_in_features=padding,
    )


def dequantize_tensor(qt: QuantizedTensor) -> torch.Tensor:
    """Reconstruct a floating point tensor from a :class:`QuantizedTensor`."""
    _require_torch()
    grouped = qt.data.to(torch.float32)
    if qt.zero_points is not None:
        grouped = grouped - qt.zero_points.to(torch.float32).unsqueeze(1)
    grouped = grouped * qt.scales.to(torch.float32).unsqueeze(1)

    if qt.config.granularity is Granularity.per_tensor:
        restored = grouped.reshape(qt.original_shape)
    else:
        out_features, in_features = qt.original_shape[0], qt.original_shape[-1]
        padded = in_features + qt.padded_in_features
        group_size = qt.config.group_size if qt.config.group_size else in_features
        num_groups = padded // group_size
        restored = grouped.reshape(out_features, num_groups, group_size).reshape(
            out_features, padded
        )[:, :in_features]

    target = torch_dtype(qt.original_dtype)
    return restored.reshape(qt.original_shape).to(target)


def quantization_error(t: torch.Tensor, qt: QuantizedTensor) -> dict[str, float]:
    """Compute reconstruction error statistics for a quantized tensor."""
    _require_torch()
    reference = t.to(torch.float32)
    reconstructed = dequantize_tensor(qt).to(torch.float32)
    diff = (reference - reconstructed).abs()
    denom = reference.abs().mean().clamp_min(_EPS)
    return {
        "mse": float((diff**2).mean().item()),
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "relative_error": float((diff.mean() / denom).item()),
    }
