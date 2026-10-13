"""Quantization configuration and precision profile domain models."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from adaptive.errors import ConfigurationError
from adaptive.models.architecture import LayerSpec
from adaptive.models.enums import (
    DType,
    EvidenceType,
    Granularity,
    QuantMethod,
    QuantScheme,
    RuntimeBackend,
)

_FLOAT_DTYPES = {DType.fp32, DType.fp16, DType.bf16, DType.fp8_e4m3, DType.fp8_e5m2}
_SUPPORTED_BITS = {4, 8, 16, 32}


class QuantConfig(BaseModel):
    """A single quantization configuration applied to one or more layers."""

    model_config = {"frozen": True}

    bits: int
    dtype: DType
    scheme: QuantScheme = QuantScheme.symmetric
    granularity: Granularity = Granularity.per_channel
    group_size: int | None = None
    method: QuantMethod = QuantMethod.reference
    scale_dtype: DType = DType.fp32
    zero_point_dtype: DType = DType.int8

    @model_validator(mode="after")
    def _validate(self) -> QuantConfig:
        if self.bits not in _SUPPORTED_BITS:
            raise ConfigurationError(f"unsupported bit width: {self.bits}")
        if self.granularity is Granularity.per_group:
            if not self.group_size or self.group_size <= 0:
                raise ConfigurationError("per_group granularity requires a positive group_size")
        elif self.group_size is not None:
            raise ConfigurationError("group_size is only valid for per_group granularity")
        if self.is_float and self.scheme is not QuantScheme.symmetric:
            raise ConfigurationError("floating point configs must be symmetric")
        return self

    @property
    def is_float(self) -> bool:
        """True when this config stores weights in a floating point format."""
        return self.dtype in _FLOAT_DTYPES and self.bits >= 16

    @property
    def is_quantized(self) -> bool:
        """True when this config applies integer affine quantization."""
        return not self.is_float

    @property
    def num_levels(self) -> int:
        return int(2**self.bits)


def baseline_config(dtype: DType) -> QuantConfig:
    """Return the reference (unquantized) config for ``dtype``."""
    bits = 32 if dtype is DType.fp32 else 16
    return QuantConfig(
        bits=bits,
        dtype=dtype,
        scheme=QuantScheme.symmetric,
        granularity=Granularity.per_tensor,
        method=QuantMethod.reference,
    )


def int_config(
    bits: int,
    *,
    scheme: QuantScheme = QuantScheme.symmetric,
    granularity: Granularity = Granularity.per_channel,
    group_size: int | None = None,
    method: QuantMethod = QuantMethod.reference,
) -> QuantConfig:
    """Construct an integer quantization config with a matching dtype."""
    if bits not in (4, 8):
        raise ConfigurationError(f"int_config only supports 4 or 8 bits, got {bits}")
    dtype = DType.int4 if bits == 4 else DType.int8
    return QuantConfig(
        bits=bits,
        dtype=dtype,
        scheme=scheme,
        granularity=granularity,
        group_size=group_size,
        method=method,
    )


class PrecisionProfile(BaseModel):
    """A layer-wise precision assignment for a specific model + runtime."""

    model_config = {"frozen": True}

    profile_id: str
    name: str
    description: str = ""
    default_config: QuantConfig
    assignments: dict[str, QuantConfig] = Field(default_factory=dict)
    runtime: RuntimeBackend
    method: QuantMethod = QuantMethod.reference
    deployable: bool = True
    evidence: EvidenceType = EvidenceType.estimated
    notes: tuple[str, ...] = Field(default_factory=tuple)

    def config_for(self, layer: LayerSpec) -> QuantConfig:
        """Return the effective config for ``layer``.

        Non-quantizable layers (norms, embeddings when excluded) always use the
        baseline floating point configuration, never the profile default.
        """
        if not layer.quantizable:
            return baseline_config(DType.fp16)
        return self.assignments.get(layer.name, self.default_config)

    @property
    def is_uniform(self) -> bool:
        """True when every assigned layer uses the default config."""
        return all(cfg == self.default_config for cfg in self.assignments.values())

    def distinct_bits(self) -> set[int]:
        bits = {self.default_config.bits}
        bits.update(cfg.bits for cfg in self.assignments.values())
        return bits
