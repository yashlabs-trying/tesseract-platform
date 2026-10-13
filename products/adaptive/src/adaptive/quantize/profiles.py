"""Precision profile generation and runtime compatibility validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from pydantic import BaseModel, Field

from adaptive.errors import CompatibilityError, UnsupportedQuantizationError
from adaptive.models.architecture import ModelArchitecture
from adaptive.models.enums import (
    DType,
    EvidenceType,
    Granularity,
    QuantMethod,
    QuantScheme,
    RuntimeBackend,
)
from adaptive.models.quantization import (
    PrecisionProfile,
    QuantConfig,
    baseline_config,
    int_config,
)


class RuntimeCapability(BaseModel):
    """Declared execution capabilities of a runtime backend."""

    model_config = {"frozen": True}

    backend: RuntimeBackend
    float_bits: frozenset[int] = Field(default_factory=frozenset)
    int_bits: frozenset[int] = Field(default_factory=frozenset)
    methods: frozenset[QuantMethod] = Field(default_factory=frozenset)
    production: bool = False
    notes: tuple[str, ...] = Field(default_factory=tuple)

    def supports_bits(self, bits: int, is_float: bool) -> bool:
        return bits in (self.float_bits if is_float else self.int_bits)


# Declared capabilities. These describe what each runtime can *execute*; they do
# not assert measured performance. GPU backends are declared but unvalidated on
# the Mac M5 development host.
RUNTIME_CAPABILITIES: dict[RuntimeBackend, RuntimeCapability] = {
    RuntimeBackend.torch_eager: RuntimeCapability(
        backend=RuntimeBackend.torch_eager,
        float_bits=frozenset({16, 32}),
        int_bits=frozenset({4, 8}),
        methods=frozenset({QuantMethod.reference}),
        production=False,
        notes=("reference simulation only; int weights are dequantized before matmul",),
    ),
    RuntimeBackend.torch_compile: RuntimeCapability(
        backend=RuntimeBackend.torch_compile,
        float_bits=frozenset({16, 32}),
        int_bits=frozenset({8}),
        methods=frozenset({QuantMethod.reference}),
        production=False,
    ),
    RuntimeBackend.mps: RuntimeCapability(
        backend=RuntimeBackend.mps,
        float_bits=frozenset({16, 32}),
        int_bits=frozenset(),
        methods=frozenset(),
        production=False,
        notes=("no integer matmul kernels; float execution only",),
    ),
    RuntimeBackend.cuda: RuntimeCapability(
        backend=RuntimeBackend.cuda,
        float_bits=frozenset({16, 32}),
        int_bits=frozenset({4, 8}),
        methods=frozenset(
            {
                QuantMethod.reference,
                QuantMethod.gptq,
                QuantMethod.awq,
                QuantMethod.bitsandbytes_int8,
                QuantMethod.bitsandbytes_nf4,
                QuantMethod.fp8,
            }
        ),
        production=True,
        notes=("declared capability; requires NVIDIA validation before claims",),
    ),
    RuntimeBackend.vllm: RuntimeCapability(
        backend=RuntimeBackend.vllm,
        float_bits=frozenset({16}),
        int_bits=frozenset({4, 8}),
        methods=frozenset({QuantMethod.gptq, QuantMethod.awq, QuantMethod.fp8}),
        production=True,
        notes=("declared capability; requires GPU validation",),
    ),
    RuntimeBackend.sglang: RuntimeCapability(
        backend=RuntimeBackend.sglang,
        float_bits=frozenset({16}),
        int_bits=frozenset({4, 8}),
        methods=frozenset({QuantMethod.gptq, QuantMethod.awq, QuantMethod.fp8}),
        production=True,
        notes=("declared capability; requires GPU validation",),
    ),
    RuntimeBackend.tensorrt_llm: RuntimeCapability(
        backend=RuntimeBackend.tensorrt_llm,
        float_bits=frozenset({16}),
        int_bits=frozenset({4, 8}),
        methods=frozenset({QuantMethod.gptq, QuantMethod.awq, QuantMethod.fp8}),
        production=True,
        notes=("declared capability; requires GPU validation",),
    ),
    RuntimeBackend.llama_cpp: RuntimeCapability(
        backend=RuntimeBackend.llama_cpp,
        float_bits=frozenset({16}),
        int_bits=frozenset({4, 8}),
        methods=frozenset({QuantMethod.gguf}),
        production=True,
        notes=("declared capability; requires validation",),
    ),
}


def capability(runtime: RuntimeBackend) -> RuntimeCapability:
    try:
        return RUNTIME_CAPABILITIES[runtime]
    except KeyError as exc:  # pragma: no cover - defensive
        raise UnsupportedQuantizationError(f"unknown runtime: {runtime}") from exc


def supported_bits(runtime: RuntimeBackend) -> frozenset[int]:
    cap = capability(runtime)
    return frozenset(cap.float_bits | cap.int_bits)


def _profile_id(name: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"{name}-{hashlib.sha256(encoded).hexdigest()[:12]}"


def baseline_profile(
    architecture: ModelArchitecture,
    *,
    runtime: RuntimeBackend = RuntimeBackend.torch_eager,
    dtype: DType | None = None,
) -> PrecisionProfile:
    """Reference (unquantized) profile at the architecture baseline dtype."""
    target = dtype or architecture.baseline_dtype
    config = baseline_config(target)
    return PrecisionProfile(
        profile_id=_profile_id("baseline", {"dtype": target.value, "runtime": runtime.value}),
        name=f"baseline-{target.value}",
        description="Unquantized reference precision",
        default_config=config,
        assignments={},
        runtime=runtime,
        method=QuantMethod.reference,
        deployable=capability(runtime).supports_bits(config.bits, is_float=True),
        evidence=EvidenceType.estimated,
        notes=("reference profile; establishes the quality and size baseline",),
    )


def uniform_profile(
    architecture: ModelArchitecture,
    bits: int,
    *,
    runtime: RuntimeBackend = RuntimeBackend.torch_eager,
    method: QuantMethod = QuantMethod.reference,
    scheme: QuantScheme = QuantScheme.symmetric,
    granularity: Granularity = Granularity.per_group,
    group_size: int | None = 128,
) -> PrecisionProfile:
    """Uniform integer profile applied to every quantizable layer."""
    effective_group_size = group_size if granularity is Granularity.per_group else None
    config = int_config(
        bits,
        scheme=scheme,
        granularity=granularity,
        group_size=effective_group_size,
        method=method,
    )
    cap = capability(runtime)
    supported = cap.supports_bits(bits, is_float=False) and method in cap.methods
    deployable = supported and cap.production
    notes: list[str] = []
    if not supported:
        notes.append(f"{runtime.value} does not declare {bits}-bit {method.value} support")
    elif not cap.production:
        notes.append("reference backend; not a production kernel")
    return PrecisionProfile(
        profile_id=_profile_id(
            f"uniform-{bits}",
            {
                "bits": bits,
                "runtime": runtime.value,
                "method": method.value,
                "granularity": granularity.value,
                "group_size": group_size,
                "scheme": scheme.value,
            },
        ),
        name=f"uniform-{bits}bit-{granularity.value}",
        description=f"Uniform {bits}-bit {scheme.value} {granularity.value} quantization",
        default_config=config,
        assignments={},
        runtime=runtime,
        method=method,
        deployable=deployable,
        evidence=EvidenceType.estimated,
        notes=tuple(notes),
    )


def mixed_profile(
    architecture: ModelArchitecture,
    layer_bits: Mapping[str, int],
    *,
    default_bits: int = 4,
    runtime: RuntimeBackend = RuntimeBackend.torch_eager,
    method: QuantMethod = QuantMethod.reference,
    scheme: QuantScheme = QuantScheme.symmetric,
    granularity: Granularity = Granularity.per_group,
    group_size: int | None = 128,
    name: str = "mixed",
) -> PrecisionProfile:
    """Mixed-precision profile with per-layer overrides.

    ``layer_bits`` maps layer names to bit widths. Layers absent from the map
    receive ``default_bits``. Non-quantizable layers always stay at baseline.
    """
    names = {layer.name: layer for layer in architecture.layers}
    for layer_name in layer_bits:
        layer = names.get(layer_name)
        if layer is None:
            raise UnsupportedQuantizationError(
                f"assignment references unknown layer {layer_name!r}"
            )
        if not layer.quantizable:
            raise UnsupportedQuantizationError(f"layer {layer_name!r} is not quantizable")

    effective_group_size = group_size if granularity is Granularity.per_group else None
    default_config = int_config(
        default_bits,
        scheme=scheme,
        granularity=granularity,
        group_size=effective_group_size,
        method=method,
    )
    assignments: dict[str, QuantConfig] = {}
    for layer_name, bits in layer_bits.items():
        if bits == default_bits:
            continue
        assignments[layer_name] = int_config(
            bits,
            scheme=scheme,
            granularity=granularity,
            group_size=effective_group_size,
            method=method,
        )
    profile = PrecisionProfile(
        profile_id=_profile_id(
            name,
            {
                "default_bits": default_bits,
                "overrides": {k: v for k, v in sorted(layer_bits.items())},
                "runtime": runtime.value,
                "method": method.value,
            },
        ),
        name=name,
        description="Mixed-precision profile",
        default_config=default_config,
        assignments=assignments,
        runtime=runtime,
        method=method,
        deployable=False,
        evidence=EvidenceType.estimated,
    )
    issues = validate_profile(architecture, profile)
    if issues:
        raise UnsupportedQuantizationError("; ".join(issues))
    return profile


def validate_profile(
    architecture: ModelArchitecture, profile: PrecisionProfile
) -> tuple[str, ...]:
    """Return a tuple of compatibility issues (empty when the profile is valid)."""
    issues: list[str] = []
    cap = capability(profile.runtime)
    names = {layer.name: layer for layer in architecture.layers}

    for layer_name in profile.assignments:
        layer = names.get(layer_name)
        if layer is None:
            issues.append(f"assignment references unknown layer {layer_name!r}")
        elif not layer.quantizable:
            issues.append(f"layer {layer_name!r} is not quantizable")

    configs = {profile.default_config, *profile.assignments.values()}
    for config in configs:
        if config.is_float:
            if not cap.supports_bits(config.bits, is_float=True):
                issues.append(
                    f"{profile.runtime.value} does not support {config.bits}-bit float storage"
                )
            continue
        if not cap.supports_bits(config.bits, is_float=False):
            issues.append(
                f"{profile.runtime.value} does not declare {config.bits}-bit integer support"
            )
        if config.method not in cap.methods:
            issues.append(
                f"{profile.runtime.value} does not support method {config.method.value}"
            )
    return tuple(issues)


def assert_profile_supported(
    architecture: ModelArchitecture, profile: PrecisionProfile
) -> None:
    """Raise when ``profile`` cannot execute on its declared runtime."""
    issues = validate_profile(architecture, profile)
    if issues:
        raise CompatibilityError("; ".join(issues))
