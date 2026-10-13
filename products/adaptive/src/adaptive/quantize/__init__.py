"""Reference quantization, profile generation, and storage accounting."""

from __future__ import annotations

from adaptive.quantize.packing import pack_int4, packed_int4_bytes, unpack_int4
from adaptive.quantize.profiles import (
    RUNTIME_CAPABILITIES,
    RuntimeCapability,
    assert_profile_supported,
    baseline_profile,
    capability,
    mixed_profile,
    supported_bits,
    uniform_profile,
    validate_profile,
)
from adaptive.quantize.reference import (
    QuantizedTensor,
    dequantize_tensor,
    quantization_error,
    quantize_tensor,
)
from adaptive.quantize.storage import compute_storage, layer_storage, num_groups

__all__ = [
    "RUNTIME_CAPABILITIES",
    "QuantizedTensor",
    "RuntimeCapability",
    "assert_profile_supported",
    "baseline_profile",
    "capability",
    "compute_storage",
    "dequantize_tensor",
    "layer_storage",
    "mixed_profile",
    "num_groups",
    "pack_int4",
    "packed_int4_bytes",
    "quantization_error",
    "quantize_tensor",
    "supported_bits",
    "uniform_profile",
    "unpack_int4",
    "validate_profile",
]
