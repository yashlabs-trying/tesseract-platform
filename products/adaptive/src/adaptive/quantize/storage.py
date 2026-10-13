"""Serialized storage accounting for precision profiles.

Accounts for *actual* serialized size: packed weight payload, scale factors,
zero points, per-tensor metadata, and parameters that remain at baseline
precision (norms, biases, tied/excluded embeddings).
"""

from __future__ import annotations

import math

from adaptive.models.architecture import LayerSpec, ModelArchitecture
from adaptive.models.enums import DType, Granularity, QuantScheme, dtype_bytes
from adaptive.models.quantization import PrecisionProfile, QuantConfig
from adaptive.models.storage import StorageBreakdown

TENSOR_METADATA_BYTES = 32
"""Fixed per-tensor serialization overhead (name, shape, dtype, offsets, hash)."""


def num_groups(layer: LayerSpec, config: QuantConfig) -> int:
    """Number of scale/zero-point groups for a quantized layer."""
    if config.granularity is Granularity.per_tensor:
        return 1
    out_features = layer.shape[0]
    in_features = layer.shape[-1]
    if config.granularity is Granularity.per_channel:
        return out_features
    group_size = config.group_size
    assert group_size is not None
    groups_per_row = math.ceil(in_features / group_size)
    return out_features * groups_per_row


def layer_storage(layer: LayerSpec, config: QuantConfig, baseline: DType) -> StorageBreakdown:
    """Storage breakdown for a single layer under ``config``."""
    baseline_bytes = layer.storage_bytes(baseline)
    if not layer.quantizable or config.is_float:
        return StorageBreakdown(
            unquantized_bytes=layer.storage_bytes(config.dtype if config.is_float else baseline),
            total_bytes=layer.storage_bytes(config.dtype if config.is_float else baseline),
            baseline_bytes=baseline_bytes,
        )

    payload_bits = layer.num_parameters * config.bits
    parameter_bytes = math.ceil(payload_bits / 8)
    groups = num_groups(layer, config)
    scale_bytes = math.ceil(groups * dtype_bytes(config.scale_dtype))
    zero_point_bytes = (
        math.ceil(groups * dtype_bytes(config.zero_point_dtype))
        if config.scheme is QuantScheme.asymmetric
        else 0
    )
    metadata_bytes = TENSOR_METADATA_BYTES
    total = parameter_bytes + scale_bytes + zero_point_bytes + metadata_bytes
    return StorageBreakdown(
        parameter_bytes=parameter_bytes,
        scale_bytes=scale_bytes,
        zero_point_bytes=zero_point_bytes,
        metadata_bytes=metadata_bytes,
        unquantized_bytes=0,
        total_bytes=total,
        baseline_bytes=baseline_bytes,
    )


def compute_storage(
    architecture: ModelArchitecture, profile: PrecisionProfile
) -> StorageBreakdown:
    """Aggregate serialized storage for an architecture under ``profile``."""
    total = StorageBreakdown()
    for layer in architecture.layers:
        config = profile.config_for(layer)
        total = total + layer_storage(layer, config, architecture.baseline_dtype)
    return total
