"""M1: reference quantization/dequantization correctness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adaptive.errors import ConfigurationError
from adaptive.models.enums import DType, Granularity, QuantScheme
from adaptive.models.quantization import int_config
from adaptive.quantize import (
    dequantize_tensor,
    pack_int4,
    packed_int4_bytes,
    quantization_error,
    quantize_tensor,
    unpack_int4,
)


def _weights(rows: int = 8, cols: int = 64, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator) * 0.5


@pytest.mark.parametrize("granularity", list(Granularity))
def test_symmetric_int8_round_trip(granularity: Granularity) -> None:
    group_size = 16 if granularity is Granularity.per_group else None
    config = int_config(
        8, scheme=QuantScheme.symmetric, granularity=granularity, group_size=group_size
    )
    weights = _weights()
    qt = quantize_tensor(weights, config)
    restored = dequantize_tensor(qt)
    assert restored.shape == weights.shape
    error = quantization_error(weights, qt)
    assert error["relative_error"] < 0.05
    assert qt.zero_points is None


def test_asymmetric_int8_has_zero_points() -> None:
    config = int_config(
        8, scheme=QuantScheme.asymmetric, granularity=Granularity.per_group, group_size=32
    )
    weights = _weights()
    qt = quantize_tensor(weights, config)
    assert qt.zero_points is not None
    error = quantization_error(weights, qt)
    assert error["relative_error"] < 0.05


def test_int4_round_trip_bounded_error() -> None:
    config = int_config(
        4, scheme=QuantScheme.symmetric, granularity=Granularity.per_group, group_size=32
    )
    weights = _weights()
    qt = quantize_tensor(weights, config)
    restored = dequantize_tensor(qt)
    assert restored.dtype == weights.dtype
    # rounding error per element is bounded by half a quantization step
    per_group_absmax = weights.abs().reshape(8, -1).amax(dim=1) / 7.0
    max_step = float(per_group_absmax.max())
    assert quantization_error(weights, qt)["max_abs_error"] <= max_step + 1e-6


def test_per_tensor_single_scale() -> None:
    config = int_config(8, granularity=Granularity.per_tensor)
    qt = quantize_tensor(_weights(), config)
    assert qt.scales.numel() == 1


def test_per_channel_scale_count() -> None:
    config = int_config(8, granularity=Granularity.per_channel)
    qt = quantize_tensor(_weights(rows=12), config)
    assert qt.scales.numel() == 12


def test_per_group_padding_handled() -> None:
    # 50 columns with group size 16 => padded to 64, 4 groups per row
    config = int_config(4, granularity=Granularity.per_group, group_size=16)
    weights = _weights(rows=3, cols=50)
    qt = quantize_tensor(weights, config)
    assert qt.padded_in_features == 14
    assert dequantize_tensor(qt).shape == weights.shape
    assert qt.scales.numel() == 3 * 4


def test_float_config_rejected_by_quantizer() -> None:
    from adaptive.models.quantization import baseline_config

    with pytest.raises(ConfigurationError):
        quantize_tensor(_weights(), baseline_config(DType.fp16))


def test_all_zero_tensor_is_stable() -> None:
    config = int_config(4, granularity=Granularity.per_group, group_size=16)
    qt = quantize_tensor(torch.zeros(4, 32), config)
    restored = dequantize_tensor(qt)
    assert torch.allclose(restored, torch.zeros(4, 32))


def test_int4_pack_unpack_round_trip() -> None:
    codes = torch.randint(-8, 8, (5, 7), generator=torch.Generator().manual_seed(1)).to(torch.int8)
    packed = pack_int4(codes)
    assert packed.dtype == torch.uint8
    assert packed.numel() == packed_int4_bytes(codes.numel())
    restored = unpack_int4(packed, codes.numel())
    assert torch.equal(restored, codes.reshape(-1))


def test_pack_int4_rejects_out_of_range() -> None:
    from adaptive.errors import UnsupportedQuantizationError

    with pytest.raises(UnsupportedQuantizationError):
        pack_int4(torch.tensor([9], dtype=torch.int8))


def test_quantize_dequantize_is_deterministic() -> None:
    config = int_config(4, granularity=Granularity.per_group, group_size=32)
    weights = _weights(seed=42)
    first = dequantize_tensor(quantize_tensor(weights, config))
    second = dequantize_tensor(quantize_tensor(weights, config))
    assert torch.equal(first, second)
