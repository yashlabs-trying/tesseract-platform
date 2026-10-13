"""Property-based tests for the reference quantizer and packing."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from hypothesis import given, settings
from hypothesis import strategies as st

from adaptive.models.enums import Granularity, QuantScheme
from adaptive.models.quantization import int_config
from adaptive.quantize import (
    dequantize_tensor,
    pack_int4,
    packed_int4_bytes,
    quantization_error,
    quantize_tensor,
    unpack_int4,
)

settings.register_profile("ci", max_examples=40, deadline=None)
settings.load_profile("ci")

_finite = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)


@given(
    rows=st.integers(1, 4),
    cols=st.integers(8, 32),
    data=st.data(),
)
def test_symmetric_int8_error_bounded(rows: int, cols: int, data: st.DataObject) -> None:
    values = data.draw(st.lists(_finite, min_size=rows * cols, max_size=rows * cols))
    weights = torch.tensor(values, dtype=torch.float32).reshape(rows, cols)
    config = int_config(8, granularity=Granularity.per_group, group_size=8)
    qt = quantize_tensor(weights, config)
    grouped_absmax = weights.abs().reshape(rows, -1).amax(dim=1) / 127.0
    bound = float(grouped_absmax.max()) + 1e-5
    assert quantization_error(weights, qt)["max_abs_error"] <= bound


@given(codes=st.lists(st.integers(-8, 7), min_size=1, max_size=64))
def test_pack_unpack_identity(codes: list[int]) -> None:
    tensor = torch.tensor(codes, dtype=torch.int8)
    packed = pack_int4(tensor)
    assert packed.numel() == packed_int4_bytes(len(codes))
    restored = unpack_int4(packed, len(codes))
    assert restored.tolist() == codes


@given(
    rows=st.integers(1, 4),
    cols=st.integers(8, 32),
    data=st.data(),
)
def test_requantizing_dequantized_is_stable(
    rows: int, cols: int, data: st.DataObject
) -> None:
    values = data.draw(st.lists(_finite, min_size=rows * cols, max_size=rows * cols))
    weights = torch.tensor(values, dtype=torch.float32).reshape(rows, cols)
    config = int_config(4, granularity=Granularity.per_group, group_size=16)
    first = quantize_tensor(weights, config)
    reconstructed = dequantize_tensor(first)
    second = quantize_tensor(reconstructed, config)
    assert torch.equal(first.data, second.data)


@given(bits=st.sampled_from([4, 8]), scheme=st.sampled_from(list(QuantScheme)))
def test_scheme_bits_matrix_round_trips(bits: int, scheme: QuantScheme) -> None:
    weights = torch.randn(6, 24, generator=torch.Generator().manual_seed(bits))
    config = int_config(bits, scheme=scheme, granularity=Granularity.per_group, group_size=8)
    qt = quantize_tensor(weights, config)
    restored = dequantize_tensor(qt)
    assert restored.shape == weights.shape
    assert torch.isfinite(restored).all()
