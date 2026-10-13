"""Phase 2 - mathematical quantization correctness (A01-A05).

Rigorous numerical tests, including adversarial inputs and comparison against an
independently implemented NumPy reference.
"""

from __future__ import annotations

import numpy as np
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


def _cfg(bits: int, granularity: Granularity, group_size: int | None = None,
         scheme: QuantScheme = QuantScheme.symmetric):
    return int_config(bits, scheme=scheme, granularity=granularity, group_size=group_size)


# ---------------------------------------------------------------- A01 schemes

@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "granularity,group_size",
    [
        (Granularity.per_tensor, None),
        (Granularity.per_channel, None),
        (Granularity.per_group, 32),
    ],
)
def test_round_trip_shapes_and_finiteness(bits, granularity, group_size) -> None:
    w = torch.randn(6, 96, generator=torch.Generator().manual_seed(bits))
    qt = quantize_tensor(w, _cfg(bits, granularity, group_size))
    restored = dequantize_tensor(qt)
    assert restored.shape == w.shape
    assert torch.isfinite(restored).all()


@pytest.mark.parametrize("bits", [4, 8])
def test_asymmetric_round_trip(bits) -> None:
    w = torch.randn(4, 64, generator=torch.Generator().manual_seed(bits)) + 3.0
    qt = quantize_tensor(
        w, _cfg(bits, Granularity.per_group, 16, QuantScheme.asymmetric)
    )
    assert qt.zero_points is not None
    err = quantization_error(w, qt)
    assert err["relative_error"] < 0.1


# -------------------------------------------------- A03 numerical stability

def test_non_divisible_group_dimensions() -> None:
    w = torch.randn(3, 50, generator=torch.Generator().manual_seed(1))
    qt = quantize_tensor(w, _cfg(4, Granularity.per_group, 16))
    assert qt.padded_in_features == 14
    assert dequantize_tensor(qt).shape == w.shape


def test_all_zero_tensor() -> None:
    qt = quantize_tensor(torch.zeros(4, 32), _cfg(4, Granularity.per_group, 16))
    assert torch.allclose(dequantize_tensor(qt), torch.zeros(4, 32))


def test_constant_tensor_is_exact() -> None:
    w = torch.full((2, 32), 0.7)
    qt = quantize_tensor(w, _cfg(8, Granularity.per_tensor))
    err = quantization_error(w, qt)
    # a constant is representable exactly (scale = value/qmax)
    assert err["max_abs_error"] < 1e-3


def test_tiny_magnitudes_do_not_underflow() -> None:
    w = torch.full((2, 16), 1e-9)
    qt = quantize_tensor(w, _cfg(8, Granularity.per_group, 8))
    restored = dequantize_tensor(qt)
    assert torch.isfinite(restored).all()
    assert restored.abs().max() <= 1e-8


def test_extreme_magnitudes() -> None:
    w = torch.tensor([[1e30, -1e30, 0.0, 5.0]])
    qt = quantize_tensor(w, _cfg(8, Granularity.per_tensor))
    restored = dequantize_tensor(qt)
    assert torch.isfinite(restored).all()
    assert torch.allclose(restored[0, 0], torch.tensor(1e30), rtol=0.02)


def test_nan_input_rejected() -> None:
    w = torch.tensor([[1.0, float("nan"), 2.0, 3.0]])
    with pytest.raises((ConfigurationError, ValueError)):
        quantize_tensor(w, _cfg(8, Granularity.per_tensor))


def test_inf_input_rejected() -> None:
    w = torch.tensor([[1.0, float("inf"), 2.0, 3.0]])
    with pytest.raises((ConfigurationError, ValueError)):
        quantize_tensor(w, _cfg(8, Granularity.per_tensor))


def test_non_contiguous_tensor() -> None:
    base = torch.randn(8, 64, generator=torch.Generator().manual_seed(2))
    w = base.t().contiguous().t()[:, ::2]
    assert not w.is_contiguous()
    qt = quantize_tensor(w, _cfg(4, Granularity.per_group, 8))
    assert dequantize_tensor(qt).shape == w.shape


@pytest.mark.parametrize("bits", [4, 8])
def test_repeated_cycles_are_stable(bits) -> None:
    w = torch.randn(4, 32, generator=torch.Generator().manual_seed(bits))
    cfg = _cfg(bits, Granularity.per_group, 16)
    qt1 = quantize_tensor(w, cfg)
    d1 = dequantize_tensor(qt1)
    qt2 = quantize_tensor(d1, cfg)
    d2 = dequantize_tensor(qt2)
    assert torch.equal(d1, d2)


def test_determinism() -> None:
    w = torch.randn(5, 40, generator=torch.Generator().manual_seed(7))
    cfg = _cfg(4, Granularity.per_group, 8)
    assert torch.equal(quantize_tensor(w, cfg).data, quantize_tensor(w, cfg).data)


# ------------------------------------------------- A04 reconstruction accuracy

def test_cosine_similarity_high() -> None:
    w = torch.randn(8, 128, generator=torch.Generator().manual_seed(3))
    restored = dequantize_tensor(quantize_tensor(w, _cfg(8, Granularity.per_group, 64)))
    cos = torch.nn.functional.cosine_similarity(w.flatten(), restored.flatten(), dim=0)
    assert cos.item() > 0.999


def test_error_metrics_monotone_in_quality() -> None:
    w = torch.randn(8, 128, generator=torch.Generator().manual_seed(4))
    e4 = quantization_error(w, quantize_tensor(w, _cfg(4, Granularity.per_group, 32)))
    e8 = quantization_error(w, quantize_tensor(w, _cfg(8, Granularity.per_group, 32)))
    assert e8["mse"] < e4["mse"]
    assert e8["max_abs_error"] <= e4["max_abs_error"]


def test_activation_weighted_error_defined() -> None:
    w = torch.randn(8, 64, generator=torch.Generator().manual_seed(5))
    restored = dequantize_tensor(quantize_tensor(w, _cfg(4, Granularity.per_channel)))
    act = torch.rand(64, generator=torch.Generator().manual_seed(6))
    awre = ((w - restored).pow(2) * act).sum().sqrt()
    assert float(awre) >= 0.0


# ------------------------------------------------- independent reference check

def _reference_symmetric_per_group(w: np.ndarray, bits: int, group_size: int):
    out, in_features = w.shape
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    pad = (-in_features) % group_size
    wp = np.pad(w, ((0, 0), (0, pad)))
    groups = wp.reshape(out, wp.shape[1] // group_size, group_size)
    scale = np.maximum(np.abs(groups).max(axis=2) / qmax, 1e-12)
    codes = np.clip(np.round(groups / scale[..., None]), qmin, qmax)
    deq = (codes * scale[..., None]).reshape(out, -1)[:, :in_features]
    return deq


@pytest.mark.parametrize("bits", [4, 8])
def test_matches_independent_numpy_reference(bits) -> None:
    rng = np.random.default_rng(bits)
    w = rng.standard_normal((5, 70)).astype(np.float32)
    expected = _reference_symmetric_per_group(w, bits, 16)
    got = dequantize_tensor(
        quantize_tensor(torch.from_numpy(w), _cfg(bits, Granularity.per_group, 16))
    ).numpy()
    assert np.allclose(got, expected, atol=1e-4)


# ------------------------------------------------- A05 precision monotonicity

def test_precision_monotonicity_within_grouping() -> None:
    rng = np.random.default_rng(0)
    w = torch.from_numpy(rng.standard_normal((6, 64)).astype(np.float32))
    for granularity, group_size in (
        (Granularity.per_tensor, None),
        (Granularity.per_channel, None),
        (Granularity.per_group, 16),
    ):
        e4 = quantization_error(w, quantize_tensor(w, _cfg(4, granularity, group_size)))["mse"]
        e8 = quantization_error(w, quantize_tensor(w, _cfg(8, granularity, group_size)))["mse"]
        assert e8 <= e4, f"{granularity}/{group_size}: int8 {e8} > int4 {e4}"


def test_finer_groups_reduce_error() -> None:
    rng = np.random.default_rng(11)
    w = torch.from_numpy(rng.standard_normal((8, 128)).astype(np.float32))
    coarse = quantization_error(w, quantize_tensor(w, _cfg(4, Granularity.per_group, 128)))["mse"]
    fine = quantization_error(w, quantize_tensor(w, _cfg(4, Granularity.per_group, 16)))["mse"]
    assert fine <= coarse


# ------------------------------------------------- packing edge cases

def test_int4_packing_odd_length() -> None:
    codes = torch.tensor([-8, -1, 0, 7, 3], dtype=torch.int8)
    packed = pack_int4(codes)
    assert packed.numel() == packed_int4_bytes(codes.numel())
    assert torch.equal(unpack_int4(packed, codes.numel()), codes)


def test_int4_packing_all_values() -> None:
    codes = torch.arange(-8, 8, dtype=torch.int8)
    assert torch.equal(unpack_int4(pack_int4(codes), codes.numel()), codes)


def test_float_config_rejected() -> None:
    from adaptive.models.quantization import baseline_config

    with pytest.raises(ConfigurationError):
        quantize_tensor(torch.randn(2, 2), baseline_config(DType.fp16))
