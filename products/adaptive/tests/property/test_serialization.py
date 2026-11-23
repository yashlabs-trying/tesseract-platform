"""Property-based tests for domain model serialization correctness."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from adaptive.models.architecture import LayerSpec
from adaptive.models.enums import (
    DType,
    Granularity,
    LayerFamily,
    LayerRole,
    QuantScheme,
    RuntimeBackend,
)
from adaptive.models.quantization import PrecisionProfile, int_config
from adaptive.models.storage import StorageBreakdown
from adaptive.schemas import profile_document

settings.register_profile("ci", max_examples=50, deadline=None)
settings.load_profile("ci")


@given(
    bits=st.sampled_from([4, 8]),
    scheme=st.sampled_from(list(QuantScheme)),
    group_size=st.sampled_from([8, 16, 32, 64, 128]),
)
def test_quant_config_round_trip(bits: int, scheme: QuantScheme, group_size: int) -> None:
    config = int_config(
        bits, scheme=scheme, granularity=Granularity.per_group, group_size=group_size
    )
    dumped = config.model_dump(mode="json")
    restored = type(config).model_validate(dumped)
    assert restored == config


@given(bits=st.sampled_from([4, 8, 16, 32]), total=st.integers(0, 10**9))
def test_storage_breakdown_round_trip(bits: int, total: int) -> None:
    breakdown = StorageBreakdown(
        parameter_bytes=total,
        scale_bytes=total // 10,
        total_bytes=total,
        baseline_bytes=total * 2,
    )
    restored = StorageBreakdown.model_validate(breakdown.model_dump())
    assert restored == breakdown


@given(bits=st.sampled_from([4, 8]))
def test_profile_document_serializable(bits: int) -> None:
    config = int_config(bits, granularity=Granularity.per_channel)
    profile = PrecisionProfile(
        profile_id="p1",
        name="p1",
        default_config=config,
        assignments={"model.layers.0.mlp.up_proj": int_config(8)},
        runtime=RuntimeBackend.cuda,
    )
    document = profile_document(profile)
    assert document["schema_version"] == "tesseract.adaptive.precision-profile.v1"
    assert document["default_config"]["bits"] == bits
    assert "model.layers.0.mlp.up_proj" in document["assignments"]


@given(
    out=st.integers(1, 64),
    in_dim=st.integers(1, 64),
    bias=st.booleans(),
)
def test_layer_spec_parameter_accounting(out: int, in_dim: int, bias: bool) -> None:
    layer = LayerSpec(
        name="x",
        role=LayerRole.mlp_up,
        family=LayerFamily.mlp,
        shape=(out, in_dim),
        num_parameters=out * in_dim + (out if bias else 0),
        bias=bias,
    )
    assert layer.out_features == out
    assert layer.in_features == in_dim
    assert layer.storage_bytes(DType.fp16) == layer.num_parameters * 2
