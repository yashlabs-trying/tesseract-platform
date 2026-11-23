"""M1: weight storage accounting."""

from __future__ import annotations

from typing import Any

from adaptive.analyzer import ModelAnalyzer
from adaptive.models.enums import Granularity, QuantScheme
from adaptive.models.quantization import int_config
from adaptive.quantize import baseline_profile, compute_storage, num_groups, uniform_profile


def test_baseline_storage_equals_weight_bytes(
    analyzer: ModelAnalyzer, llama_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = baseline_profile(arch)
    storage = compute_storage(arch, profile)
    assert storage.total_bytes == arch.baseline_weight_bytes
    assert storage.compression_ratio == 1.0
    assert storage.unquantized_bytes == arch.baseline_weight_bytes


def test_uniform_int4_compresses(analyzer: ModelAnalyzer, llama_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = uniform_profile(arch, 4, granularity=Granularity.per_group, group_size=128)
    storage = compute_storage(arch, profile)
    assert storage.total_bytes < arch.baseline_weight_bytes
    assert storage.compression_ratio > 1.5
    assert storage.parameter_bytes > 0
    assert storage.scale_bytes > 0
    assert storage.metadata_bytes > 0
    assert storage.zero_point_bytes == 0  # symmetric


def test_asymmetric_accounts_zero_points(
    analyzer: ModelAnalyzer, llama_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = uniform_profile(
        arch,
        4,
        scheme=QuantScheme.asymmetric,
        granularity=Granularity.per_group,
        group_size=128,
    )
    storage = compute_storage(arch, profile)
    assert storage.zero_point_bytes > 0


def test_quantized_parameter_bytes(analyzer: ModelAnalyzer, llama_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = uniform_profile(arch, 4, granularity=Granularity.per_channel)
    storage = compute_storage(arch, profile)
    import math

    expected = math.ceil(arch.quantizable_parameters * 4 / 8)
    assert storage.parameter_bytes == expected


def test_unquantized_params_retained(analyzer: ModelAnalyzer, llama_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = uniform_profile(arch, 4, granularity=Granularity.per_channel)
    storage = compute_storage(arch, profile)
    non_quantizable = arch.total_parameters - arch.quantizable_parameters
    assert storage.unquantized_bytes == non_quantizable * 2  # bf16


def test_num_groups_per_group(analyzer: ModelAnalyzer, llama_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    q_proj = arch.layer("model.layers.0.self_attn.q_proj")
    config = int_config(4, granularity=Granularity.per_group, group_size=128)
    assert num_groups(q_proj, config) == (2048 // 128) * 2048


def test_num_groups_per_channel(analyzer: ModelAnalyzer, llama_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    k_proj = arch.layer("model.layers.0.self_attn.k_proj")
    config = int_config(8, granularity=Granularity.per_channel)
    assert num_groups(k_proj, config) == 512


def test_lower_bits_use_less_payload(analyzer: ModelAnalyzer, llama_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    p4 = uniform_profile(arch, 4, granularity=Granularity.per_group, group_size=128)
    p8 = uniform_profile(arch, 8, granularity=Granularity.per_group, group_size=128)
    s4 = compute_storage(arch, p4)
    s8 = compute_storage(arch, p8)
    assert s4.parameter_bytes < s8.parameter_bytes
    assert s4.total_bytes < s8.total_bytes


def test_storage_breakdown_is_additive(
    analyzer: ModelAnalyzer, llama_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = uniform_profile(arch, 4, granularity=Granularity.per_group, group_size=128)
    total = compute_storage(arch, profile)
    recomputed = (
        total.parameter_bytes
        + total.scale_bytes
        + total.zero_point_bytes
        + total.metadata_bytes
        + total.unquantized_bytes
    )
    assert total.total_bytes == recomputed
