"""M1: precision profile generation and compatibility rejection."""

from __future__ import annotations

from typing import Any

import pytest

from adaptive.analyzer import ModelAnalyzer
from adaptive.errors import CompatibilityError, UnsupportedQuantizationError
from adaptive.models.enums import QuantMethod, RuntimeBackend
from adaptive.quantize import (
    assert_profile_supported,
    baseline_profile,
    capability,
    mixed_profile,
    supported_bits,
    uniform_profile,
    validate_profile,
)


def test_baseline_profile_matches_dtype(
    analyzer: ModelAnalyzer, llama_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = baseline_profile(arch)
    assert profile.default_config.bits == 16
    assert profile.is_uniform


def test_uniform_reference_profile_not_deployable(
    analyzer: ModelAnalyzer, llama_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = uniform_profile(arch, 4)
    assert profile.deployable is False
    assert "reference" in " ".join(profile.notes)


def test_uniform_profile_is_uniform(analyzer: ModelAnalyzer, llama_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = uniform_profile(arch, 8)
    assert profile.is_uniform
    assert profile.distinct_bits() == {8}


def test_mps_rejects_integer_profiles(
    analyzer: ModelAnalyzer, llama_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = uniform_profile(arch, 4, runtime=RuntimeBackend.mps)
    issues = validate_profile(arch, profile)
    assert issues
    with pytest.raises(CompatibilityError):
        assert_profile_supported(arch, profile)


def test_cuda_accepts_int4_gptq(analyzer: ModelAnalyzer, llama_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = uniform_profile(arch, 4, runtime=RuntimeBackend.cuda, method=QuantMethod.gptq)
    assert validate_profile(arch, profile) == ()


def test_mixed_profile_overrides(analyzer: ModelAnalyzer, tiny_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(tiny_config, model_id="tiny")
    target = "model.layers.0.self_attn.v_proj"
    profile = mixed_profile(arch, {target: 8}, default_bits=4)
    assert profile.config_for(arch.layer(target)).bits == 8
    assert profile.default_config.bits == 4
    assert not profile.is_uniform


def test_mixed_profile_rejects_unknown_layer(
    analyzer: ModelAnalyzer, tiny_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(tiny_config, model_id="tiny")
    with pytest.raises(UnsupportedQuantizationError):
        mixed_profile(arch, {"model.layers.99.self_attn.q_proj": 8}, default_bits=4)


def test_mixed_profile_rejects_non_quantizable_layer(
    analyzer: ModelAnalyzer, tiny_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(tiny_config, model_id="tiny")
    with pytest.raises(UnsupportedQuantizationError):
        mixed_profile(arch, {"model.layers.0.input_layernorm": 4}, default_bits=4)


def test_supported_bits_reported() -> None:
    assert 4 in supported_bits(RuntimeBackend.cuda)
    assert 4 not in supported_bits(RuntimeBackend.mps)
    assert capability(RuntimeBackend.mps).production is False


def test_validate_flags_unknown_runtime_method(
    analyzer: ModelAnalyzer, llama_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(llama_config, model_id="m")
    profile = uniform_profile(
        arch, 4, runtime=RuntimeBackend.vllm, method=QuantMethod.reference
    )
    issues = validate_profile(arch, profile)
    assert any("method" in issue for issue in issues)


def test_non_quantizable_layers_stay_baseline(
    analyzer: ModelAnalyzer, tiny_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(tiny_config, model_id="tiny")
    profile = uniform_profile(arch, 4)
    norm = arch.layer("model.layers.0.input_layernorm")
    assert profile.config_for(norm).bits == 16
