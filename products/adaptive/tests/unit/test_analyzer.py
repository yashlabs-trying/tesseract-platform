"""M1: model architecture detection and parameter accounting."""

from __future__ import annotations

from typing import Any

import pytest

from adaptive.analyzer import AdapterRegistry, ModelAnalyzer
from adaptive.analyzer.adapters import LlamaAdapter
from adaptive.errors import ConfigurationError, UnsupportedModelError
from adaptive.models.enums import LayerFamily, LayerRole, ModelFamily


def test_llama_detection(analyzer: ModelAnalyzer, llama_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(llama_config, model_id="meta-llama/Llama-3.2-1B")
    assert arch.family is ModelFamily.llama
    assert arch.num_layers == 16
    assert arch.hidden_size == 2048
    assert arch.num_key_value_heads == 8
    assert arch.head_dim == 64
    assert arch.tie_word_embeddings is True


def test_qwen2_detection(analyzer: ModelAnalyzer, qwen2_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(qwen2_config, model_id="Qwen/Qwen2.5-0.5B")
    assert arch.family is ModelFamily.qwen2
    assert arch.num_layers == 24
    assert arch.attention_bias is True
    # qwen2 has no per-head qk norm
    assert not any(layer.role is LayerRole.attention_q_norm for layer in arch.layers)


def test_qwen3_has_qk_norm(analyzer: ModelAnalyzer, qwen3_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(qwen3_config, model_id="Qwen/Qwen3-0.6B")
    assert arch.family is ModelFamily.qwen3
    q_norms = [layer for layer in arch.layers if layer.role is LayerRole.attention_q_norm]
    assert len(q_norms) == arch.num_layers
    assert all(not layer.quantizable for layer in q_norms)


def test_unknown_model_type_rejected(analyzer: ModelAnalyzer) -> None:
    with pytest.raises(UnsupportedModelError):
        analyzer.analyze({"model_type": "mamba"}, model_id="state-space/x")


def test_missing_required_keys_rejected(analyzer: ModelAnalyzer) -> None:
    with pytest.raises(ConfigurationError):
        analyzer.analyze({"model_type": "llama", "hidden_size": 64}, model_id="broken/x")


def test_invalid_head_divisibility_rejected(analyzer: ModelAnalyzer) -> None:
    config = {
        "model_type": "llama",
        "hidden_size": 100,
        "num_hidden_layers": 1,
        "num_attention_heads": 3,
        "vocab_size": 32,
    }
    with pytest.raises(ConfigurationError):
        analyzer.analyze(config, model_id="broken/divisibility")


def test_layer_structure(analyzer: ModelAnalyzer, tiny_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(tiny_config, model_id="tiny")
    names = {layer.name for layer in arch.layers}
    assert "model.embed_tokens" in names
    assert "model.norm" in names
    assert "lm_head" in names
    assert "model.layers.0.self_attn.q_proj" in names
    assert "model.layers.1.mlp.down_proj" in names
    # 2 blocks * (2 norms + 4 attn + 3 mlp) + embed + final norm + lm_head
    assert len(arch.layers) == 2 * 9 + 3


def test_parameter_accounting_exact(
    analyzer: ModelAnalyzer, llama_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(llama_config, model_id="meta-llama/Llama-3.2-1B")
    # hand-computed total for the fixture
    assert arch.total_parameters == 1_235_814_400


def test_parameter_accounting_reconstructs_from_layers(
    analyzer: ModelAnalyzer, llama_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(llama_config, model_id="meta-llama/Llama-3.2-1B")
    assert arch.total_parameters == sum(layer.num_parameters for layer in arch.layers)
    by_family = arch.parameter_count_by_family()
    assert by_family[LayerFamily.mlp] > by_family[LayerFamily.attention]
    assert by_family[LayerFamily.embedding] > 0
    assert by_family[LayerFamily.norm] > 0


def test_baseline_weight_bytes(analyzer: ModelAnalyzer, llama_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(llama_config, model_id="meta-llama/Llama-3.2-1B")
    assert arch.baseline_weight_bytes == arch.total_parameters * 2  # bf16


def test_norms_not_quantizable(analyzer: ModelAnalyzer, tiny_config: dict[str, Any]) -> None:
    arch = analyzer.analyze(tiny_config, model_id="tiny")
    for layer in arch.layers:
        if layer.family is LayerFamily.norm:
            assert not layer.quantizable


def test_structural_fingerprint_stable(
    analyzer: ModelAnalyzer, llama_config: dict[str, Any]
) -> None:
    arch = analyzer.analyze(llama_config, model_id="meta-llama/Llama-3.2-1B")
    assert arch.structural_fingerprint() == arch.structural_fingerprint()
    assert len(arch.structural_fingerprint()) == 64


def test_registry_custom_adapter(llama_config: dict[str, Any]) -> None:
    registry = AdapterRegistry([LlamaAdapter()])
    assert registry.find(llama_config).name == "llama"
    with pytest.raises(UnsupportedModelError):
        registry.find({"model_type": "qwen2"})
