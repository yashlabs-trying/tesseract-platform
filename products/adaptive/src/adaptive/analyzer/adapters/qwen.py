"""Qwen2 / Qwen3 family adapters."""

from __future__ import annotations

from adaptive.analyzer.adapters.dense import DenseTransformerAdapter
from adaptive.models.enums import ModelFamily


class Qwen2Adapter(DenseTransformerAdapter):
    """Qwen2 / Qwen2.5 dense decoder adapters."""

    name = "qwen2"
    model_types = ("qwen2",)
    family = ModelFamily.qwen2
    default_attention_bias = True
    has_qk_norm = False
    default_tie_word_embeddings = False


class Qwen3Adapter(DenseTransformerAdapter):
    """Qwen3 dense decoder adapters (adds per-head QK RMSNorm)."""

    name = "qwen3"
    model_types = ("qwen3",)
    family = ModelFamily.qwen3
    default_attention_bias = False
    has_qk_norm = True
    default_tie_word_embeddings = False
