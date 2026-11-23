"""Llama family adapter."""

from __future__ import annotations

from adaptive.analyzer.adapters.dense import DenseTransformerAdapter
from adaptive.models.enums import ModelFamily


class LlamaAdapter(DenseTransformerAdapter):
    """Meta Llama / Llama-2 / Llama-3 dense decoder adapters."""

    name = "llama"
    model_types = ("llama",)
    family = ModelFamily.llama
    default_attention_bias = False
    has_qk_norm = False
    default_tie_word_embeddings = False
