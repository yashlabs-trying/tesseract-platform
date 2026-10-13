"""Mistral family adapter."""

from __future__ import annotations

from adaptive.analyzer.adapters.dense import DenseTransformerAdapter
from adaptive.models.enums import ModelFamily


class MistralAdapter(DenseTransformerAdapter):
    """Mistral / Mixtral-style dense decoder adapter (dense blocks only)."""

    name = "mistral"
    model_types = ("mistral",)
    family = ModelFamily.mistral
    default_attention_bias = False
    has_qk_norm = False
    default_tie_word_embeddings = False
