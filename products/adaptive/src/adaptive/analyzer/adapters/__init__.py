"""Built-in architecture adapters."""

from __future__ import annotations

from adaptive.analyzer.adapters.dense import DenseTransformerAdapter
from adaptive.analyzer.adapters.llama import LlamaAdapter
from adaptive.analyzer.adapters.mistral import MistralAdapter
from adaptive.analyzer.adapters.qwen import Qwen2Adapter, Qwen3Adapter

__all__ = [
    "DenseTransformerAdapter",
    "LlamaAdapter",
    "MistralAdapter",
    "Qwen2Adapter",
    "Qwen3Adapter",
]
