"""Shared pytest fixtures for Tesseract Adaptive tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adaptive.analyzer import ModelAnalyzer

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "models"


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text())


@pytest.fixture
def llama_config() -> dict[str, Any]:
    return _load("llama-3.2-1b.json")


@pytest.fixture
def qwen2_config() -> dict[str, Any]:
    return _load("qwen2.5-0.5b.json")


@pytest.fixture
def qwen3_config() -> dict[str, Any]:
    return _load("qwen3-0.6b.json")


@pytest.fixture
def tiny_config() -> dict[str, Any]:
    """A tiny deterministic Llama-style config for fast unit tests."""
    return {
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
    }


@pytest.fixture
def analyzer() -> ModelAnalyzer:
    return ModelAnalyzer()


@pytest.fixture
def tiny_architecture(analyzer: ModelAnalyzer, tiny_config: dict[str, Any]):
    return analyzer.analyze(tiny_config, model_id="tiny")


@pytest.fixture
def tiny_model(tiny_architecture):
    pytest.importorskip("torch")
    from adaptive.nets import ReferenceTransformer

    return ReferenceTransformer.from_architecture(tiny_architecture, seed=0)


@pytest.fixture
def calibration_dataset(tiny_config: dict[str, Any]):
    from adaptive.quality import DatasetSplit, make_synthetic_dataset

    return make_synthetic_dataset(
        "calibration",
        DatasetSplit.calibration,
        vocab_size=tiny_config["vocab_size"],
        num_sequences=4,
        seq_len=16,
        seed=1,
    )


@pytest.fixture
def evaluation_dataset(tiny_config: dict[str, Any]):
    from adaptive.quality import DatasetSplit, make_synthetic_dataset

    return make_synthetic_dataset(
        "evaluation",
        DatasetSplit.evaluation,
        vocab_size=tiny_config["vocab_size"],
        num_sequences=3,
        seq_len=16,
        seed=2,
    )
