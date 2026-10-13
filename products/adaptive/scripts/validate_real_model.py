"""Phase 3 - real-model quality validation (emulated reference quantizer).

Loads a pinned Hugging Face causal LM, evaluates baseline perplexity on a
held-out corpus, then evaluates uniform 8-bit, uniform 4-bit and two mixed
profiles using Adaptive's *reference* quantizer (weights are dequantized before
matmul). This is explicitly NOT a production kernel benchmark.

Usage:
    python scripts/validate_real_model.py <model_id> <revision> [seq_len] [num_chunks]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive.analyzer import ModelAnalyzer
from adaptive.models.quantization import PrecisionProfile
from adaptive.quality import DatasetSplit, TokenDataset
from adaptive.quality.metrics import perplexity
from adaptive.quantize import (
    compute_storage,
    mixed_profile,
    uniform_profile,
)
from adaptive.quantize.model import quantize_model

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "fixtures" / "corpus" / "alice.txt"


def _dataset(token_ids: list[int], seq_len: int, num_chunks: int, split: DatasetSplit):
    chunks = [
        tuple(token_ids[i : i + seq_len])
        for i in range(0, len(token_ids) - seq_len, seq_len)
    ][:num_chunks]
    return TokenDataset(name="alice", split=split, sequences=tuple(chunks), seed=0)


def main() -> None:
    model_id = sys.argv[1]
    revision = sys.argv[2] if len(sys.argv) > 2 else "main"
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    num_chunks = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.float32
    )
    model.eval()

    architecture = ModelAnalyzer().analyze(
        model.config, model_id=model_id, revision=revision
    )

    text = CORPUS.read_text()
    token_ids = tokenizer(text)["input_ids"]
    evaluation = _dataset(token_ids, seq_len, num_chunks, DatasetSplit.evaluation)

    def logits_fn(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(batch).logits

    baseline = perplexity(logits_fn, evaluation)

    profiles: list[tuple[str, PrecisionProfile]] = [
        ("uniform-8", uniform_profile(architecture, 8, group_size=128)),
        ("uniform-4", uniform_profile(architecture, 4, group_size=128)),
        (
            "mixed-mlp8",
            mixed_profile(
                architecture,
                {s.name: 8 for s in architecture.layers if s.family.value == "mlp"},
                default_bits=4,
                group_size=128,
                name="mixed-mlp8",
            ),
        ),
    ]

    results = {
        "model_id": model_id,
        "revision": revision,
        "parameters": architecture.total_parameters,
        "baseline_perplexity": baseline.value,
        "num_tokens": baseline.num_samples,
        "evidence": "measured-emulated-reference",
        "profiles": [],
    }

    for name, profile in profiles:
        started = time.time()
        candidate = quantize_model(architecture, profile, model)
        candidate.eval()

        def cand_fn(batch: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return candidate(batch).logits

        ppl = perplexity(cand_fn, evaluation, baseline_value=baseline.value)
        storage = compute_storage(architecture, profile)
        results["profiles"].append(
            {
                "name": name,
                "bits": sorted(profile.distinct_bits()),
                "perplexity": ppl.value,
                "relative_degradation": ppl.relative_degradation(),
                "serialized_mb": storage.total_mb,
                "compression_ratio": storage.compression_ratio,
                "quantize_seconds": time.time() - started,
            }
        )
        del candidate

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
