"""Phase 3 / B04 - does sensitivity-guided mixed precision beat baselines?

Compares, at a matched number of 8-bit-upgraded layers:
  * sensitivity-guided (activation-weighted)
  * random selection
  * heuristic (all MLP layers)
against uniform 4-bit, on a real model using the reference (emulated) quantizer.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive.analyzer import ModelAnalyzer
from adaptive.quality import DatasetSplit, TokenDataset
from adaptive.quality.metrics import perplexity
from adaptive.quantize import compute_storage, mixed_profile
from adaptive.quantize.model import quantize_model
from adaptive.sensitivity import SensitivityMethod, analyze_sensitivity

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "fixtures" / "corpus" / "alice.txt"


def _dataset(token_ids: list[int], seq_len: int, num_chunks: int):
    chunks = [
        tuple(token_ids[i : i + seq_len])
        for i in range(0, len(token_ids) - seq_len, seq_len)
    ][:num_chunks]
    return TokenDataset(name="alice", split=DatasetSplit.evaluation, sequences=tuple(chunks))


def main() -> None:
    model_id = sys.argv[1]
    revision = sys.argv[2] if len(sys.argv) > 2 else "main"
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    num_chunks = int(sys.argv[4]) if len(sys.argv) > 4 else 6

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32)
    model.eval()
    architecture = ModelAnalyzer().analyze(model.config, model_id=model_id, revision=revision)

    token_ids = tokenizer(CORPUS.read_text())["input_ids"]
    evaluation = _dataset(token_ids, seq_len, num_chunks)

    def logits_fn(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(batch).logits

    baseline = perplexity(logits_fn, evaluation)

    quant_layers = [
        s.name for s in architecture.layers if s.quantizable and len(s.shape) == 2
    ]
    k = max(len(quant_layers) // 4, 1)

    report = analyze_sensitivity(
        architecture,
        model,
        evaluation.sequences,
        methods=(SensitivityMethod.activation_weighted,),
        bits=4,
        group_size=128,
        batch_size=1,
        max_batches=4,
    )
    guided = report.top_k(SensitivityMethod.activation_weighted, k)

    rng = random.Random(0)
    random_choice = sorted(rng.sample(quant_layers, k))

    mlp_layers = [
        s.name for s in architecture.layers if s.quantizable and s.family.value == "mlp"
    ]

    def evaluate(name: str, bits_map: dict[str, int]) -> dict[str, object]:
        profile = mixed_profile(
            architecture, bits_map, default_bits=4, group_size=128, name=name
        )
        candidate = quantize_model(architecture, profile, model)
        candidate.eval()

        def cand_fn(batch: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return candidate(batch).logits

        ppl = perplexity(cand_fn, evaluation, baseline_value=baseline.value)
        storage = compute_storage(architecture, profile)
        return {
            "name": name,
            "upgraded_layers": len(bits_map),
            "perplexity": ppl.value,
            "relative_degradation": ppl.relative_degradation(),
            "serialized_mb": storage.total_mb,
            "compression_ratio": storage.compression_ratio,
        }

    results = {
        "model_id": model_id,
        "revision": revision,
        "baseline_perplexity": baseline.value,
        "num_tokens": baseline.num_samples,
        "upgraded_layer_count": k,
        "uniform_4": evaluate("uniform-4", {}),
        "sensitivity_guided": evaluate("sensitivity-guided", {n: 8 for n in guided}),
        "random": evaluate("random", {n: 8 for n in random_choice}),
        "heuristic_mlp": evaluate("heuristic-mlp", {n: 8 for n in mlp_layers}),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
