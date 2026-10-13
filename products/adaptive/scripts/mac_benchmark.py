"""Phase 9 - reproducible CPU benchmarks (Apple MPS unavailable here).

Measures quantization, optimizer, controller and quality timings on the
synthetic reference transformer. Reports CPU numbers only; it does not claim
Apple MPS or NVIDIA performance.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from adaptive.analyzer import ModelAnalyzer
from adaptive.controller import Controller, ControllerPolicy, ProfileOption
from adaptive.models.optimization import OptimizationConstraints
from adaptive.nets import ReferenceTransformer
from adaptive.optimizer import AnalyticCostModel, optimize
from adaptive.quality import DatasetSplit, TokenDataset
from adaptive.quality.metrics import perplexity
from adaptive.quantize import compute_storage, quantize_tensor, uniform_profile
from adaptive.quantize.model import quantize_model
from adaptive.sensitivity import SensitivityMethod, analyze_sensitivity

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "models" / "tiny-llama.json"


def _time(fn, repeats: int = 5) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
    }


def main() -> None:
    torch.manual_seed(0)
    arch = ModelAnalyzer().analyze_file(FIXTURE, model_id="tiny-llama")
    model = ReferenceTransformer.from_architecture(arch, seed=0)

    token_ids = list(range(1, arch.vocab_size)) * 4
    sequences = tuple(
        tuple(token_ids[i : i + 16]) for i in range(0, 128, 16)
    )
    evaluation = TokenDataset(
        name="bench", split=DatasetSplit.evaluation, sequences=sequences, seed=0
    )

    report: dict[str, object] = {
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "machine": "x86_64 (WSL2, CPU)",
        },
        "model_parameters": arch.total_parameters,
    }

    report["analyze"] = _time(lambda: ModelAnalyzer().analyze_file(FIXTURE, model_id="x"), 20)

    profile4 = uniform_profile(arch, 4, group_size=16)
    profile8 = uniform_profile(arch, 8, group_size=16)

    layer = dict(model.linear_layers())["model.layers.0.mlp.up_proj"].weight.detach()
    report["quantize_tensor_int4"] = _time(
        lambda: quantize_tensor(layer, profile4.default_config), 20
    )
    report["quantize_tensor_int8"] = _time(
        lambda: quantize_tensor(layer, profile8.default_config), 20
    )
    report["compute_storage_int4"] = _time(lambda: compute_storage(arch, profile4), 50)
    report["quantize_model_int4"] = _time(lambda: quantize_model(arch, profile4, model), 5)

    def _sensitivity() -> None:
        analyze_sensitivity(
            arch,
            model,
            sequences,
            methods=(SensitivityMethod.weight_error, SensitivityMethod.activation_weighted),
            bits=4,
            group_size=16,
            max_batches=2,
        )

    report["sensitivity"] = _time(_sensitivity, 5)

    sensitivity = analyze_sensitivity(
        arch,
        model,
        sequences,
        methods=(SensitivityMethod.weight_error, SensitivityMethod.activation_weighted),
        bits=4,
        group_size=16,
        max_batches=2,
    )

    def _optimize() -> None:
        optimize(
            arch,
            sensitivity,
            OptimizationConstraints(quality_tolerance=0.5, allowed_bits=(4, 8, 16)),
            baseline_metric_value=100.0,
            group_size=16,
        )

    report["optimize"] = _time(_optimize, 5)

    cost = AnalyticCostModel()
    options = [
        ProfileOption(
            profile_id=uniform_profile(arch, bits, group_size=16).profile_id,
            precision_bits=bits,
            estimated_cost=cost.estimate(arch, uniform_profile(arch, bits, group_size=16)).lower_cost,
            weight_bytes=compute_storage(arch, uniform_profile(arch, bits, group_size=16)).total_bytes,
            quality_degradation={4: 0.05, 8: 0.01, 16: 0.0}[bits],
        )
        for bits in (4, 8, 16)
    ]

    controller = Controller(ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0), options, options[-1].profile_id)

    from adaptive.models.controller import TelemetrySnapshot

    snapshot = TelemetrySnapshot(memory_pressure=0.97, ttft_ms=120.0, itl_ms=20.0)
    report["controller_decision"] = _time(lambda: controller.recommend(snapshot, now=0.0), 200)

    def _logits(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(batch)

    def _ppl() -> None:
        perplexity(_logits, evaluation, max_batches=4)

    report["perplexity_eval"] = _time(_ppl, 5)

    report["storage"] = {
        "int4_mb": round(compute_storage(arch, profile4).total_mb, 4),
        "int8_mb": round(compute_storage(arch, profile8).total_mb, 4),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
