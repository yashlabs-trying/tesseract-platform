"""Example: sensitivity -> quality -> optimization on a synthetic reference model.

Run from the repository root::

    python examples/demo_intelligence.py
"""

from __future__ import annotations

from pathlib import Path

from adaptive.analyzer import ModelAnalyzer
from adaptive.models.optimization import OptimizationConstraints
from adaptive.models.quality import QualityMetric
from adaptive.nets import ReferenceTransformer
from adaptive.optimizer import optimize
from adaptive.quality import (
    DatasetSplit,
    make_synthetic_dataset,
    run_quality_suite,
)
from adaptive.quantize import uniform_profile
from adaptive.sensitivity import SensitivityMethod, analyze_sensitivity

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "models" / "tiny-llama.json"


def main() -> None:
    arch = ModelAnalyzer().analyze_file(FIXTURE, model_id="tiny-llama")
    model = ReferenceTransformer.from_architecture(arch, seed=0)
    calibration = make_synthetic_dataset(
        "calibration", DatasetSplit.calibration, vocab_size=arch.vocab_size, seed=1
    )
    evaluation = make_synthetic_dataset(
        "evaluation", DatasetSplit.evaluation, vocab_size=arch.vocab_size, seed=2
    )

    report = analyze_sensitivity(
        arch,
        model,
        calibration.sequences,
        methods=(
            SensitivityMethod.weight_error,
            SensitivityMethod.activation_weighted,
            SensitivityMethod.output_divergence,
        ),
        bits=4,
        group_size=16,
        calibration_fingerprint=calibration.fingerprint(),
    )
    print("most sensitive layers (activation_weighted):")
    for rank, name in enumerate(report.top_k(SensitivityMethod.activation_weighted, 3), 1):
        print(f"  {rank}. {name}")

    print("\nquality (uniform 4-bit vs baseline):")
    for bits in (16, 8, 4):
        quality = run_quality_suite(
            arch,
            model,
            uniform_profile(arch, bits, group_size=16),
            calibration=calibration,
            evaluation=evaluation,
            suite=(QualityMetric.perplexity, QualityMetric.output_divergence),
        )
        ppl = quality.metric(QualityMetric.perplexity)
        kl = quality.metric(QualityMetric.output_divergence)
        print(f"  {bits}-bit: ppl={ppl.value:.3f} (base {ppl.baseline_value:.3f}) kl={kl.value:.5f}")

    print("\noptimizer:")
    result = optimize(
        arch,
        report,
        OptimizationConstraints(quality_tolerance=0.05, allowed_bits=(4, 8, 16)),
        baseline_metric_value=1.0,
        group_size=16,
    )
    print(f"  candidates={len(result.candidates)} pareto={len(result.pareto_frontier)}")
    selected = result.selected()
    if selected is not None:
        print(f"  selected={selected.profile.name} bits={sorted(selected.profile.distinct_bits())}")
        print(f"  size={selected.storage.total_mb:.4f} MB cost={selected.cost.lower_cost:.6f}")


if __name__ == "__main__":
    main()
