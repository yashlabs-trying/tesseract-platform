"""Example: analyze a model and compare precision profile storage.

Run from the repository root::

    python examples/demo_analyze.py
"""

from __future__ import annotations

from pathlib import Path

from adaptive.analyzer import ModelAnalyzer
from adaptive.models.enums import Granularity
from adaptive.quantize import baseline_profile, compute_storage, uniform_profile

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "models" / "llama-3.2-1b.json"


def main() -> None:
    arch = ModelAnalyzer().analyze_file(FIXTURE, model_id="meta-llama/Llama-3.2-1B")
    print(f"model:      {arch.model_id} ({arch.family.value})")
    print(f"parameters: {arch.total_parameters:,}")
    print(f"baseline:   {arch.baseline_dtype.value} {arch.baseline_weight_bytes / 1e6:.1f} MB")
    print()

    profiles = [
        baseline_profile(arch),
        uniform_profile(arch, 8, granularity=Granularity.per_group, group_size=128),
        uniform_profile(arch, 4, granularity=Granularity.per_group, group_size=128),
    ]
    print(f"{'profile':<28}{'size (MB)':>12}{'ratio':>10}{'deployable':>12}")
    for profile in profiles:
        storage = compute_storage(arch, profile)
        print(
            f"{profile.name:<28}{storage.total_mb:>12.1f}"
            f"{storage.compression_ratio:>10.2f}{str(profile.deployable):>12}"
        )


if __name__ == "__main__":
    main()
