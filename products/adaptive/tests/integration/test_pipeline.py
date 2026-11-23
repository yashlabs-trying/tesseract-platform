"""M1 end-to-end integration: analyze -> profile -> storage -> serialize."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive.analyzer import ModelAnalyzer
from adaptive.models.enums import Granularity
from adaptive.models.integrity import ArtifactManifest
from adaptive.quantize import (
    assert_profile_supported,
    baseline_profile,
    compute_storage,
    uniform_profile,
    validate_profile,
)
from adaptive.schemas import (
    architecture_document,
    profile_document,
    storage_document,
    write_document,
)

CONFIGS = ["llama-3.2-1b.json", "qwen2.5-0.5b.json", "qwen3-0.6b.json"]


@pytest.mark.parametrize("config_name", CONFIGS)
def test_full_m1_pipeline(config_name: str, tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "models" / config_name
    analyzer = ModelAnalyzer()
    arch = analyzer.analyze_file(fixture, model_id=config_name, revision="main")

    baseline = baseline_profile(arch)
    base_storage = compute_storage(arch, baseline)
    assert base_storage.total_bytes == arch.baseline_weight_bytes

    uniform4 = uniform_profile(arch, 4, granularity=Granularity.per_group, group_size=128)
    uniform8 = uniform_profile(arch, 8, granularity=Granularity.per_group, group_size=128)
    s4 = compute_storage(arch, uniform4)
    s8 = compute_storage(arch, uniform8)
    assert s4.total_bytes < s8.total_bytes < base_storage.total_bytes

    # reference runtime accepts the reference configs
    assert validate_profile(arch, uniform4) == ()

    out_dir = tmp_path / config_name.replace(".json", "")
    write_document(architecture_document(arch), out_dir / "architecture.json")
    write_document(profile_document(uniform4), out_dir / "profile.json")
    write_document(storage_document(arch, uniform4, s4), out_dir / "storage.json")

    for name in ("architecture.json", "profile.json", "storage.json"):
        manifest = ArtifactManifest.from_file(out_dir / name)
        manifest.verify(out_dir)

    reloaded = json.loads((out_dir / "profile.json").read_text())
    assert reloaded["default_config"]["bits"] == 4
    assert reloaded["runtime"] == "torch_eager"


def test_gpu_profile_declared_but_unvalidated() -> None:
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "models" / "llama-3.2-1b.json"
    arch = ModelAnalyzer().analyze_file(fixture, model_id="m")
    from adaptive.models.enums import QuantMethod, RuntimeBackend

    gpu_profile = uniform_profile(
        arch, 4, runtime=RuntimeBackend.vllm, method=QuantMethod.gptq
    )
    # declared capability, but evidence is estimated (no GPU run on this host)
    assert gpu_profile.evidence.value == "estimated"
    assert_profile_supported(arch, gpu_profile)
