"""M5: benchmark report generation and GPU test manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive.benchmark import (
    build_benchmark_report,
    load_manifest,
    load_manifests,
    write_benchmark_report,
)
from adaptive.errors import ConfigurationError
from adaptive.integration import import_performance_profile
from adaptive.models.enums import EvidenceType

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT = ROOT / "fixtures" / "foundry" / "performance-profile-example.json"
MANIFEST_DIR = ROOT / "gpu" / "manifests"


def _measurement():
    return import_performance_profile(json.loads(MEASUREMENT.read_text()))


def test_build_report_from_measured() -> None:
    report = build_benchmark_report([_measurement()])
    assert report.model_id == "Qwen/Qwen2.5-0.5B"
    assert report.evidence is EvidenceType.measured
    assert len(report.summaries) == 1
    summary = report.summaries[0]
    assert summary.ttft_p95_ms == 71.0
    assert summary.throughput_tps == 1850.0


def test_report_rejects_unmeasured_data() -> None:
    estimated = _measurement().model_copy(update={"evidence": EvidenceType.estimated})
    with pytest.raises(ConfigurationError):
        build_benchmark_report([estimated])


def test_report_allows_unmeasured_when_requested() -> None:
    estimated = _measurement().model_copy(update={"evidence": EvidenceType.estimated})
    report = build_benchmark_report([estimated], require_measured=False)
    assert report.evidence is EvidenceType.measured
    assert report.summaries[0].evidence is EvidenceType.estimated


def test_report_rejects_empty() -> None:
    with pytest.raises(ConfigurationError):
        build_benchmark_report([])


def test_report_markdown() -> None:
    markdown = build_benchmark_report([_measurement()]).to_markdown()
    assert "# Benchmark Report" in markdown
    assert "foundry-profile-qwen2-0.5b-int4" in markdown


def test_write_report_atomic(tmp_path: Path) -> None:
    report = build_benchmark_report([_measurement()])
    target = tmp_path / "reports" / "bench.json"
    write_benchmark_report(report, target)
    assert target.exists()
    assert not target.with_suffix(".json.tmp").exists()
    data = json.loads(target.read_text())
    assert data["schema_version"] == "tesseract.adaptive.benchmark-report.v1"


def test_load_manifest() -> None:
    manifest = load_manifest(MANIFEST_DIR / "qwen2.5-0.5b-int4-gptq.yaml")
    assert manifest.model_id == "Qwen/Qwen2.5-0.5B"
    assert manifest.profile.bits == 4
    assert manifest.profile.method == "gptq"
    assert manifest.thresholds.ttft_p95_ms == 120.0


def test_load_manifests_directory() -> None:
    manifests = load_manifests(MANIFEST_DIR)
    assert len(manifests) >= 3
    assert all(manifest.schema_version.endswith("gpu-test-manifest.v1") for manifest in manifests)


def test_load_manifest_missing() -> None:
    with pytest.raises(ConfigurationError):
        load_manifest(MANIFEST_DIR / "does-not-exist.yaml")


def test_load_manifest_rejects_bad_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: nope\nname: x\n")
    with pytest.raises(ConfigurationError):
        load_manifest(bad)
