"""Phase 8 - adversarial security and reliability tests (safe fixtures only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive.analyzer import ModelAnalyzer
from adaptive.benchmark import build_benchmark_report, load_manifest
from adaptive.errors import ConfigurationError, IntegrityError, UnsupportedModelError
from adaptive.integration import import_performance_profile
from adaptive.models.integrity import ArtifactManifest
from adaptive.schemas import write_document


def test_malformed_model_config_rejected() -> None:
    analyzer = ModelAnalyzer()
    with pytest.raises(UnsupportedModelError):
        analyzer.analyze({"model_type": "unknown-arch"}, model_id="x")
    with pytest.raises(ConfigurationError):
        analyzer.analyze({"model_type": "llama", "hidden_size": 8}, model_id="x")


def test_invalid_tensor_shape_rejected() -> None:
    torch = pytest.importorskip("torch")
    from adaptive.models.enums import Granularity
    from adaptive.models.quantization import int_config
    from adaptive.quantize import quantize_tensor

    config = int_config(8, granularity=Granularity.per_channel)
    with pytest.raises(ConfigurationError):
        quantize_tensor(torch.randn(16), config)  # 1D weight cannot be per-channel


def test_corrupted_artifact_detected(tmp_path: Path) -> None:
    artifact = tmp_path / "profile.json"
    artifact.write_text('{"a": 1}')
    manifest = ArtifactManifest.from_file(artifact)
    artifact.write_text('{"a": 2}')
    with pytest.raises(IntegrityError):
        manifest.verify(tmp_path)


def test_write_document_is_atomic_and_valid(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "doc.json"
    write_document({"k": "v"}, target)
    assert json.loads(target.read_text()) == {"k": "v"}
    assert list(target.parent.glob("*.tmp")) == []


def test_write_document_failure_leaves_no_partial_target(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    target = blocker / "doc.json"  # parent is a file, cannot create
    with pytest.raises((FileNotFoundError, NotADirectoryError, OSError)):
        write_document({"k": "v"}, target)
    assert not target.exists()


def test_manifest_path_traversal_is_not_followed(tmp_path: Path) -> None:
    # a manifest path escaping the directory simply must not exist -> rejected
    with pytest.raises(ConfigurationError):
        load_manifest(tmp_path / ".." / ".." / "etc" / "passwd")


def test_manifest_wrong_types_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: tesseract.adaptive.gpu-test-manifest.v1\n"
        "name: x\nmodel_id: m\n"
        "profile: {bits: 3, method: gptq}\n"
        "hardware: {gpu_model: g, vram_gb: -1}\n"
        "workload: {}\nthresholds: {ttft_p95_ms: 1, itl_p95_ms: 1}\n"
        "calibration_dataset: c\nevaluation_dataset: e\n"
    )
    with pytest.raises((ValidationError, ConfigurationError)):
        load_manifest(bad)


def test_partial_benchmark_report_rejected() -> None:
    with pytest.raises(ConfigurationError):
        build_benchmark_report([])


def test_provenance_cannot_be_forged_to_measured() -> None:
    document = {
        "schema_version": "tesseract.performance-profile.v1",
        "profile_id": "p",
        "model_id": "m",
        "evidence_type": "estimated",
        "ttft_p95_ms": 10.0,
        "throughput_tps": 100.0,
    }
    measurement = import_performance_profile(document)
    # an estimated measurement stays estimated and is rejected by default
    assert measurement.evidence.value == "estimated"
    with pytest.raises(ConfigurationError):
        build_benchmark_report([measurement])


def test_unknown_evidence_type_rejected() -> None:
    with pytest.raises(ConfigurationError):
        import_performance_profile(
            {
                "schema_version": "tesseract.performance-profile.v1",
                "profile_id": "p",
                "model_id": "m",
                "evidence_type": "definitely-real",
            }
        )


def test_backend_dependency_detection_is_safe() -> None:
    from adaptive.backends import package_available

    assert package_available("definitely_not_a_real_package_xyz") is False
    # find_spec on a missing parent must not raise
    assert package_available("definitely.not.a.package") is False


def test_oversized_metadata_does_not_crash_loader(tmp_path: Path) -> None:
    # a very large but valid field is accepted structurally; it is just data
    doc = {
        "schema_version": "tesseract.performance-profile.v1",
        "profile_id": "p" * 10000,
        "model_id": "m",
        "evidence_type": "measured",
    }
    measurement = import_performance_profile(doc)
    assert len(measurement.profile_id) == 10000
