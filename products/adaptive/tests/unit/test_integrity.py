"""M1: model revision and artifact integrity."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive.errors import IntegrityError
from adaptive.models.integrity import ArtifactManifest, sha256_bytes, sha256_file


def test_manifest_verify_ok(tmp_path: Path) -> None:
    artifact = tmp_path / "profile.json"
    artifact.write_text('{"profile": "uniform-4"}')
    manifest = ArtifactManifest.from_file(artifact)
    manifest.verify(tmp_path)
    assert manifest.size_bytes == artifact.stat().st_size
    assert len(manifest.sha256) == 64


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "profile.json"
    artifact.write_text('{"profile": "uniform-4"}')
    manifest = ArtifactManifest.from_file(artifact)
    artifact.write_text('{"profile": "uniform-8"}')
    with pytest.raises(IntegrityError):
        manifest.verify(tmp_path)


def test_manifest_detects_missing_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "profile.json"
    artifact.write_text("x")
    manifest = ArtifactManifest.from_file(artifact)
    artifact.unlink()
    with pytest.raises(IntegrityError):
        manifest.verify(tmp_path)


def test_manifest_detects_size_change(tmp_path: Path) -> None:
    artifact = tmp_path / "profile.json"
    artifact.write_text("abcd")
    manifest = ArtifactManifest.from_file(artifact)
    artifact.write_text("abcdefgh")
    with pytest.raises(IntegrityError):
        manifest.verify(tmp_path)


def test_sha256_helpers_consistent(tmp_path: Path) -> None:
    data = b"tesseract-adaptive"
    artifact = tmp_path / "blob.bin"
    artifact.write_bytes(data)
    assert sha256_bytes(data) == sha256_file(artifact)
