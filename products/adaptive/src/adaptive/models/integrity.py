"""Artifact integrity: content hashing and manifest verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from adaptive.errors import IntegrityError

_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Stream-hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactManifest(BaseModel):
    """Integrity metadata for a serialized artifact."""

    model_config = {"frozen": True}

    path: str
    sha256: str
    size_bytes: int
    schema_version: str = "tesseract.adaptive.artifact.v1"

    @classmethod
    def from_file(cls, path: Path | str) -> ArtifactManifest:
        resolved = Path(path)
        return cls(
            path=resolved.name,
            sha256=sha256_file(resolved),
            size_bytes=resolved.stat().st_size,
        )

    def verify(self, root: Path | str) -> None:
        """Verify the referenced artifact against this manifest.

        Raises:
            IntegrityError: if the file is missing, resized or re-hashed.
        """
        target = Path(root) / self.path
        if not target.exists():
            raise IntegrityError(f"artifact missing: {target}")
        actual_size = target.stat().st_size
        if actual_size != self.size_bytes:
            raise IntegrityError(
                f"size mismatch for {self.path}: expected {self.size_bytes}, got {actual_size}"
            )
        actual_hash = sha256_file(target)
        if actual_hash != self.sha256:
            raise IntegrityError(
                f"hash mismatch for {self.path}: expected {self.sha256}, got {actual_hash}"
            )
