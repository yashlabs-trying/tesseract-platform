"""Serialized storage accounting domain model."""

from __future__ import annotations

from pydantic import BaseModel


class StorageBreakdown(BaseModel):
    """Byte-level breakdown of a serialized precision profile.

    Separating payload from overhead is essential: scales, zero points and
    metadata can dominate the apparent savings at 4-bit for small group sizes.
    """

    model_config = {"frozen": True}

    parameter_bytes: int = 0
    """Quantized weight payload (packed bits)."""

    scale_bytes: int = 0
    zero_point_bytes: int = 0
    metadata_bytes: int = 0
    unquantized_bytes: int = 0
    """Weights kept at baseline precision (norms, excluded embeddings, biases)."""

    total_bytes: int = 0
    baseline_bytes: int = 0
    """Bytes required to store the same model at its baseline dtype."""

    @property
    def overhead_bytes(self) -> int:
        return self.scale_bytes + self.zero_point_bytes + self.metadata_bytes

    @property
    def compression_ratio(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.baseline_bytes / self.total_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1_000_000

    @property
    def baseline_mb(self) -> float:
        return self.baseline_bytes / 1_000_000

    def __add__(self, other: StorageBreakdown) -> StorageBreakdown:
        return StorageBreakdown(
            parameter_bytes=self.parameter_bytes + other.parameter_bytes,
            scale_bytes=self.scale_bytes + other.scale_bytes,
            zero_point_bytes=self.zero_point_bytes + other.zero_point_bytes,
            metadata_bytes=self.metadata_bytes + other.metadata_bytes,
            unquantized_bytes=self.unquantized_bytes + other.unquantized_bytes,
            total_bytes=self.total_bytes + other.total_bytes,
            baseline_bytes=self.baseline_bytes + other.baseline_bytes,
        )
