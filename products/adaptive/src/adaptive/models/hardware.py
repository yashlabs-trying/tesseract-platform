"""Hardware description domain model."""

from __future__ import annotations

from pydantic import BaseModel, Field

from adaptive.models.enums import RuntimeBackend


class HardwareSpec(BaseModel):
    """A target device and the runtimes/bit widths it can execute."""

    model_config = {"frozen": True}

    hardware_id: str
    vendor: str
    device: str
    gpu_model: str = ""
    gpu_count: int = 1
    vram_gb: float = 0.0
    supported_backends: tuple[RuntimeBackend, ...] = Field(default_factory=tuple)
    supported_bits: tuple[int, ...] = Field(default_factory=tuple)
    notes: tuple[str, ...] = Field(default_factory=tuple)

    def supports(self, backend: RuntimeBackend, bits: int) -> bool:
        return backend in self.supported_backends and bits in self.supported_bits
