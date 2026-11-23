"""Inference cost estimation (Milestone 2).

The analytic cost model is an *estimate* derived from weight bytes and memory
bandwidth. It is explicitly labelled ``estimated`` and is replaced by measured
costs (Foundry import) once real benchmarks exist. Lower precision is never
assumed to be faster; the model derives latency from bytes moved.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive.models.architecture import ModelArchitecture
from adaptive.models.cost import InferenceCost
from adaptive.models.enums import EvidenceType
from adaptive.models.quantization import PrecisionProfile
from adaptive.quantize.storage import compute_storage


@dataclass(frozen=True)
class HardwareProfile:
    """Parameters of the analytic cost model (not measured device telemetry)."""

    hardware_id: str = "analytic-default"
    memory_bandwidth_gbps: float = 2000.0
    peak_tflops: float = 100.0
    price_per_hour: float = 2.0
    batch_size: int = 1
    dtype_bytes: float = 2.0


class AnalyticCostModel:
    """Estimate serving cost from serialized size and memory bandwidth."""

    def __init__(self, hardware: HardwareProfile | None = None) -> None:
        self.hardware = hardware or HardwareProfile()

    def estimate(
        self, architecture: ModelArchitecture, profile: PrecisionProfile
    ) -> InferenceCost:
        storage = compute_storage(architecture, profile)
        activation_bytes = int(architecture.hidden_size * self.hardware.dtype_bytes * 4)
        bytes_per_token = storage.total_bytes + activation_bytes
        bandwidth_bytes_per_s = max(self.hardware.memory_bandwidth_gbps * 1e9, 1.0)
        latency_s = bytes_per_token * self.hardware.batch_size / bandwidth_bytes_per_s
        latency_ms = latency_s * 1000.0
        throughput = self.hardware.batch_size / latency_s if latency_s > 0 else 0.0
        cost_per_1m = (
            latency_s * (self.hardware.price_per_hour / 3600.0) / self.hardware.batch_size * 1e6
        )
        return InferenceCost(
            weight_bytes=storage.total_bytes,
            bytes_per_token=bytes_per_token,
            estimated_latency_ms=latency_ms,
            estimated_throughput_tps=throughput,
            estimated_cost_per_million_tokens=cost_per_1m,
            lower_cost=cost_per_1m,
            evidence=EvidenceType.estimated,
        )
