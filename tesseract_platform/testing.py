"""In-process test doubles for the platform integration protocols.

These fakes let the test suite exercise the *real* coordinator, state machine,
store, telemetry and reconciler without external services. They are shipped with
the package so downstream users can also test integrations deterministically.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from integrations.base import (
    DeploymentHandle,
    IntegrationError,
    PlacementResult,
)
from tesseract_platform.models import (
    Capacity,
    Evidence,
    PlacementSlot,
    WorkloadProfile,
    WorkloadRequest,
    utcnow,
)


def make_request(
    *,
    workload_id: str = "wl-1",
    idempotency_key: str = "idem-key-0001",
    model_id: str = "qwen3-0.6b",
    revision: str = "main",
    concurrency: int = 1,
    replicas: int = 1,
    min_gpu_count: int = 1,
    ttft_p95_ms: Optional[float] = None,
) -> WorkloadRequest:
    from tesseract_platform.models import ModelRef, PlacementPreferences, SLOSpec, WorkloadSpec

    return WorkloadRequest(
        workload_id=workload_id,
        idempotency_key=idempotency_key,
        model=ModelRef(model_id=model_id, revision=revision),
        workload=WorkloadSpec(concurrency=concurrency),
        slo=SLOSpec(ttft_p95_ms=ttft_p95_ms),
        placement=PlacementPreferences(replicas=replicas, min_gpu_count=min_gpu_count),
    )


class FakeProfileProvider:
    def __init__(
        self,
        *,
        evidence: Evidence = Evidence.measured,
        ttft_p95_ms: float = 120.0,
        itl_p95_ms: float = 20.0,
        output_tokens_per_second: float = 900.0,
        fail: bool = False,
    ) -> None:
        self.evidence = evidence
        self.ttft_p95_ms = ttft_p95_ms
        self.itl_p95_ms = itl_p95_ms
        self.output_tokens_per_second = output_tokens_per_second
        self.fail = fail
        self.calls = 0

    def get_profile(self, request: WorkloadRequest) -> WorkloadProfile:
        self.calls += 1
        if self.fail:
            raise IntegrationError("foundry unavailable")
        return WorkloadProfile(
            profile_id=f"profile-{request.model.model_id}",
            model_id=request.model.model_id,
            model_revision=request.model.revision,
            hardware_gpu_model="simulated-gpu",
            engine="vLLM",
            ttft_p95_ms=self.ttft_p95_ms,
            itl_p95_ms=self.itl_p95_ms,
            output_tokens_per_second=self.output_tokens_per_second,
            concurrency=request.workload.concurrency,
            evidence=self.evidence,
        )


class FakePlacementProvider:
    def __init__(
        self,
        *,
        gpu_count: int = 1,
        gpu_model: str = "simulated-gpu",
        fail: bool = False,
        ttl_seconds: int = 900,
    ) -> None:
        self.gpu_count = gpu_count
        self.gpu_model = gpu_model
        self.fail = fail
        self.ttl_seconds = ttl_seconds
        self.reserved: list[str] = []
        self.released: list[str] = []

    def reserve(self, request, profile, precision=None) -> PlacementResult:
        if self.fail:
            raise IntegrationError("fusion cannot place workload (insufficient capacity)")
        reservation_id = f"res-{request.workload_id}"
        self.reserved.append(reservation_id)
        placement = [
            PlacementSlot(
                gpu_id=f"gpu-{i}",
                node_id=f"node-{i // 4}",
                gpu_model=self.gpu_model,
                memory_mb=24576,
                isolation="exclusive",
                evidence=Evidence.estimated,
            )
            for i in range(self.gpu_count)
        ]
        return PlacementResult(
            reservation_id=reservation_id,
            placement=placement,
            capacity=Capacity(gpu_count=self.gpu_count, cost_per_hour=2.5 * self.gpu_count),
            expires_at=utcnow() + timedelta(seconds=self.ttl_seconds),
            evidence=Evidence.estimated,
        )

    def release(self, reservation_id: str) -> None:
        self.released.append(reservation_id)


class FakeDeploymentProvider:
    def __init__(self, *, ready: bool = True, fail: bool = False) -> None:
        self.ready = ready
        self.fail = fail
        self.deployed: list[str] = []
        self.released: list[str] = []

    def deploy(self, request, reservation, precision=None) -> DeploymentHandle:
        if self.fail:
            raise IntegrationError("ignite deployment create failed")
        deployment_id = f"wl-{request.workload_id}"
        self.deployed.append(deployment_id)
        return DeploymentHandle(
            deployment_id=deployment_id,
            endpoint=f"/v1/workloads/{request.workload_id}",
            replicas=request.placement.replicas or 1,
            precision_profile_ref=precision.profile_ref if precision else None,
        )

    def await_ready(self, deployment_id: str, timeout_seconds: float) -> bool:
        return self.ready

    def release(self, deployment_id: str) -> None:
        self.released.append(deployment_id)
