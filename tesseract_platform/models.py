"""Canonical typed models for the Tesseract Platform.

These models are the in-process representation of the JSON contracts under
``contracts/``. Evidence semantics are aligned with Foundry's
``tesseract.performance-profile.v1`` (``synthetic | estimated | measured |
validated``) so that a single vocabulary flows across all products.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Evidence(str, Enum):
    """Provenance/trust level of a measurement or estimate."""

    synthetic = "synthetic"
    estimated = "estimated"
    measured = "measured"
    validated = "validated"

    @property
    def is_real_measurement(self) -> bool:
        return self in (Evidence.measured, Evidence.validated)


class ReservationState(str, Enum):
    PENDING = "PENDING"
    RESERVING = "RESERVING"
    RESERVED = "RESERVED"
    DEPLOYING = "DEPLOYING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    RELEASING = "RELEASING"
    RELEASED = "RELEASED"
    FAILED = "FAILED"


class DeploymentState(str, Enum):
    PENDING = "PENDING"
    DEPLOYING = "DEPLOYING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    RELEASING = "RELEASING"
    RELEASED = "RELEASED"
    FAILED = "FAILED"


class OperationState(str, Enum):
    PENDING = "PENDING"
    PROFILING = "PROFILING"
    RESERVED = "RESERVED"
    DEPLOYING = "DEPLOYING"
    ACTIVE = "ACTIVE"
    RELEASING = "RELEASING"
    RELEASED = "RELEASED"
    FAILED = "FAILED"


class ModelRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    architecture: Optional[str] = None
    parameter_count: Optional[int] = Field(default=None, ge=1)
    precision: str = "bf16"
    weight_memory_gb: Optional[float] = Field(default=None, ge=0)


class WorkloadSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    context_length: Optional[int] = Field(default=None, ge=1)
    input_tokens_mean: Optional[float] = Field(default=None, ge=1)
    output_tokens_mean: Optional[float] = Field(default=None, ge=1)
    arrival_rate_rps: float = Field(default=0.0, ge=0)
    concurrency: int = Field(default=1, ge=1)


class SLOSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    ttft_p95_ms: Optional[float] = Field(default=None, ge=0)
    itl_p95_ms: Optional[float] = Field(default=None, ge=0)
    request_p99_ms: Optional[float] = Field(default=None, ge=0)
    availability: Optional[float] = Field(default=None, ge=0, le=1)


class PlacementPreferences(BaseModel):
    model_config = ConfigDict(frozen=True)

    preferred_gpu_model: Optional[str] = None
    min_gpu_count: int = Field(default=1, ge=1)
    replicas: int = Field(default=1, ge=0)


class WorkloadRequest(BaseModel):
    schema_version: str = "tesseract.platform.workload.v1"
    workload_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=8)
    tenant_id: Optional[str] = None
    model: ModelRef
    workload: WorkloadSpec = Field(default_factory=WorkloadSpec)
    slo: SLOSpec = Field(default_factory=SLOSpec)
    placement: PlacementPreferences = Field(default_factory=PlacementPreferences)
    created_at: datetime = Field(default_factory=utcnow)


class WorkloadProfile(BaseModel):
    """Normalized workload measurement produced by Foundry.

    A profile with ``evidence`` of ``synthetic``/``estimated`` is NOT a real GPU
    measurement and must not be presented as one.
    """

    profile_id: str
    model_id: str
    model_revision: str
    hardware_gpu_model: Optional[str] = None
    hardware_gpu_count: int = Field(default=1, ge=1)
    hardware_vram_gb: Optional[float] = Field(default=None, ge=0)
    engine: Optional[str] = None
    engine_version: Optional[str] = None
    ttft_p50_ms: Optional[float] = Field(default=None, ge=0)
    ttft_p95_ms: Optional[float] = Field(default=None, ge=0)
    ttft_p99_ms: Optional[float] = Field(default=None, ge=0)
    itl_p50_ms: Optional[float] = Field(default=None, ge=0)
    itl_p95_ms: Optional[float] = Field(default=None, ge=0)
    request_latency_p95_ms: Optional[float] = Field(default=None, ge=0)
    completed_requests_per_second: Optional[float] = Field(default=None, ge=0)
    output_tokens_per_second: Optional[float] = Field(default=None, ge=0)
    input_tokens_per_second: Optional[float] = Field(default=None, ge=0)
    error_rate: Optional[float] = Field(default=None, ge=0, le=1)
    concurrency: int = Field(default=1, ge=1)
    evidence: Evidence = Evidence.estimated
    collection_timestamp: Optional[datetime] = None
    source_type: Optional[str] = None


class PlacementSlot(BaseModel):
    gpu_id: str
    node_id: str
    gpu_model: Optional[str] = None
    memory_mb: Optional[int] = Field(default=None, ge=0)
    isolation: Optional[str] = None
    evidence: Evidence = Evidence.estimated


class Capacity(BaseModel):
    gpu_count: int = Field(default=0, ge=0)
    gpu_hours: float = Field(default=0.0, ge=0)
    cost_per_hour: float = Field(default=0.0, ge=0)


class Reservation(BaseModel):
    schema_version: str = "tesseract.platform.reservation.v1"
    reservation_id: str
    operation_id: str
    idempotency_key: str
    workload_id: str
    state: ReservationState = ReservationState.PENDING
    revision: int = Field(default=0, ge=0)
    placement: list[PlacementSlot] = Field(default_factory=list)
    capacity: Capacity = Field(default_factory=Capacity)
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    failure_reason: Optional[str] = None


class ModelDeploymentSpec(BaseModel):
    name: str
    engine: str = "vLLM"
    precision: str = "bf16"
    tensorParallel: int = Field(default=1, ge=1)
    pipelineParallel: int = Field(default=1, ge=1)
    gpuPerWorker: int = Field(default=1, ge=1)
    workers: int = Field(default=1, ge=1)


class DeploymentRecord(BaseModel):
    schema_version: str = "tesseract.platform.deployment.v1"
    deployment_id: str
    operation_id: str
    reservation_id: Optional[str] = None
    workload_id: str
    namespace: str = "tesseract"
    state: DeploymentState = DeploymentState.PENDING
    replicas: int = Field(default=1, ge=0)
    model: ModelDeploymentSpec
    precision_profile_ref: Optional[str] = None
    endpoint: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    failure_reason: Optional[str] = None


class TelemetryEvent(BaseModel):
    schema_version: str = "tesseract.platform.telemetry.v1"
    event_id: str
    correlation_id: str
    workload_id: str
    deployment_id: Optional[str] = None
    reservation_id: Optional[str] = None
    event_type: str
    timestamp: datetime = Field(default_factory=utcnow)
    cold_start_ms: Optional[float] = Field(default=None, ge=0)
    deployment_startup_ms: Optional[float] = Field(default=None, ge=0)
    readiness_ms: Optional[float] = Field(default=None, ge=0)
    queue_wait_ms: Optional[float] = Field(default=None, ge=0)
    ttft_ms: Optional[float] = Field(default=None, ge=0)
    itl_ms: Optional[float] = Field(default=None, ge=0)
    request_latency_ms: Optional[float] = Field(default=None, ge=0)
    tokens_input: Optional[int] = Field(default=None, ge=0)
    tokens_output: Optional[int] = Field(default=None, ge=0)
    success: Optional[bool] = None
    replica_count: Optional[int] = Field(default=None, ge=0)
    evidence: Evidence = Evidence.measured
    source: str = "unknown"

    @field_validator("event_type")
    @classmethod
    def _known_event(cls, value: str) -> str:
        allowed = {
            "cold_start",
            "deployment_startup",
            "readiness",
            "queue_wait",
            "request",
            "scale_event",
            "release",
        }
        if value not in allowed:
            raise ValueError(f"unknown telemetry event_type: {value!r}")
        return value


class LifecycleTransition(BaseModel):
    schema_version: str = "tesseract.platform.lifecycle.v1"
    entity_type: str
    entity_id: str
    operation_id: str
    from_state: Optional[str]
    to_state: str
    reason: str = ""
    actor: str = "coordinator"
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=utcnow)


class RollbackStep(BaseModel):
    component: str
    action: str
    status: str = "pending"
    detail: str = ""


class RollbackPlan(BaseModel):
    schema_version: str = "tesseract.platform.rollback.v1"
    rollback_id: str
    operation_id: str
    target_type: str
    target_id: str
    reason: str = ""
    state: str = "PENDING"
    steps: list[RollbackStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = None


class Operation(BaseModel):
    operation_id: str
    idempotency_key: str
    workload_id: str
    correlation_id: str
    tenant_id: Optional[str] = None
    state: OperationState = OperationState.PENDING
    request: WorkloadRequest
    reservation_id: Optional[str] = None
    deployment_id: Optional[str] = None
    endpoint: Optional[str] = None
    profile: Optional[WorkloadProfile] = None
    precision_profile_ref: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class PrecisionProfileRef(BaseModel):
    """Platform-facing reference to an Adaptive precision profile (optional)."""

    schema_version: str = "tesseract.platform.precision-profile.v1"
    profile_ref: str
    model_id: str
    model_revision: Optional[str] = None
    runtime: str
    method: str
    deployable: bool
    evidence: Evidence
    bits: Optional[int] = None
    weight_memory_gb: Optional[float] = Field(default=None, ge=0)
    quality_certificate_ref: Optional[str] = None
    backend_compatibility: list[str] = Field(default_factory=list)
    transition_overhead_ms: Optional[float] = Field(default=None, ge=0)
    provenance: Optional[str] = None


__all__ = [
    "Capacity",
    "DeploymentRecord",
    "DeploymentState",
    "Evidence",
    "LifecycleTransition",
    "ModelDeploymentSpec",
    "ModelRef",
    "Operation",
    "OperationState",
    "PlacementPreferences",
    "PlacementSlot",
    "PrecisionProfileRef",
    "Reservation",
    "ReservationState",
    "RollbackPlan",
    "RollbackStep",
    "SLOSpec",
    "TelemetryEvent",
    "WorkloadProfile",
    "WorkloadRequest",
    "WorkloadSpec",
    "utcnow",
]
