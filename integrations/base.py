"""Integration protocols and shared result types.

These protocols are the seam between the platform coordinator and each product.
The coordinator depends only on these protocols, so it can be driven by real
HTTP/Kubernetes adapters in production and by in-process fakes in tests — while
still exercising the *same* coordinator code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from tesseract_platform.models import (
    Capacity,
    Evidence,
    PlacementSlot,
    PrecisionProfileRef,
    Reservation,
    WorkloadProfile,
    WorkloadRequest,
)


class IntegrationError(RuntimeError):
    """A product integration failed (remote error, bad payload, ...)."""


class IntegrationUnavailable(IntegrationError):
    """A product is not configured/reachable.

    The coordinator must treat this as a first-class condition: it either applies
    explicit conservative defaults or fails the operation, but never silently
    pretends a measurement exists.
    """


class PlacementResult(BaseModel):
    """Outcome of asking Fusion for a placement + reservation."""

    reservation_id: str
    placement: list[PlacementSlot] = Field(default_factory=list)
    capacity: Capacity = Field(default_factory=Capacity)
    expires_at: Optional[datetime] = None
    evidence: Evidence = Evidence.estimated


class DeploymentHandle(BaseModel):
    """Reference to a deployment owned by Ignite."""

    deployment_id: str
    endpoint: Optional[str] = None
    replicas: int = 1
    namespace: str = "tesseract"
    precision_profile_ref: Optional[str] = None


@runtime_checkable
class ProfileProvider(Protocol):
    """Foundry: workload profiling / measurement."""

    def get_profile(self, request: WorkloadRequest) -> WorkloadProfile:
        ...


@runtime_checkable
class PlacementProvider(Protocol):
    """Fusion: placement + capacity reservation."""

    def reserve(
        self,
        request: WorkloadRequest,
        profile: WorkloadProfile,
        precision: Optional[PrecisionProfileRef] = None,
    ) -> PlacementResult:
        ...

    def release(self, reservation_id: str) -> None:
        ...


@runtime_checkable
class DeploymentProvider(Protocol):
    """Ignite: deployment, wake, readiness and routing."""

    def deploy(
        self,
        request: WorkloadRequest,
        reservation: Reservation,
        precision: Optional[PrecisionProfileRef] = None,
    ) -> DeploymentHandle:
        ...

    def await_ready(self, deployment_id: str, timeout_seconds: float) -> bool:
        ...

    def release(self, deployment_id: str) -> None:
        ...


@runtime_checkable
class AdaptivePrecisionProvider(Protocol):
    """Adaptive (optional): precision profile discovery and planning.

    This is a RESERVED interface. Implementations are optional; the platform
    must function with no provider installed.
    """

    def list_profiles(self, model_id: str) -> list[PrecisionProfileRef]:
        ...

    def evaluate_constraints(self, request: WorkloadRequest) -> list[str]:
        ...

    def recommend_profile(
        self,
        request: WorkloadRequest,
        capacity: Optional[Capacity] = None,
    ) -> Optional[PrecisionProfileRef]:
        ...

    def validate_profile(self, profile: PrecisionProfileRef) -> bool:
        ...

    def plan_transition(
        self,
        current: Optional[PrecisionProfileRef],
        target: PrecisionProfileRef,
    ) -> dict:
        ...

    def plan_rollback(self, transition: dict) -> dict:
        ...


__all__ = [
    "AdaptivePrecisionProvider",
    "DeploymentHandle",
    "DeploymentProvider",
    "IntegrationError",
    "IntegrationUnavailable",
    "PlacementProvider",
    "PlacementResult",
    "ProfileProvider",
]
