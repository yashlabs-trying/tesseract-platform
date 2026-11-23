"""Ignite integration adapter (Milestone 4).

Loads prevalidated profiles, plans safe replica transitions, and exposes
readiness and rollback signals. Ignite remains independently usable; this
adapter is a pure contract translation with no network dependency. Transitions
are always replica-based and never in-place during active inference.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from adaptive.models.controller import QualityCertificate, SwitchStep
from adaptive.models.enums import EvidenceType

IGNITE_PROFILE_LOAD_SCHEMA = "tesseract.adaptive.ignite-profile-load.v1"


class IgniteReplicaTransition(BaseModel):
    """A safe, replica-based profile transition plan."""

    model_config = {"frozen": True}

    from_profile_id: str
    to_profile_id: str
    replica_count: int = Field(ge=1)
    steps: tuple[SwitchStep, ...] = (
        SwitchStep.drain,
        SwitchStep.load,
        SwitchStep.warmup,
        SwitchStep.cutover,
        SwitchStep.verify,
    )
    max_unavailable: int = 0
    in_place: bool = False
    readiness_signal: str = "ready"
    rollback_signal: str = "rollback"


class IgniteLoadRequest(BaseModel):
    """Request to load a (prevalidated) profile onto a deployment."""

    model_config = {"frozen": True}

    deployment_id: str
    model_id: str
    profile_id: str
    from_profile_id: str | None = None
    replica_count: int = Field(default=1, ge=1)
    certificate: QualityCertificate | None = None
    require_validation: bool = True


class IgniteLoadResponse(BaseModel):
    """Ignite's response to a profile load request."""

    model_config = {"frozen": True}

    deployment_id: str
    profile_id: str
    accepted: bool
    reason: str
    transition: IgniteReplicaTransition | None = None
    readiness_signal: str = "not-ready"
    rollback_signal: str = "rollback"
    evidence: EvidenceType = EvidenceType.estimated


class IgniteProvider(Protocol):
    """Protocol implemented by an Ignite client."""

    def load_profile(self, request: IgniteLoadRequest) -> IgniteLoadResponse: ...


def plan_replica_transition(
    from_profile_id: str,
    to_profile_id: str,
    replica_count: int,
    *,
    max_unavailable: int = 0,
) -> IgniteReplicaTransition:
    """Plan a rolling, replica-based transition (never in-place)."""
    return IgniteReplicaTransition(
        from_profile_id=from_profile_id,
        to_profile_id=to_profile_id,
        replica_count=replica_count,
        max_unavailable=max_unavailable,
        in_place=False,
    )


def load_document(response: IgniteLoadResponse) -> dict[str, object]:
    """Serialize an Ignite load response with a versioned schema id."""
    return {"schema_version": IGNITE_PROFILE_LOAD_SCHEMA, **response.model_dump(mode="json")}


class InProcessIgnite:
    """Default Ignite provider holding a set of prevalidated profiles."""

    def __init__(self, validated_profiles: set[str] | None = None) -> None:
        self.validated_profiles = set(validated_profiles or set())

    def register_validated(self, profile_id: str) -> None:
        self.validated_profiles.add(profile_id)

    def is_validated(self, request: IgniteLoadRequest) -> bool:
        if request.profile_id in self.validated_profiles:
            return True
        certificate = request.certificate
        return bool(
            certificate is not None
            and certificate.validated
            and certificate.profile_id == request.profile_id
        )

    def load_profile(self, request: IgniteLoadRequest) -> IgniteLoadResponse:
        if request.require_validation and not self.is_validated(request):
            return IgniteLoadResponse(
                deployment_id=request.deployment_id,
                profile_id=request.profile_id,
                accepted=False,
                reason="profile is not prevalidated; refusing to load",
                readiness_signal="not-ready",
            )
        from_profile = request.from_profile_id or f"{request.profile_id}-previous"
        transition = plan_replica_transition(
            from_profile, request.profile_id, request.replica_count
        )
        return IgniteLoadResponse(
            deployment_id=request.deployment_id,
            profile_id=request.profile_id,
            accepted=True,
            reason="profile prevalidated; rolling transition planned",
            transition=transition,
            readiness_signal="ready",
            rollback_signal="rollback",
        )


__all__ = [
    "IGNITE_PROFILE_LOAD_SCHEMA",
    "IgniteLoadRequest",
    "IgniteLoadResponse",
    "IgniteProvider",
    "IgniteReplicaTransition",
    "InProcessIgnite",
    "load_document",
    "plan_replica_transition",
]
