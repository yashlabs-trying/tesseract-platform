"""Adaptive runtime controller domain models (Milestone 3)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from adaptive.models.enums import EvidenceType


class ControllerAction(str, Enum):
    hold = "hold"
    downshift = "downshift"
    upshift = "upshift"
    rollback = "rollback"


class SwitchStep(str, Enum):
    drain = "drain"
    load = "load"
    warmup = "warmup"
    cutover = "cutover"
    verify = "verify"


class TelemetrySnapshot(BaseModel):
    """A point-in-time view of serving pressure and performance."""

    model_config = {"frozen": True}

    timestamp_s: float = 0.0
    memory_pressure: float = Field(ge=0.0, le=1.0)
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    queue_depth: int = Field(default=0, ge=0)
    ttft_ms: float = Field(default=0.0, ge=0.0)
    itl_ms: float = Field(default=0.0, ge=0.0)
    tokens_per_second: float = Field(default=0.0, ge=0.0)
    model_demand: float = Field(default=0.0, ge=0.0)
    evidence: EvidenceType = EvidenceType.synthetic


class QualityCertificate(BaseModel):
    """A measured quality certificate authorising a profile for deployment."""

    model_config = {"frozen": True}

    profile_id: str
    metric: str
    degradation: float
    validated: bool = False
    evidence: EvidenceType = EvidenceType.measured


class ProfileOption(BaseModel):
    """A candidate precision profile as seen by the controller."""

    model_config = {"frozen": True}

    profile_id: str
    precision_bits: int
    estimated_cost: float
    weight_bytes: int
    quality_degradation: float = 0.0


class ControllerPolicy(BaseModel):
    """Tunable controller policy (anti-oscillation and SLO thresholds)."""

    model_config = {"frozen": True}

    memory_high: float = 0.90
    memory_low: float = 0.60
    hysteresis: float = 0.05
    ttft_slo_ms: float = 500.0
    itl_slo_ms: float = 50.0
    min_throughput_tps: float = 0.0
    quality_tolerance: float = 0.05
    min_dwell_s: float = 300.0
    cooldown_s: float = 120.0
    rate_limit_per_hour: int = 4
    rollback_window_s: float = 600.0
    recommendation_only: bool = True


class SwitchPlan(BaseModel):
    """A safe, replica-based profile switch plan."""

    model_config = {"frozen": True}

    from_profile_id: str
    to_profile_id: str
    steps: tuple[SwitchStep, ...] = (
        SwitchStep.drain,
        SwitchStep.load,
        SwitchStep.warmup,
        SwitchStep.cutover,
        SwitchStep.verify,
    )
    estimated_downtime_s: float = 0.0
    requires_replica_transition: bool = True
    in_place: bool = False
    reason: str = ""


class RollbackPlan(BaseModel):
    """A plan to revert to the previous known-good profile."""

    model_config = {"frozen": True}

    from_profile_id: str
    to_profile_id: str
    triggers: tuple[str, ...] = Field(default_factory=tuple)


class ControlDecision(BaseModel):
    """The controller's recommendation for one telemetry snapshot."""

    model_config = {"frozen": True}

    action: ControllerAction
    current_profile_id: str
    recommended_profile_id: str
    reason: str
    automatic: bool = False
    rate_limited: bool = False
    dwell_remaining_s: float = 0.0
    expected_memory_gb: float = 0.0
    expected_cost: float = 0.0
    expected_quality_degradation: float = 0.0
    switch_plan: SwitchPlan | None = None
    rollback_plan: RollbackPlan | None = None
    evidence: EvidenceType = EvidenceType.synthetic


class SimulationResult(BaseModel):
    """Outcome of a controller simulation over a telemetry timeline."""

    model_config = {"frozen": True}

    decisions: tuple[ControlDecision, ...] = Field(default_factory=tuple)
    switch_count: int = 0
    rollback_count: int = 0
    direction_changes: int = 0
    oscillated: bool = False
    final_profile_id: str = ""
    evidence: EvidenceType = EvidenceType.synthetic
