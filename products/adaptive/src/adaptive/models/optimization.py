"""Precision optimization domain models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from adaptive.models.cost import InferenceCost
from adaptive.models.enums import EvidenceType, QuantMethod, RuntimeBackend
from adaptive.models.quality import QualityMetric
from adaptive.models.quantization import PrecisionProfile
from adaptive.models.storage import StorageBreakdown


class OptimizationConstraints(BaseModel):
    """Hard constraints the optimizer must satisfy."""

    model_config = {"frozen": True}

    runtime: RuntimeBackend = RuntimeBackend.torch_eager
    method: QuantMethod = QuantMethod.reference
    quality_metric: QualityMetric = QualityMetric.perplexity
    quality_tolerance: float = 0.01
    """Maximum allowed relative degradation of the reference quality metric."""

    p95_latency_ms: float | None = None
    memory_budget_gb: float | None = None
    min_throughput_tps: float = 0.0
    allowed_bits: tuple[int, ...] = (4, 8, 16)
    start_bits: int = 4
    max_beam_width: int = 8


class CandidateEvaluation(BaseModel):
    """A generated profile with its projected cost, size and quality."""

    model_config = {"frozen": True}

    profile: PrecisionProfile
    storage: StorageBreakdown
    cost: InferenceCost
    predicted_quality: float
    predicted_degradation: float
    feasible: bool
    violations: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id


class OptimizationResult(BaseModel):
    """Outcome of a precision optimization run."""

    model_config = {"frozen": True}

    model_id: str
    structural_fingerprint: str
    constraints: OptimizationConstraints
    candidates: tuple[CandidateEvaluation, ...] = Field(default_factory=tuple)
    pareto_frontier: tuple[str, ...] = Field(default_factory=tuple)
    selected_profile_id: str | None = None
    uniform_4_profile_id: str | None = None
    uniform_8_profile_id: str | None = None
    algorithm: str = "greedy"
    evidence: EvidenceType = EvidenceType.estimated
    notes: tuple[str, ...] = Field(default_factory=tuple)

    def candidate(self, profile_id: str) -> CandidateEvaluation:
        for candidate in self.candidates:
            if candidate.profile_id == profile_id:
                return candidate
        raise KeyError(profile_id)

    def selected(self) -> CandidateEvaluation | None:
        if self.selected_profile_id is None:
            return None
        return self.candidate(self.selected_profile_id)
