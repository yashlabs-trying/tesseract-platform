"""Inference cost domain model."""

from __future__ import annotations

from pydantic import BaseModel

from adaptive.models.enums import EvidenceType


class InferenceCost(BaseModel):
    """Estimated or measured serving cost of a precision profile.

    ``lower_cost`` is the scalar objective used by the optimizer; it is derived
    from the components and is expressed in arbitrary cost units.
    """

    model_config = {"frozen": True}

    weight_bytes: int = 0
    bytes_per_token: int = 0
    estimated_latency_ms: float = 0.0
    estimated_throughput_tps: float = 0.0
    estimated_cost_per_million_tokens: float = 0.0
    lower_cost: float = 0.0
    evidence: EvidenceType = EvidenceType.estimated
