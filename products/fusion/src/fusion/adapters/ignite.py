from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime
from pydantic import BaseModel, Field


class IgniteCapacityContract(BaseModel):
    deployment_id: str
    model_id: str
    workload_forecast_rps: float = 0.0
    cold_start_distribution_ms: Optional[float] = None
    required_slo_p99_ms: Optional[float] = None
    cost_of_warm_workers_per_hour: float = 0.0
    cost_of_reactivation: float = 0.0
    available_capacity_slots: int = 0


class IgniteDecision(BaseModel):
    recommended_warm_replica_floor: int = 0
    recommended_idle_policy: str = "warm"
    estimated_cost_per_hour: float = 0.0
    estimated_cold_start_impact_ms: Optional[float] = None
    assumptions: List[str] = Field(default_factory=list)
    confidence: float = 0.5


class IgniteProvider(Protocol):
    def recommend_capacity(self, contract: IgniteCapacityContract) -> IgniteDecision:
        ...
