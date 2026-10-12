from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field, model_validator
import uuid


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    INR = "INR"


class IsolationMode(str, Enum):
    EXCLUSIVE = "exclusive"
    TIME_SLICED = "time_sliced"
    MIG = "mig"
    VGPU = "vgpu"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class SharingPolicy(str, Enum):
    EXCLUSIVE_ONLY = "exclusive_only"
    TRUSTED_TENANTS = "trusted_tenants_only"
    ANY_SHARED = "any_shared"


class PreemptionPolicy(str, Enum):
    NEVER = "never"
    LOWER_PRIORITY = "lower_priority_only"
    ALWAYS = "always"


class BillingModel(str, Enum):
    PER_TOKEN = "per_token"
    PER_REQUEST = "per_request"
    PER_HOUR_RESERVED = "per_hour_reserved"
    PER_HOUR_ON_DEMAND = "per_hour_on_demand"
    COMMITTED = "committed"
    HYBRID = "hybrid"


class EvidenceType(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    SIMULATED = "simulated"
    EXTRAPOLATED = "extrapolated"
    UNKNOWN = "unknown"


class AllocationStatus(str, Enum):
    CURRENT = "current"
    CANDIDATE = "candidate"
    REJECTED = "rejected"
    RECOMMENDED = "recommended"
    APPROVED = "approved"
    EXECUTED = "executed"
    FAILED = "failed"


class RecommendationStatus(str, Enum):
    ADVISORY_ONLY = "advisory_only"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    ROLLED_BACK = "rolled_back"
    STALE = "stale"


class ConstraintResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class GPUInventory(BaseModel):
    model_config = {"frozen": True}
    gpu_id: str = Field(description="Unique GPU identifier")
    cluster_id: str = "default"
    node_id: str
    vendor: str = "nvidia"
    gpu_model: str
    memory_capacity_mb: int = Field(ge=0)
    available_memory_mb: Optional[int] = Field(default=None, ge=0)
    available_resource_slots: int = Field(default=1, ge=0)
    interconnect: Optional[str] = None
    topology: Optional[str] = None
    supported_isolation_modes: List[IsolationMode] = [IsolationMode.EXCLUSIVE]
    health_status: HealthStatus = HealthStatus.UNKNOWN
    sharing_policy: SharingPolicy = SharingPolicy.EXCLUSIVE_ONLY
    location_zone: Optional[str] = None
    resource_cost_per_hour: Decimal = Field(default=Decimal("0"), ge=0)
    currency: Currency = Currency.USD
    resource_cost_source: str = "manual"
    observed_at: datetime = Field(default_factory=datetime.utcnow)
    refresh_timestamp: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def memory_consistency(self):
        if self.available_memory_mb is not None and self.available_memory_mb > self.memory_capacity_mb:
            raise ValueError("available_memory_mb cannot exceed memory_capacity_mb")
        return self


class TenantPolicy(BaseModel):
    model_config = {"frozen": True}
    tenant_id: str
    allowed_gpu_classes: List[str] = Field(default_factory=lambda: ["*"])
    resource_quotas: Dict[str, int] = Field(default_factory=dict)
    priority: int = 0
    isolation_requirements: IsolationMode = IsolationMode.EXCLUSIVE
    data_residency_zones: Optional[List[str]] = None
    latency_p95_ms: Optional[int] = None
    latency_p99_ms: Optional[int] = None
    availability_objective: float = 0.99
    preemption_policy: PreemptionPolicy = PreemptionPolicy.NEVER
    allowed_scheduling_flexibility: bool = False
    commercial_contract_ref: Optional[str] = None
    max_concurrent_workers: Optional[int] = None


class ModelDeployment(BaseModel):
    model_config = {"frozen": True}
    deployment_id: str
    tenant_id: str
    model_id: str
    model_revision: str = "latest"
    serving_engine: str = "vllm"
    current_gpu_allocation: List[str] = Field(default_factory=list)
    worker_count: int = Field(default=1, ge=0)
    warm_workers: int = Field(default=1, ge=0)
    cold_state: bool = False
    observed_demand: Optional[WorkloadDemand] = None
    expected_memory_mb: Optional[int] = Field(default=None, ge=0)
    current_capacity_rps: Optional[float] = None
    resource_owner: str = "kubernetes"
    performance_profile_ref: Optional[str] = None
    scaling_controller: Optional[str] = None  # e.g. "dynamo-planner", "manual", "k8s-hpa"


class WorkloadDemand(BaseModel):
    model_config = {"frozen": True}
    deployment_id: str
    requests_per_second: float = 0.0
    current_concurrency: int = 0
    queue_depth: int = 0
    prompt_tokens_p50: Optional[int] = None
    prompt_tokens_p95: Optional[int] = None
    output_tokens_p50: Optional[int] = None
    output_tokens_p95: Optional[int] = None
    arrival_timestamps: Optional[List[datetime]] = None
    observed_throughput_rps: Optional[float] = None
    predicted_demand_rps: Optional[float] = None
    forecast_uncertainty: Optional[float] = None
    time_window_seconds: int = 60
    source: str = "observed"
    measured_at: datetime = Field(default_factory=datetime.utcnow)


class PerformanceProfile(BaseModel):
    model_config = {"frozen": True}
    profile_id: str
    model_id: str
    model_revision: str = "latest"
    hardware_id: str
    engine_version: str = "unknown"
    candidate_id: str
    gpu_count: int = 1
    workload_conditions: Dict[str, Any] = Field(default_factory=dict)
    evidence_type: EvidenceType = EvidenceType.UNKNOWN
    ttft_p50_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    itl_p50_ms: Optional[float] = None
    itl_p99_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    max_concurrency: Optional[int] = None
    memory_usage_mb: Optional[int] = None
    source: str = "manual"
    measured_at: datetime = Field(default_factory=datetime.utcnow)
    schema_version: str = "tesseract.performance-profile.v1"


class ServiceEconomics(BaseModel):
    model_config = {"frozen": True}
    line_item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    deployment_id: str
    billing_model: BillingModel = BillingModel.PER_TOKEN
    billed_token_rate: Optional[Decimal] = None
    billed_request_rate: Optional[Decimal] = None
    billed_hourly_reserved: Optional[Decimal] = None
    billed_hourly_on_demand: Optional[Decimal] = None
    realized_revenue: Optional[Decimal] = None
    revenue_source: str = "estimated"
    forecast_revenue: Optional[Decimal] = None
    effective_gpu_cost: Optional[Decimal] = None
    electricity_cost: Optional[Decimal] = None
    hosting_cost: Optional[Decimal] = None
    allocated_cost: Optional[Decimal] = None
    avoidable_cost: Optional[Decimal] = None
    opportunity_cost: Optional[Decimal] = None
    amortized_capital_cost: Optional[Decimal] = None
    contractual_penalty_model: Optional[str] = None
    currency: Currency = Currency.USD
    time_window_start: datetime = Field(default_factory=datetime.utcnow)
    time_window_end: Optional[datetime] = None
    assumptions: List[str] = Field(default_factory=list)
    provenance: str = "manual",
    cost_source: str = "manual"

    @property
    def contribution_margin(self) -> Optional[Decimal]:
        if self.realized_revenue is not None and self.allocated_cost is not None:
            return self.realized_revenue - self.allocated_cost
        return None


class DeploymentAlternative(BaseModel):
    model_config = {"frozen": True}
    deployment_id: str
    gpu_pool_id: str
    gpu_type: str
    replica_count: int
    expected_memory_mb: Optional[int] = None
    expected_performance: Optional[PerformanceProfile] = None
    estimated_cost_per_hour: Decimal = Decimal("0")
    estimated_revenue_per_hour: Optional[Decimal] = None
    estimated_contribution: Optional[Decimal] = None
    constraint_results: Dict[str, ConstraintResult] = Field(default_factory=dict)
    constraint_details: List[ConstraintCheck] = Field(default_factory=list)
    warm_policy: str = "warm"
    cold_start_risk: Optional[str] = None
    allocation_status: AllocationStatus = AllocationStatus.CANDIDATE
    explanation: Optional[str] = None


class ConstraintCheck(BaseModel):
    name: str
    result: ConstraintResult
    details: Optional[str] = None


class ExpectedOutcome(BaseModel):
    model_config = {"frozen": True}
    expected_revenue: Decimal = Decimal("0")
    expected_operating_cost: Decimal = Decimal("0")
    expected_contribution_margin: Optional[Decimal] = None
    modeled_risk_adjustment: Decimal = Decimal("0")
    expected_p95_latency_ms: Optional[float] = None
    expected_p99_latency_ms: Optional[float] = None
    slo_violation_probability: Optional[float] = None
    uncertainty_description: Optional[str] = None



class FusionRecommendation(BaseModel):
    model_config = {"frozen": True}
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario_id: str
    snapshot_version: str = "0.1.0"
    policy_version: str = "0.1.0"
    recommended_plan: List[DeploymentAlternative]
    baseline_plan: List[DeploymentAlternative]
    alternative_plans: List[List[DeploymentAlternative]] = Field(default_factory=list)
    expected_economics: ExpectedOutcome
    baseline_economics: ExpectedOutcome
    confidence: float = 0.5
    constraint_checks: Dict[str, ConstraintResult] = Field(default_factory=dict)
    rejected_alternatives: List[str] = Field(default_factory=list)
    rejected_reasons: Dict[str, str] = Field(default_factory=dict)
    risk_explanation: Optional[str] = None
    approval_status: RecommendationStatus = RecommendationStatus.ADVISORY_ONLY
    execution_eligibility: bool = False
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class OptimizationScenario(BaseModel):
    model_config = {"frozen": True}
    scenario_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    inventory_snapshot_ref: str = "latest"
    workload_snapshot_ref: str = "latest"
    candidate_allocation_policy: str = "economic"
    demand_forecast: Optional[ForecastResult] = None
    resource_constraints: Dict[str, Any] = Field(default_factory=dict)
    slo_constraints: Dict[str, Any] = Field(default_factory=dict)
    revenue_assumptions: Dict[str, Any] = Field(default_factory=dict)
    cost_assumptions: Dict[str, Any] = Field(default_factory=dict)
    risk_tolerance: float = 0.1
    scenario_horizon_hours: int = 24
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class ForecastResult(BaseModel):
    model_config = {"frozen": True}
    deployment_id: str
    forecast_rps: List[float]
    forecast_uncertainty: List[float]
    time_points: List[datetime]
    method: str = "moving_average"
    confidence_level: float = 0.95
    seasonality_detected: bool = False
    burst_probability: float = 0.0
    evaluation_score: Optional[float] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class SimulationState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SimulationEvent(BaseModel):
    model_config = {"frozen": True}
    timestamp: datetime
    event_type: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)


class SimulationResult(BaseModel):
    model_config = {"frozen": False}
    simulation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario_name: str
    state: SimulationState = SimulationState.PENDING
    events: List[SimulationEvent] = Field(default_factory=list)
    total_requests: int = 0
    completed_requests: int = 0
    slo_violations: int = 0
    queue_duration_p50_ms: Optional[float] = None
    queue_duration_p99_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    total_gpu_hours: float = 0.0
    total_cost: Decimal = Decimal("0")
    total_revenue: Decimal = Decimal("0")
    total_contribution: Optional[Decimal] = None
    rejected_workloads: int = 0
    scaling_events: int = 0
    migration_events: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    seed: int = 42
    config: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


