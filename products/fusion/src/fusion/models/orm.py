from datetime import datetime
from decimal import Decimal
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Enum, Boolean, JSON, Numeric
from fusion.database.session import Base


class GPUInventoryORM(Base):
    __tablename__ = "gpu_inventory"
    gpu_id = Column(String, primary_key=True)
    cluster_id = Column(String, default="default", index=True)
    node_id = Column(String, index=True)
    vendor = Column(String, default="nvidia")
    gpu_model = Column(String)
    memory_capacity_mb = Column(Integer)
    available_memory_mb = Column(Integer, nullable=True)
    available_resource_slots = Column(Integer, default=1)
    interconnect = Column(String, nullable=True)
    topology = Column(String, nullable=True)
    supported_isolation_modes = Column(JSON, default=["exclusive"])
    health_status = Column(String, default="unknown")
    sharing_policy = Column(String, default="exclusive_only")
    location_zone = Column(String, nullable=True)
    resource_cost_per_hour = Column(Numeric(12, 4), default=0)
    currency = Column(String, default="USD")
    resource_cost_source = Column(String, default="manual")
    observed_at = Column(DateTime, default=datetime.utcnow)
    refresh_timestamp = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TenantPolicyORM(Base):
    __tablename__ = "tenant_policies"
    tenant_id = Column(String, primary_key=True)
    allowed_gpu_classes = Column(JSON, default=["*"])
    resource_quotas = Column(JSON, default={})
    priority = Column(Integer, default=0)
    isolation_requirements = Column(String, default="exclusive")
    data_residency_zones = Column(JSON, nullable=True)
    latency_p95_ms = Column(Integer, nullable=True)
    latency_p99_ms = Column(Integer, nullable=True)
    availability_objective = Column(Float, default=0.99)
    preemption_policy = Column(String, default="never")
    allowed_scheduling_flexibility = Column(Boolean, default=False)
    commercial_contract_ref = Column(String, nullable=True)
    max_concurrent_workers = Column(Integer, nullable=True)


class ModelDeploymentORM(Base):
    __tablename__ = "model_deployments"
    deployment_id = Column(String, primary_key=True)
    tenant_id = Column(String, index=True)
    model_id = Column(String)
    model_revision = Column(String, default="latest")
    serving_engine = Column(String, default="vllm")
    current_gpu_allocation = Column(JSON, default=[])
    worker_count = Column(Integer, default=1)
    warm_workers = Column(Integer, default=1)
    cold_state = Column(Boolean, default=False)
    expected_memory_mb = Column(Integer, nullable=True)
    current_capacity_rps = Column(Float, nullable=True)
    resource_owner = Column(String, default="kubernetes")
    performance_profile_ref = Column(String, nullable=True)
    scaling_controller = Column(String, nullable=True)


class WorkloadDemandORM(Base):
    __tablename__ = "workload_demand"
    id = Column(Integer, primary_key=True, autoincrement=True)
    deployment_id = Column(String, index=True)
    requests_per_second = Column(Float, default=0.0)
    current_concurrency = Column(Integer, default=0)
    queue_depth = Column(Integer, default=0)
    prompt_tokens_p50 = Column(Integer, nullable=True)
    prompt_tokens_p95 = Column(Integer, nullable=True)
    output_tokens_p50 = Column(Integer, nullable=True)
    output_tokens_p95 = Column(Integer, nullable=True)
    observed_throughput_rps = Column(Float, nullable=True)
    predicted_demand_rps = Column(Float, nullable=True)
    forecast_uncertainty = Column(Float, nullable=True)
    time_window_seconds = Column(Integer, default=60)
    source = Column(String, default="observed")
    measured_at = Column(DateTime, default=datetime.utcnow)


class PerformanceProfileORM(Base):
    __tablename__ = "performance_profiles"
    profile_id = Column(String, primary_key=True)
    model_id = Column(String, index=True)
    model_revision = Column(String, default="latest")
    hardware_id = Column(String)
    engine_version = Column(String, default="unknown")
    candidate_id = Column(String)
    gpu_count = Column(Integer, default=1)
    workload_conditions = Column(JSON, default={})
    evidence_type = Column(String, default="unknown")
    ttft_p50_ms = Column(Float, nullable=True)
    ttft_p99_ms = Column(Float, nullable=True)
    itl_p50_ms = Column(Float, nullable=True)
    itl_p99_ms = Column(Float, nullable=True)
    throughput_rps = Column(Float, nullable=True)
    max_concurrency = Column(Integer, nullable=True)
    memory_usage_mb = Column(Integer, nullable=True)
    source = Column(String, default="manual")
    measured_at = Column(DateTime, default=datetime.utcnow)
    schema_version = Column(String, default="tesseract.performance-profile.v1")


class ServiceEconomicsORM(Base):
    __tablename__ = "service_economics"
    line_item_id = Column(String, primary_key=True)
    tenant_id = Column(String, index=True)
    deployment_id = Column(String, index=True)
    billing_model = Column(String, default="per_token")
    billed_token_rate = Column(Numeric(12, 6), nullable=True)
    billed_request_rate = Column(Numeric(12, 6), nullable=True)
    billed_hourly_reserved = Column(Numeric(12, 4), nullable=True)
    billed_hourly_on_demand = Column(Numeric(12, 4), nullable=True)
    realized_revenue = Column(Numeric(14, 4), nullable=True)
    revenue_source = Column(String, default="estimated")
    forecast_revenue = Column(Numeric(14, 4), nullable=True)
    effective_gpu_cost = Column(Numeric(12, 4), nullable=True)
    electricity_cost = Column(Numeric(12, 4), nullable=True)
    hosting_cost = Column(Numeric(12, 4), nullable=True)
    allocated_cost = Column(Numeric(14, 4), nullable=True)
    avoidable_cost = Column(Numeric(14, 4), nullable=True)
    opportunity_cost = Column(Numeric(14, 4), nullable=True)
    amortized_capital_cost = Column(Numeric(14, 4), nullable=True)
    contractual_penalty_model = Column(String, nullable=True)
    currency = Column(String, default="USD")
    time_window_start = Column(DateTime, default=datetime.utcnow)
    time_window_end = Column(DateTime, nullable=True)
    assumptions = Column(JSON, default=[])
    provenance = Column(String, default="manual")
    cost_source = Column(String, default="manual")


class OptimizationScenarioORM(Base):
    __tablename__ = "optimization_scenarios"
    scenario_id = Column(String, primary_key=True)
    inventory_snapshot_ref = Column(String, default="latest")
    workload_snapshot_ref = Column(String, default="latest")
    candidate_allocation_policy = Column(String, default="economic")
    resource_constraints = Column(JSON, default={})
    slo_constraints = Column(JSON, default={})
    revenue_assumptions = Column(JSON, default={})
    cost_assumptions = Column(JSON, default={})
    risk_tolerance = Column(Float, default=0.1)
    scenario_horizon_hours = Column(Integer, default=24)
    evaluated_at = Column(DateTime, default=datetime.utcnow)


class FusionRecommendationORM(Base):
    __tablename__ = "fusion_recommendations"
    recommendation_id = Column(String, primary_key=True)
    scenario_id = Column(String, index=True)
    snapshot_version = Column(String, default="0.1.0")
    policy_version = Column(String, default="0.1.0")
    recommended_plan = Column(JSON)
    baseline_plan = Column(JSON)
    alternative_plans = Column(JSON, default=[])
    expected_economics = Column(JSON)
    baseline_economics = Column(JSON)
    confidence = Column(Float, default=0.5)
    constraint_checks = Column(JSON, default={})
    rejected_alternatives = Column(JSON, default=[])
    rejected_reasons = Column(JSON, default={})
    risk_explanation = Column(Text, nullable=True)
    approval_status = Column(String, default="advisory_only")
    execution_eligibility = Column(Boolean, default=False)
    evaluated_at = Column(DateTime, default=datetime.utcnow)
    provenance = Column(JSON, default={})


class SimulationResultORM(Base):
    __tablename__ = "simulation_results"
    simulation_id = Column(String, primary_key=True)
    scenario_name = Column(String)
    state = Column(String, default="pending")
    events = Column(JSON, default=[])
    total_requests = Column(Integer, default=0)
    completed_requests = Column(Integer, default=0)
    slo_violations = Column(Integer, default=0)
    queue_duration_p50_ms = Column(Float, nullable=True)
    queue_duration_p99_ms = Column(Float, nullable=True)
    latency_p50_ms = Column(Float, nullable=True)
    latency_p99_ms = Column(Float, nullable=True)
    total_gpu_hours = Column(Float, default=0.0)
    total_cost = Column(Numeric(14, 4), default=0)
    total_revenue = Column(Numeric(14, 4), default=0)
    total_contribution = Column(Numeric(14, 4), nullable=True)
    rejected_workloads = Column(Integer, default=0)
    scaling_events = Column(Integer, default=0)
    migration_events = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    seed = Column(Integer, default=42)
    config = Column(JSON, default={})
    error = Column(Text, nullable=True)
