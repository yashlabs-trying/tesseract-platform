from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fusion.models.types import GPUInventory, ModelDeployment, TenantPolicy, WorkloadDemand, PerformanceProfile


class SimulatedGPU(BaseModel):
    gpu_model: str
    count: int
    memory_capacity_mb: int = 81920
    cost_per_hour: float = 3.0
    interconnect: str = "nvlink"
    supported_isolation: List[str] = ["exclusive"]


class SimulatedWorkload(BaseModel):
    name: str
    model_id: str
    tenant_id: str
    priority: int = 0
    base_rps: float = 10.0
    burst_multiplier: float = 2.0
    burst_probability: float = 0.1
    expected_memory_mb: int = 40960
    min_replicas: int = 1
    max_replicas: int = 4
    strict_latency_target: bool = False
    isolation_requirement: str = "exclusive"
    warm_workers: int = 1
    scale_to_zero_eligible: bool = False
    duration_hours: float = 24.0


class ScenarioDefinition(BaseModel):
    name: str
    description: str = ""
    seed: int = 42
    duration_hours: float = 24.0
    gpu_pools: List[SimulatedGPU] = Field(default_factory=list)
    workloads: List[SimulatedWorkload] = Field(default_factory=list)
    tenant_policies: Dict[str, Any] = Field(default_factory=dict)
    dt_seconds: float = 5.0


class ScenarioLoader:
    def load_from_dict(self, data: Dict[str, Any]) -> ScenarioDefinition:
        return ScenarioDefinition(**data)

    def load_from_yaml(self, path: str) -> ScenarioDefinition:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return ScenarioDefinition(**data)

    def load_from_json(self, path: str) -> ScenarioDefinition:
        import json
        with open(path) as f:
            data = json.load(f)
        return ScenarioDefinition(**data)

    def to_inventory(self, scenario: ScenarioDefinition) -> List[GPUInventory]:
        inventory = []
        gpu_id = 0
        for pool in scenario.gpu_pools:
            for i in range(pool.count):
                inventory.append(GPUInventory(
                    gpu_id=f"gpu-{gpu_id}",
                    cluster_id="simulated",
                    node_id=f"node-{pool.gpu_model}-{i // 4}",
                    vendor="nvidia",
                    gpu_model=pool.gpu_model,
                    memory_capacity_mb=pool.memory_capacity_mb,
                    available_resource_slots=1,
                    interconnect=pool.interconnect,
                    supported_isolation_modes=[pool.supported_isolation[0]],
                    health_status="healthy",
                    sharing_policy="exclusive_only",
                    resource_cost_per_hour=Decimal(str(pool.cost_per_hour)),
                ))
                gpu_id += 1
        return inventory

    def to_deployments(self, scenario: ScenarioDefinition) -> List[ModelDeployment]:
        return [
            ModelDeployment(
                deployment_id=w.name,
                tenant_id=w.tenant_id,
                model_id=w.model_id,
                serving_engine="vllm",
                worker_count=w.min_replicas,
                warm_workers=w.warm_workers,
                cold_state=w.scale_to_zero_eligible and w.warm_workers == 0,
                expected_memory_mb=w.expected_memory_mb,
                scaling_controller="manual",
            )
            for w in scenario.workloads
        ]

    def to_tenant_policies(self, scenario: ScenarioDefinition) -> Dict[str, TenantPolicy]:
        policies = {}
        for w in scenario.workloads:
            if w.tenant_id not in policies:
                policies[w.tenant_id] = TenantPolicy(
                    tenant_id=w.tenant_id,
                    priority=w.priority,
                    isolation_requirements=w.isolation_requirement,
                )
        return policies
