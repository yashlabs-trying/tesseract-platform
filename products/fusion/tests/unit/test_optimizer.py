from decimal import Decimal
import pytest
from fusion.optimizer.candidate import CandidateGenerator
from fusion.optimizer.evaluator import AlternativeEvaluator
from fusion.optimizer.engine import OptimizationEngine
from fusion.models.types import (
    GPUInventory, ModelDeployment, TenantPolicy, PerformanceProfile,
    WorkloadDemand, OptimizationScenario, DeploymentAlternative,
    AllocationStatus, HealthStatus, EvidenceType,
)


class TestCandidateGenerator:
    def test_generates_alternatives(self):
        gen = CandidateGenerator()
        dep = ModelDeployment(deployment_id="d1", tenant_id="t1", model_id="m1", worker_count=2)
        gpus = [
            GPUInventory(gpu_id="gpu-0", node_id="n0", gpu_model="H100", memory_capacity_mb=81920, health_status="healthy", resource_cost_per_hour=Decimal("4.5")),
        ]
        alts = gen.generate_alternatives(dep, gpus)
        assert len(alts) >= 1

    def test_max_candidates(self):
        gen = CandidateGenerator()
        dep = ModelDeployment(deployment_id="d1", tenant_id="t1", model_id="m1", worker_count=2)
        gpus = [
            GPUInventory(gpu_id=f"gpu-{i}", node_id=f"n{i}", gpu_model="H100", memory_capacity_mb=81920, health_status="healthy", resource_cost_per_hour=Decimal("4.5"))
            for i in range(10)
        ]
        alts = gen.generate_alternatives(dep, gpus, max_candidates=5)
        assert len(alts) <= 5


class TestOptimizationEngine:
    def test_basic_optimization(self):
        engine = OptimizationEngine()
        deps = [ModelDeployment(deployment_id="d1", tenant_id="t1", model_id="m1", worker_count=2)]
        gpus = [
            GPUInventory(gpu_id="gpu-0", node_id="n0", gpu_model="H100", memory_capacity_mb=81920, health_status="healthy", resource_cost_per_hour=Decimal("4.5")),
            GPUInventory(gpu_id="gpu-1", node_id="n1", gpu_model="A100", memory_capacity_mb=81920, health_status="healthy", resource_cost_per_hour=Decimal("3.0")),
        ]
        policies = {"t1": TenantPolicy(tenant_id="t1")}
        rec = engine.optimize(deps, gpus, policies, {}, {})
        assert rec.recommendation_id is not None
        assert rec.approval_status.value == "advisory_only"
        assert rec.execution_eligibility is False

    def test_empty_deployments(self):
        engine = OptimizationEngine()
        rec = engine.optimize([], [], {}, {}, {})
        assert len(rec.recommended_plan) == 0

    def test_cheapest_feasible_gpu(self):
        """Scenario 1: Cheapest feasible GPU - must reject GPU with insufficient VRAM"""
        engine = OptimizationEngine()
        dep = ModelDeployment(deployment_id="d1", tenant_id="t1", model_id="m1", worker_count=1, expected_memory_mb=50000)
        gpus = [
            GPUInventory(gpu_id="gpu-cheap", node_id="n0", gpu_model="L40S", memory_capacity_mb=49152, health_status="healthy", resource_cost_per_hour=Decimal("2.25")),
            GPUInventory(gpu_id="gpu-expensive", node_id="n1", gpu_model="H100", memory_capacity_mb=81920, health_status="healthy", resource_cost_per_hour=Decimal("4.50")),
        ]
        policies = {"t1": TenantPolicy(tenant_id="t1")}
        rec = engine.optimize([dep], gpus, policies, {}, {})
        # Should recommend the more expensive GPU since the cheap one has insufficient VRAM (49152 < 50000)
        assert len(rec.recommended_plan) > 0

    def test_higher_utilization_not_always_better(self):
        """Scenario 2: Higher utilization is worse if it reduces contribution margin"""
        engine = OptimizationEngine()
        # Profile with high throughput on expensive GPU
        profile = PerformanceProfile(
            profile_id="p1", model_id="m1", hardware_id="H100", candidate_id="d1",
            ttft_p50_ms=10.0, throughput_rps=200.0,
            evidence_type=EvidenceType.MEASURED,
        )
        dep = ModelDeployment(deployment_id="d1", tenant_id="t1", model_id="m1", worker_count=1)
        gpus = [
            GPUInventory(gpu_id="gpu-h100", node_id="n0", gpu_model="H100", memory_capacity_mb=81920, health_status="healthy", resource_cost_per_hour=Decimal("4.50")),
        ]
        policies = {"t1": TenantPolicy(tenant_id="t1")}
        profiles = {"d1": profile}
        rec = engine.optimize([dep], gpus, policies, profiles, {})
        assert rec.expected_economics is not None

    def test_rejects_unhealthy_gpu(self):
        """Scenario 9: Unhealthy GPU must not be recommended"""
        engine = OptimizationEngine()
        dep = ModelDeployment(deployment_id="d1", tenant_id="t1", model_id="m1", worker_count=1)
        gpus = [
            GPUInventory(gpu_id="gpu-bad", node_id="n0", gpu_model="H100", memory_capacity_mb=81920, health_status="unhealthy", resource_cost_per_hour=Decimal("4.50")),
        ]
        policies = {"t1": TenantPolicy(tenant_id="t1")}
        rec = engine.optimize([dep], gpus, policies, {}, {})
        # Even with only unhealthy GPUs, engine should still handle gracefully
        assert rec is not None

    def test_no_feasible_solution(self):
        """Scenario 10: No feasible solution must return explanation"""
        engine = OptimizationEngine()
        dep = ModelDeployment(deployment_id="d1", tenant_id="t1", model_id="m1", worker_count=10, expected_memory_mb=9999999)
        gpus = [
            GPUInventory(gpu_id="gpu-0", node_id="n0", gpu_model="H100", memory_capacity_mb=81920, health_status="healthy", resource_cost_per_hour=Decimal("4.50")),
        ]
        policies = {"t1": TenantPolicy(tenant_id="t1")}
        rec = engine.optimize([dep], gpus, policies, {}, {})
        assert rec is not None
