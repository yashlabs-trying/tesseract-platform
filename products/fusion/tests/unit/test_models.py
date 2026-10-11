from datetime import datetime
from decimal import Decimal
import pytest
from fusion.models.types import (
    GPUInventory, TenantPolicy, ModelDeployment, WorkloadDemand,
    PerformanceProfile, ServiceEconomics, FusionRecommendation,
    DeploymentAlternative, ExpectedOutcome, ConstraintResult,
    ConstraintCheck, OptimizationScenario, SimulationResult,
    AllocationStatus, RecommendationStatus, HealthStatus,
    IsolationMode, BillingModel, Currency, EvidenceType,
)


class TestGPUInventory:
    def test_create_valid(self):
        gpu = GPUInventory(
            gpu_id="gpu-0", node_id="node-0",
            vendor="nvidia", gpu_model="H100",
            memory_capacity_mb=81920,
        )
        assert gpu.gpu_id == "gpu-0"
        assert gpu.health_status == HealthStatus.UNKNOWN
        assert gpu.resource_cost_per_hour == Decimal("0")

    def test_memory_consistency(self):
        with pytest.raises(ValueError):
            GPUInventory(
                gpu_id="gpu-0", node_id="node-0",
                gpu_model="H100", memory_capacity_mb=81920,
                available_memory_mb=99999,
            )

    def test_cost_conversion(self):
        gpu = GPUInventory(
            gpu_id="gpu-0", node_id="node-0",
            gpu_model="H100", memory_capacity_mb=81920,
            resource_cost_per_hour=4.50,
        )
        assert float(gpu.resource_cost_per_hour) == 4.50


class TestTenantPolicy:
    def test_default_isolation(self):
        policy = TenantPolicy(tenant_id="t1")
        assert policy.isolation_requirements == IsolationMode.EXCLUSIVE

    def test_default_priority(self):
        policy = TenantPolicy(tenant_id="t1")
        assert policy.priority == 0


class TestServiceEconomics:
    def test_contribution_margin(self):
        econ = ServiceEconomics(
            tenant_id="t1", deployment_id="d1",
            realized_revenue=Decimal("100"),
            allocated_cost=Decimal("60"),
        )
        assert econ.contribution_margin == Decimal("40")

    def test_no_contribution_when_missing(self):
        econ = ServiceEconomics(
            tenant_id="t1", deployment_id="d1",
            realized_revenue=Decimal("100"),
        )
        assert econ.contribution_margin is None


class TestFusionRecommendation:
    def test_advisory_default(self):
        alt = DeploymentAlternative(
            deployment_id="d1", gpu_pool_id="p1",
            gpu_type="H100", replica_count=1,
        )
        rec = FusionRecommendation(
            scenario_id="s1",
            recommended_plan=[alt],
            baseline_plan=[alt],
            expected_economics=ExpectedOutcome(),
            baseline_economics=ExpectedOutcome(),
        )
        assert rec.approval_status == RecommendationStatus.ADVISORY_ONLY
        assert rec.execution_eligibility is False


class TestSimulationResult:
    def test_contribution_computed(self):
        result = SimulationResult(
            scenario_name="test",
            total_cost=Decimal("100"),
            total_revenue=Decimal("150"),
            total_contribution=Decimal("50"),
        )
        assert result.total_contribution == Decimal("50")


class TestConstraintCheck:
    def test_pass_fail_unknown(self):
        assert ConstraintResult.PASS == "pass"
        assert ConstraintResult.FAIL == "fail"
        assert ConstraintResult.UNKNOWN == "unknown"
