from decimal import Decimal
import pytest
from fusion.inventory.fixture_adapter import FixtureInventoryAdapter
from fusion.inventory.provider import InventoryService
from fusion.models.types import GPUInventory, HealthStatus


class TestFixtureInventoryAdapter:
    def test_load_from_dict(self):
        adapter = FixtureInventoryAdapter()
        data = [
            {"gpu_id": "gpu-0", "node_id": "node-0", "gpu_model": "H100", "memory_capacity_mb": 81920, "resource_cost_per_hour": 4.5},
            {"gpu_id": "gpu-1", "node_id": "node-1", "gpu_model": "A100", "memory_capacity_mb": 81920, "resource_cost_per_hour": 3.0, "health_status": "healthy"},
        ]
        inventory = adapter.load_from_dict(data)
        assert len(inventory) == 2
        assert inventory[0].gpu_model == "H100"
        assert float(inventory[0].resource_cost_per_hour) == 4.5
        assert inventory[1].health_status == HealthStatus.HEALTHY

    def test_empty_dict(self):
        adapter = FixtureInventoryAdapter()
        inventory = adapter.load_from_dict([])
        assert len(inventory) == 0


class TestInventoryService:
    def test_caching(self):
        adapter = FixtureInventoryAdapter()
        data = [
            {"gpu_id": "gpu-0", "node_id": "node-0", "gpu_model": "H100", "memory_capacity_mb": 81920, "resource_cost_per_hour": 4.5},
        ]
        adapter.load_from_dict(data)
        service = InventoryService(adapter, stale_threshold_seconds=3600)
        inv1 = service.get_inventory()
        inv2 = service.get_inventory()
        assert len(inv1) == 1
        assert inv1 is inv2  # cached

    def test_get_available_gpus(self):
        adapter = FixtureInventoryAdapter()
        data = [
            {"gpu_id": "gpu-0", "node_id": "node-0", "gpu_model": "H100", "memory_capacity_mb": 81920, "health_status": "healthy"},
            {"gpu_id": "gpu-1", "node_id": "node-1", "gpu_model": "A100", "memory_capacity_mb": 81920, "health_status": "unhealthy"},
        ]
        adapter.load_from_dict(data)
        service = InventoryService(adapter)
        available = service.get_available_gpus()
        assert len(available) == 1
        assert available[0].gpu_id == "gpu-0"

    def test_get_by_model(self):
        adapter = FixtureInventoryAdapter()
        data = [
            {"gpu_id": "gpu-0", "node_id": "node-0", "gpu_model": "H100", "memory_capacity_mb": 81920},
            {"gpu_id": "gpu-1", "node_id": "node-1", "gpu_model": "A100", "memory_capacity_mb": 81920},
        ]
        adapter.load_from_dict(data)
        service = InventoryService(adapter)
        h100s = service.get_gpus_by_model("H100")
        assert len(h100s) == 1


class TestConstraintValidator:
    def test_gpu_count_pass(self):
        from fusion.policies.validator import ConstraintValidator
        from fusion.models.types import GPUInventory, ModelDeployment, TenantPolicy, DeploymentAlternative
        v = ConstraintValidator()
        gpu = GPUInventory(gpu_id="gpu-0", node_id="n0", gpu_model="H100", memory_capacity_mb=81920)
        alt = DeploymentAlternative(deployment_id="d1", gpu_pool_id="p1", gpu_type="H100", replica_count=1)
        dep = ModelDeployment(deployment_id="d1", tenant_id="t1", model_id="m1")
        policy = TenantPolicy(tenant_id="t1")
        checks = v.validate_all(alt, gpu, dep, policy)
        failures = [c for c in checks if c.result == "fail"]
        assert len(failures) == 0, f"Expected no failures, got: {[(c.name, c.details) for c in failures]}"

    def test_memory_fail(self):
        from fusion.policies.validator import ConstraintValidator
        from fusion.models.types import DeploymentAlternative, ModelDeployment, TenantPolicy
        v = ConstraintValidator()
        gpu = GPUInventory(gpu_id="gpu-0", node_id="n0", gpu_model="H100", memory_capacity_mb=100)
        alt = DeploymentAlternative(deployment_id="d1", gpu_pool_id="p1", gpu_type="H100", replica_count=1)
        dep = ModelDeployment(deployment_id="d1", tenant_id="t1", model_id="m1", expected_memory_mb=99999)
        policy = TenantPolicy(tenant_id="t1")
        checks = v.validate_all(alt, gpu, dep, policy)
        memory_check = [c for c in checks if c.name == "memory_capacity"]
        assert len(memory_check) > 0
        assert memory_check[0].result == "fail"
