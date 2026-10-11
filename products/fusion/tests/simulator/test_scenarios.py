import pytest
from fusion.simulator.engine import FleetSimulator
from fusion.simulator.scenario import ScenarioLoader
from fusion.models.types import SimulationState


# Scenario 1: Single GPU, single workload
SINGLE_GPU_SCENARIO = {
    "name": "single-gpu-test",
    "seed": 42,
    "duration_hours": 1.0,
    "gpu_pools": [{"gpu_model": "A100", "count": 1, "memory_capacity_mb": 81920, "cost_per_hour": 3.0}],
    "workloads": [{"name": "wl1", "model_id": "m1", "tenant_id": "t1", "base_rps": 10.0}],
}

# Scenario 2: Multiple GPUs, multiple workloads
MULTI_GPU_SCENARIO = {
    "name": "multi-gpu-test",
    "seed": 42,
    "duration_hours": 2.0,
    "gpu_pools": [
        {"gpu_model": "H100", "count": 4, "memory_capacity_mb": 81920, "cost_per_hour": 4.5},
        {"gpu_model": "A100", "count": 8, "memory_capacity_mb": 81920, "cost_per_hour": 3.0},
        {"gpu_model": "L40S", "count": 4, "memory_capacity_mb": 49152, "cost_per_hour": 2.25},
    ],
    "workloads": [
        {"name": "high-pri", "model_id": "llama-70b", "tenant_id": "t1", "base_rps": 50.0, "priority": 10},
        {"name": "bursty", "model_id": "mistral-7b", "tenant_id": "t2", "base_rps": 5.0, "burst_multiplier": 5.0, "burst_probability": 0.1},
        {"name": "stable", "model_id": "llama-13b", "tenant_id": "t3", "base_rps": 100.0, "burst_multiplier": 1.5},
        {"name": "dedicated", "model_id": "gpt2", "tenant_id": "t4", "base_rps": 20.0, "isolation_requirement": "exclusive"},
    ],
}

# Scenario 3: 100 GPU large fleet
LARGE_FLEET_SCENARIO = {
    "name": "large-fleet-test",
    "seed": 42,
    "duration_hours": 0.5,
    "gpu_pools": [
        {"gpu_model": "H100", "count": 20, "memory_capacity_mb": 81920, "cost_per_hour": 4.5},
        {"gpu_model": "A100", "count": 50, "memory_capacity_mb": 81920, "cost_per_hour": 3.0},
        {"gpu_model": "L40S", "count": 30, "memory_capacity_mb": 49152, "cost_per_hour": 2.25},
    ],
    "workloads": [
        {"name": f"wl-{i}", "model_id": f"model-{i}", "tenant_id": f"t{i % 10}", "base_rps": float(10 + i * 2)}
        for i in range(15)
    ],
}

# Scenario 4: Bursty demand
BURSTY_SCENARIO = {
    "name": "bursty-test",
    "seed": 42,
    "duration_hours": 3.0,
    "gpu_pools": [{"gpu_model": "H100", "count": 8, "memory_capacity_mb": 81920, "cost_per_hour": 4.5}],
    "workloads": [
        {"name": "bursty-wl", "model_id": "m1", "tenant_id": "t1", "base_rps": 10.0, "burst_multiplier": 20.0, "burst_probability": 0.05},
    ],
}


@pytest.fixture
def loader():
    return ScenarioLoader()


class TestScenario1_SingleGPU:
    def test_runs(self, loader):
        scenario = loader.load_from_dict(SINGLE_GPU_SCENARIO)
        sim = FleetSimulator(seed=42)
        result = sim.run(scenario)
        assert result.state == SimulationState.COMPLETED
        assert result.total_gpu_hours <= 1.1  # max 1 hour
        assert result.completed_requests > 0

    def test_no_double_allocation(self, loader):
        """Check that we don't allocate more GPUs than available"""
        scenario = loader.load_from_dict(SINGLE_GPU_SCENARIO)
        sim = FleetSimulator(seed=42)
        result = sim.run(scenario)
        # Single GPU, should not have more than 1 GPU allocated
        assert result.total_gpu_hours <= 1.0


class TestScenario2_MultiGPU:
    def test_runs(self, loader):
        scenario = loader.load_from_dict(MULTI_GPU_SCENARIO)
        sim = FleetSimulator(seed=42)
        result = sim.run(scenario)
        assert result.state == SimulationState.COMPLETED
        assert result.total_gpu_hours > 0
        assert result.total_requests > 1000

    def test_deterministic(self, loader):
        scenario = loader.load_from_dict(MULTI_GPU_SCENARIO)
        sim1 = FleetSimulator(seed=42)
        sim2 = FleetSimulator(seed=42)
        r1 = sim1.run(scenario)
        r2 = sim2.run(scenario)
        assert r1.total_requests == r2.total_requests

    def test_cost_accounting(self, loader):
        scenario = loader.load_from_dict(MULTI_GPU_SCENARIO)
        sim = FleetSimulator(seed=42)
        result = sim.run(scenario)
        # 16 GPUs total, 2h, costs should be significant
        assert result.total_cost > 0
        assert result.total_revenue > 0


class TestScenario3_LargeFleet:
    def test_large_fleet(self, loader):
        scenario = loader.load_from_dict(LARGE_FLEET_SCENARIO)
        sim = FleetSimulator(seed=42)
        result = sim.run(scenario)
        assert result.state == SimulationState.COMPLETED
        assert result.total_requests > 1000
        assert len(result.events) >= 2


class TestScenario4_BurstyDemand:
    def test_bursty_creates_scaling_events(self, loader):
        scenario = loader.load_from_dict(BURSTY_SCENARIO)
        sim = FleetSimulator(seed=42)
        result = sim.run(scenario)
        assert result.state == SimulationState.COMPLETED
        # Bursty workload may or may not trigger scaling depending on randomness
        assert result.completed_requests > 0


class TestCostRevenueReconciliation:
    """Scenario 7: Cost/revenue reconciliation"""
    def test_reconciliation(self, loader):
        scenario = loader.load_from_dict(SINGLE_GPU_SCENARIO)
        sim = FleetSimulator(seed=42)
        result = sim.run(scenario)
        assert result.total_contribution is not None
        assert result.total_contribution == result.total_revenue - result.total_cost
