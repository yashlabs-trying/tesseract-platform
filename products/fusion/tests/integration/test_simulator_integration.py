from pathlib import Path
import pytest
from fusion.simulator.scenario import ScenarioLoader
from fusion.simulator.engine import FleetSimulator


@pytest.fixture
def small_scenario():
    loader = ScenarioLoader()
    return loader.load_from_dict({
        "name": "integration-test",
        "seed": 42,
        "duration_hours": 0.5,
        "gpu_pools": [
            {"gpu_model": "H100", "count": 2, "memory_capacity_mb": 81920, "cost_per_hour": 4.5},
        ],
        "workloads": [
            {"name": "test-wl", "model_id": "test-model", "tenant_id": "t1", "base_rps": 10.0},
        ],
    })


class TestSimulatorIntegration:
    def test_simulator_runs(self, small_scenario):
        sim = FleetSimulator(seed=42)
        result = sim.run(small_scenario)
        assert result.state.value == "completed"
        assert result.total_requests > 0
        assert result.total_gpu_hours > 0

    def test_deterministic(self, small_scenario):
        sim1 = FleetSimulator(seed=42)
        sim2 = FleetSimulator(seed=42)
        r1 = sim1.run(small_scenario)
        r2 = sim2.run(small_scenario)
        assert r1.total_requests == r2.total_requests
        assert r1.total_gpu_hours == r2.total_gpu_hours

    def test_different_seed(self, small_scenario):
        sim1 = FleetSimulator(seed=1)
        sim2 = FleetSimulator(seed=99)
        r1 = sim1.run(small_scenario)
        r2 = sim2.run(small_scenario)
        # Different seeds may give different results, but both should complete
        assert r1.state.value == "completed"
        assert r2.state.value == "completed"

    def test_costs_and_revenue(self, small_scenario):
        sim = FleetSimulator(seed=42)
        result = sim.run(small_scenario)
        assert result.total_cost > 0
        assert result.total_revenue >= 0
        assert result.total_contribution is not None

    def test_event_tracking(self, small_scenario):
        sim = FleetSimulator(seed=42)
        result = sim.run(small_scenario)
        assert len(result.events) >= 2  # start + end
        assert result.events[0].event_type == "simulation_start"
