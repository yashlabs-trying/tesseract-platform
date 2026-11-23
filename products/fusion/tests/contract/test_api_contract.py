import pytest
from fastapi.testclient import TestClient
from fusion.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readyz(self, client):
        resp = client.get("/readyz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


class TestInventoryAPI:
    def test_get_inventory(self, client):
        resp = client.get("/v1/inventory")
        assert resp.status_code == 200

    def test_get_capacity(self, client):
        resp = client.get("/v1/inventory/capacity")
        assert resp.status_code == 200

    def test_import_inventory(self, client):
        data = [
            {"gpu_id": "gpu-0", "node_id": "n0", "gpu_model": "H100", "memory_capacity_mb": 81920},
        ]
        resp = client.post("/v1/inventory/import", json=data)
        assert resp.status_code == 200
        assert resp.json()["imported"] == 1


class TestWorkloadAPI:
    def test_import_and_list(self, client):
        data = [
            {"deployment_id": "d1", "tenant_id": "t1", "model_id": "m1"},
        ]
        resp = client.post("/v1/workloads/import", json=data)
        assert resp.status_code == 200
        resp = client.get("/v1/workloads")
        assert resp.status_code == 200

    def test_import_profile(self, client):
        data = {
            "profile_id": "p1", "model_id": "m1", "hardware_id": "H100",
            "candidate_id": "c1", "ttft_p50_ms": 10.0, "ttft_p99_ms": 20.0,
        }
        resp = client.post("/v1/performance-profiles/import", json=data)
        assert resp.status_code == 200

    def test_get_profile(self, client):
        data = {
            "profile_id": "p2", "model_id": "m2", "hardware_id": "H100",
            "candidate_id": "c2",
        }
        client.post("/v1/performance-profiles/import", json=data)
        resp = client.get("/v1/performance-profiles/p2")
        assert resp.status_code == 200


class TestOptimizationAPI:
    def test_optimization_endpoint(self, client):
        data = {"scenario": {"scenario_id": "test-scenario"}}
        resp = client.post("/v1/optimization/evaluate", json=data)
        assert resp.status_code == 200
        result = resp.json()
        assert "recommendation_id" in result

    def test_get_recommendation(self, client):
        # First create one
        client.post("/v1/optimization/evaluate", json={"scenario": {"scenario_id": "rec-test"}})
        resp = client.get("/v1/optimization/recommendations/test")
        assert resp.status_code == 404


class TestSimulationAPI:
    def test_create_simulation(self, client):
        data = {
            "scenario": {
                "name": "api-test",
                "gpu_pools": [{"gpu_model": "H100", "count": 1, "memory_capacity_mb": 81920, "cost_per_hour": 4.5}],
                "workloads": [{"name": "wl1", "model_id": "m1", "tenant_id": "t1", "base_rps": 5.0}],
            }
        }
        resp = client.post("/v1/simulations", json=data)
        assert resp.status_code == 200

    def test_get_simulation(self, client):
        data = {
            "scenario": {
                "name": "get-test",
                "gpu_pools": [{"gpu_model": "A100", "count": 1, "memory_capacity_mb": 81920, "cost_per_hour": 3.0}],
                "workloads": [{"name": "wl1", "model_id": "m1", "tenant_id": "t1", "base_rps": 5.0}],
            }
        }
        resp = client.post("/v1/simulations", json=data)
        result = resp.json()
        sim_id = result["simulation_id"]
        resp = client.get(f"/v1/simulations/{sim_id}")
        assert resp.status_code == 200


class TestForecastAPI:
    def test_forecast(self, client):
        from datetime import datetime
        data = {
            "demand_history": [
                {"deployment_id": "d1", "requests_per_second": 10.0, "measured_at": datetime.utcnow().isoformat()},
                {"deployment_id": "d1", "requests_per_second": 12.0, "measured_at": datetime.utcnow().isoformat()},
                {"deployment_id": "d1", "requests_per_second": 8.0, "measured_at": datetime.utcnow().isoformat()},
            ],
            "horizon_hours": 1,
        }
        resp = client.post("/v1/forecast/evaluate", json=data)
        assert resp.status_code == 200
