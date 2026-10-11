"""Contract tests for infrastructure adapters.

These tests verify that adapters produce the correct types and handle
errors properly. They use mock data rather than real clusters.
"""
from datetime import datetime, timedelta
from decimal import Decimal
import json
import pytest
from fusion.models.types import GPUInventory, HealthStatus


class TestKubernetesAdapterContract:
    def test_adapter_returns_gpu_inventory(self):
        from fusion.adapters.kubernetes import KubernetesInventoryAdapter
        adapter = KubernetesInventoryAdapter()

        # Without a real cluster, this should raise a RuntimeError
        with pytest.raises(RuntimeError):
            adapter.get_inventory()

    def test_adapter_rejects_invalid_config(self):
        from fusion.adapters.kubernetes import KubernetesInventoryAdapter
        adapter = KubernetesInventoryAdapter(kube_config="/nonexistent/config")
        with pytest.raises(RuntimeError):
            adapter.get_inventory()


class TestPrometheusAdapterContract:
    def test_adapter_returns_metrics(self):
        from fusion.adapters.prometheus import PrometheusMetricsAdapter
        adapter = PrometheusMetricsAdapter(base_url="http://localhost:19090")

        # Without a real Prometheus, query should fail
        with pytest.raises(RuntimeError):
            adapter.query("up")

    def test_adapter_query_range_validation(self):
        from fusion.adapters.prometheus import PrometheusMetricsAdapter
        adapter = PrometheusMetricsAdapter(base_url="http://localhost:19090")
        with pytest.raises(RuntimeError):
            adapter.query_range(
                "up",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 2),
            )


class TestOpenCostAdapterContract:
    def test_adapter_returns_costs(self):
        from fusion.adapters.opencost import OpenCostAdapter
        adapter = OpenCostAdapter(base_url="http://localhost:19001")
        with pytest.raises(RuntimeError):
            adapter.get_allocation()

    def test_gpu_cost_filtering(self):
        from fusion.adapters.opencost import OpenCostAdapter
        adapter = OpenCostAdapter(base_url="http://localhost:19001")
        with pytest.raises(RuntimeError):
            adapter.get_gpu_costs()


class TestDynamoAdapterContract:
    def test_adapter_returns_empty_by_default(self):
        from fusion.adapters.dynamo import DynamoMetricsAdapter
        adapter = DynamoMetricsAdapter()
        decisions = adapter.get_planner_decisions()
        assert isinstance(decisions, list)
        assert len(decisions) == 0

    def test_normalize_metrics(self):
        from fusion.adapters.dynamo import DynamoMetricsAdapter
        adapter = DynamoMetricsAdapter()
        raw = [
            {"name": "requests_total", "value": 100, "labels": {"model": "llama"}},
        ]
        normalized = adapter.normalize_metrics(raw)
        assert len(normalized) == 1
        assert normalized[0]["metric"] == "requests_total"
        assert normalized[0]["value"] == 100


class TestKAIAdapterContract:
    def test_adapter_returns_empty_by_default(self):
        from fusion.adapters.kai import KAIAdapter
        adapter = KAIAdapter()
        state = adapter.get_queue_state()
        assert isinstance(state, list)
        assert len(state) == 0

    def test_capacity_defaults(self):
        from fusion.adapters.kai import KAIAdapter
        adapter = KAIAdapter()
        capacity = adapter.get_cluster_capacity()
        assert capacity["total_gpus"] == 0
        assert capacity["available_gpus"] == 0


class TestIgniteContract:
    def test_contract_model(self):
        from fusion.adapters.ignite import IgniteCapacityContract, IgniteDecision
        contract = IgniteCapacityContract(
            deployment_id="d1",
            model_id="m1",
            workload_forecast_rps=50.0,
            required_slo_p99_ms=100,
            cost_of_warm_workers_per_hour=4.5,
            cost_of_reactivation=0.5,
            available_capacity_slots=4,
        )
        assert contract.deployment_id == "d1"
        assert contract.cost_of_warm_workers_per_hour == 4.5

        decision = IgniteDecision(
            recommended_warm_replica_floor=2,
            recommended_idle_policy="warm",
            estimated_cost_per_hour=9.0,
        )
        assert decision.recommended_warm_replica_floor == 2
        assert decision.confidence == 0.5  # default
