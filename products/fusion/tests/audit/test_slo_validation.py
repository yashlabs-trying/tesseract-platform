"""SLO validation audit.

The current :class:`FleetSimulator` tracks SLO violations as a random per-step
counter and does *not* populate latency/queue percentiles.  These tests therefore
assert the SLO accounting that actually exists and explicitly document that
p50/p95/p99 latency metrics are unavailable from the simulator.

Every measurement is simulated.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from fusion.models.types import (
    DeploymentAlternative,
    EvidenceType,
    ForecastResult,
    GPUInventory,
    HealthStatus,
    ModelDeployment,
    PerformanceProfile,
    TenantPolicy,
    WorkloadDemand,
)
from fusion.optimizer.demand_aware import DemandAwareOptimizer
from fusion.optimizer.engine import OptimizationEngine
from fusion.optimizer.evaluator import AlternativeEvaluator
from fusion.simulator.engine import FleetSimulator
from fusion.simulator.scenario import ScenarioLoader

from .fleet_scenarios import class_a_steady, class_b_bursty, class_i_multi_tenant_slo

#: Recorded simulated violation rates keyed by test scenario name.
MEASURED_VIOLATION_RATES: dict[str, float] = {}


def _run(scenario_dict, seed=42):
    scenario = ScenarioLoader().load_from_dict(scenario_dict)
    return FleetSimulator(seed=seed).run(scenario)


def _violation_rate(result) -> float:
    return result.slo_violations / max(result.total_requests, 1)


def _record(name: str, result) -> float:
    rate = _violation_rate(result)
    MEASURED_VIOLATION_RATES[name] = rate
    return rate


def _simple_scenario(
    name,
    base_rps,
    gpu_count=8,
    duration_hours=2.0,
    burst_multiplier=1.0,
    burst_probability=0.0,
):
    return {
        "name": name,
        "seed": 42,
        "duration_hours": duration_hours,
        "gpu_pools": [
            {
                "gpu_model": "H100",
                "count": gpu_count,
                "memory_capacity_mb": 81920,
                "cost_per_hour": 4.5,
            }
        ],
        "workloads": [
            {
                "name": f"{name}-wl",
                "model_id": "llama-8b",
                "tenant_id": "tenant-1",
                "base_rps": base_rps,
                "burst_multiplier": burst_multiplier,
                "burst_probability": burst_probability,
            }
        ],
    }


def test_normal_load_below_capacity():
    """Steady load well below fleet capacity should produce few SLO violations."""
    result = _run(class_a_steady(seed=42, gpu_count=8))
    assert result.state.value == "completed"
    rate = _record("normal_load", result)
    assert 0.0 <= rate < 0.5
    assert result.slo_violations >= 0


def test_load_approaching_saturation():
    """At high utilisation the simulator must still account for SLO violations."""
    scenario = _simple_scenario("saturation", base_rps=400.0, gpu_count=2)
    result = _run(scenario)
    assert result.state.value == "completed"
    rate = _record("saturation", result)
    assert isinstance(result.slo_violations, int)
    assert result.slo_violations >= 0
    assert rate >= 0.0
    steps = int(scenario["duration_hours"] * 3600 / 5.0)
    assert result.slo_violations <= steps


def test_sustained_overload():
    """Sustained overload must be handled deterministically for a fixed seed."""
    scenario = _simple_scenario("overload", base_rps=1000.0, gpu_count=1)
    r1 = _run(scenario, seed=42)
    r2 = _run(scenario, seed=42)
    assert r1.slo_violations == r2.slo_violations
    assert r1.completed_requests == r2.completed_requests
    assert r1.state.value == "completed"


def test_sudden_burst():
    """A bursty workload must complete and generate requests."""
    result = _run(class_b_bursty(seed=42, gpu_count=8))
    assert result.state.value == "completed"
    assert result.total_requests > 0
    assert result.scaling_events >= 0
    _record("sudden_burst", result)


def test_multi_tenant_competing():
    """Three competing tenants must all be accounted for."""
    scenario_dict = class_i_multi_tenant_slo(seed=42, gpu_count=8)
    scenario = ScenarioLoader().load_from_dict(scenario_dict)
    policies = ScenarioLoader().to_tenant_policies(scenario)
    assert len(policies) == 3
    assert {"tenant-gold", "tenant-silver", "tenant-bronze"} <= set(policies)

    result = FleetSimulator(seed=42).run(scenario)
    assert result.state.value == "completed"
    assert result.total_requests > 0
    assert result.completed_requests > 0
    _record("multi_tenant", result)


def test_priority_sensitive():
    """Tenant priorities must be preserved through scenario loading."""
    scenario = ScenarioLoader().load_from_dict(class_i_multi_tenant_slo(seed=42, gpu_count=8))
    policies = ScenarioLoader().to_tenant_policies(scenario)
    assert policies["tenant-gold"].priority > policies["tenant-silver"].priority
    assert policies["tenant-silver"].priority > policies["tenant-bronze"].priority

    result = FleetSimulator(seed=42).run(scenario)
    assert result.state.value == "completed"


def test_unavailable_hardware():
    """An unhealthy GPU must never be selected by the optimizer."""
    engine = OptimizationEngine()
    dep = ModelDeployment(
        deployment_id="d1",
        tenant_id="t1",
        model_id="m1",
        worker_count=1,
        expected_memory_mb=40960,
    )
    unhealthy = GPUInventory(
        gpu_id="gpu-bad",
        node_id="n0",
        gpu_model="H100",
        memory_capacity_mb=81920,
        health_status=HealthStatus.UNHEALTHY,
        resource_cost_per_hour=Decimal("4.5"),
    )
    policies = {"t1": TenantPolicy(tenant_id="t1")}
    rec = engine.optimize([dep], [unhealthy], policies, {}, {})
    assert rec is not None
    assert all(a.gpu_pool_id != "gpu-bad" for a in rec.recommended_plan)


def test_missing_performance_predictions():
    """Without a performance profile the evaluator must behave conservatively."""
    evaluator = AlternativeEvaluator()
    dep = ModelDeployment(deployment_id="d1", tenant_id="t1", model_id="m1", worker_count=1)
    alt = DeploymentAlternative(
        deployment_id="d1",
        gpu_pool_id="gpu-0",
        gpu_type="H100",
        replica_count=1,
        estimated_cost_per_hour=Decimal("4.5"),
    )
    policy = TenantPolicy(tenant_id="t1")

    no_profile = evaluator.evaluate(alt, None, dep, policy, None)
    assert no_profile.slo_violation_probability == pytest.approx(0.10)
    assert no_profile.modeled_risk_adjustment > 0

    measured = PerformanceProfile(
        profile_id="p1",
        model_id="m1",
        hardware_id="H100",
        candidate_id="d1",
        ttft_p50_ms=20.0,
        ttft_p99_ms=60.0,
        throughput_rps=100.0,
        evidence_type=EvidenceType.MEASURED,
    )
    with_profile = evaluator.evaluate(alt, None, dep, policy, measured)
    assert with_profile.slo_violation_probability == pytest.approx(0.01)
    assert with_profile.slo_violation_probability < no_profile.slo_violation_probability


def test_queue_growth_bounded():
    """A saturated scenario must terminate (no unbounded queue growth)."""
    scenario = _simple_scenario("queue-bound", base_rps=5000.0, gpu_count=1, duration_hours=1.0)
    result = _run(scenario)
    assert result.state.value == "completed"
    assert result.completed_at is not None


def test_cold_vs_hot_models():
    """should_scale_to_zero must scale idle models and retain warm active models."""
    optimizer = DemandAwareOptimizer()

    idle = WorkloadDemand(deployment_id="d1", requests_per_second=0.0)
    scale, reason = optimizer.should_scale_to_zero(
        idle, None, warm_cost=Decimal("5"), cold_start_cost=Decimal("1")
    )
    assert scale is True
    assert "scale to zero" in reason.lower()

    keep, reason = optimizer.should_scale_to_zero(
        idle, None, warm_cost=Decimal("0.5"), cold_start_cost=Decimal("1")
    )
    assert keep is False
    assert "keep warm" in reason.lower()

    active = WorkloadDemand(deployment_id="d1", requests_per_second=50.0)
    keep_active, _ = optimizer.should_scale_to_zero(active, None, Decimal("5"), Decimal("1"))
    assert keep_active is False

    bursty = ForecastResult(
        deployment_id="d1",
        forecast_rps=[50.0],
        forecast_uncertainty=[1.0],
        time_points=[active.measured_at],
        burst_probability=0.6,
    )
    keep_burst, reason = optimizer.should_scale_to_zero(active, bursty, Decimal("5"), Decimal("1"))
    assert keep_burst is False
    assert "burst" in reason.lower()


def test_latency_percentiles_not_provided():
    """Document that the simulator does not provide latency percentiles.

    ``SimulationResult`` defines p50/p99 latency and queue-duration fields, but
    :class:`FleetSimulator` never populates them.  Rather than assert fabricated
    numbers, this test pins the current contract: they are ``None``.
    """
    result = _run(class_a_steady(seed=42, gpu_count=8))
    assert result.latency_p50_ms is None
    assert result.latency_p99_ms is None
    assert result.queue_duration_p50_ms is None
    assert result.queue_duration_p99_ms is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
