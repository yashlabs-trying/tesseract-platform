"""Optimization-loop stability audit.

These tests exercise the hysteresis / minimum-decision-interval machinery of
:class:`DemandAwareOptimizer`, the latency constraint validator, and the
simulator's migration accounting.  Everything is simulated.

Observed behaviour that the tests encode:

* ``DemandAwareOptimizer`` compares contribution margins with ``or`` against a
  ``-inf`` sentinel, so a margin of exactly zero (the baseline allocation) is
  treated as worst-case.  A real GPU alternative is therefore always preferred
  when one is feasible.
* With a wide hysteresis threshold and an active minimum decision interval,
  alternatives within the threshold of the last decision are suppressed and the
  decision does not flip.
* With the minimum decision interval disabled, decisions are a stable function
  of near-identical demand.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from fusion.models.types import (
    DeploymentAlternative,
    EvidenceType,
    GPUInventory,
    ModelDeployment,
    PerformanceProfile,
    TenantPolicy,
    WorkloadDemand,
)
from fusion.optimizer.demand_aware import DemandAwareOptimizer
from fusion.policies.validator import ConstraintValidator
from fusion.simulator.engine import FleetSimulator
from fusion.simulator.scenario import ScenarioLoader

from .fleet_scenarios import class_j_spike
from fusion._time import utcnow_naive


def _gpu(cost: str = "2.0", gpu_id: str = "gpu-0") -> GPUInventory:
    return GPUInventory(
        gpu_id=gpu_id,
        node_id="node-0",
        gpu_model="H100",
        memory_capacity_mb=81920,
        health_status="healthy",
        resource_cost_per_hour=Decimal(cost),
    )


def _deployment(worker_count: int = 10) -> ModelDeployment:
    return ModelDeployment(
        deployment_id="d1",
        tenant_id="t1",
        model_id="m1",
        worker_count=worker_count,
        expected_memory_mb=40960,
    )


def _demand(rps: float) -> list[WorkloadDemand]:
    return [WorkloadDemand(deployment_id="d1", requests_per_second=rps)]


def _evaluate(optimizer, rps, worker_count=10, gpu_cost="2.0"):
    demand = _demand(rps)
    return optimizer.evaluate_with_demand(
        _deployment(worker_count),
        [_gpu(gpu_cost)],
        TenantPolicy(tenant_id="t1"),
        None,
        demand[0],
        demand,
    )


def test_no_oscillation_small_changes():
    """Hysteresis must prevent flips for small changes within the interval."""
    optimizer = DemandAwareOptimizer(
        {"hysteresis_threshold": 0.5, "min_decision_interval_minutes": 15}
    )
    optimizer._last_decisions["d1"] = (8, utcnow_naive())

    decisions = []
    for rps in (10.0, 10.5, 9.5, 10.2, 9.8):
        alt, _, _ = _evaluate(optimizer, rps)
        decisions.append(None if alt is None else alt.replica_count)

    # Every candidate is within 50% of the seeded decision, so all are suppressed.
    assert decisions == [None] * 5
    assert optimizer._last_decisions["d1"][0] == 8

    # With the interval disabled, near-identical demand yields a stable decision.
    stable = DemandAwareOptimizer({"min_decision_interval_minutes": 0})
    chosen = []
    for rps in (10.0, 10.5, 9.5, 10.2, 9.8):
        alt, _, _ = _evaluate(stable, rps)
        chosen.append((alt.gpu_pool_id, alt.replica_count))
    assert len(set(chosen)) == 1


def test_alternating_traffic():
    """High/low alternation must be deterministic and bounded."""
    def run_sequence():
        optimizer = DemandAwareOptimizer()
        sequence = []
        for i in range(6):
            rps = 20.0 if i % 2 == 0 else 5.0
            alt, _, _ = _evaluate(optimizer, rps)
            sequence.append(None if alt is None else (alt.gpu_pool_id, alt.replica_count))
        return sequence

    first = run_sequence()
    second = run_sequence()
    assert first == second
    # Decisions stay within the generated candidate set (never unbounded).
    assert len(set(first)) <= 3


def test_hot_cold_model_repeated():
    """Repeated hot/cold transitions must yield consistent scale decisions."""
    optimizer = DemandAwareOptimizer()
    for _ in range(3):
        idle = WorkloadDemand(deployment_id="d1", requests_per_second=0.0)
        scale, _ = optimizer.should_scale_to_zero(idle, None, Decimal("5"), Decimal("1"))
        assert scale is True

        hot = WorkloadDemand(deployment_id="d1", requests_per_second=100.0)
        keep, _ = optimizer.should_scale_to_zero(hot, None, Decimal("5"), Decimal("1"))
        assert keep is False


def test_cost_changes():
    """Changing GPU prices must move the decision to the cheaper feasible GPU."""
    optimizer = DemandAwareOptimizer({"min_decision_interval_minutes": 0})
    gpus = [_gpu("2.0", "gpu-cheap"), _gpu("5.0", "gpu-pricey")]
    alt, _, _ = optimizer.evaluate_with_demand(
        _deployment(10), gpus, TenantPolicy(tenant_id="t1"), None, _demand(10.0)[0], _demand(10.0)
    )
    assert alt.gpu_pool_id == "gpu-cheap"

    # Swap the prices; a fresh optimizer must follow the new cheapest GPU.
    optimizer2 = DemandAwareOptimizer({"min_decision_interval_minutes": 0})
    gpus2 = [_gpu("9.0", "gpu-cheap"), _gpu("1.0", "gpu-pricey")]
    alt2, _, _ = optimizer2.evaluate_with_demand(
        _deployment(10), gpus2, TenantPolicy(tenant_id="t1"), None, _demand(10.0)[0], _demand(10.0)
    )
    assert alt2.gpu_pool_id == "gpu-pricey"


def test_slo_tightening():
    """A stricter latency SLO must flip the latency constraint to FAIL."""
    validator = ConstraintValidator()
    gpu = _gpu("2.0")
    dep = _deployment(1)
    alt = DeploymentAlternative(
        deployment_id="d1",
        gpu_pool_id="gpu-0",
        gpu_type="H100",
        replica_count=1,
        estimated_cost_per_hour=Decimal("2.0"),
    )
    profile = PerformanceProfile(
        profile_id="p1",
        model_id="m1",
        hardware_id="H100",
        candidate_id="d1",
        ttft_p99_ms=500.0,
        evidence_type=EvidenceType.MEASURED,
    )

    loose = TenantPolicy(tenant_id="t1", latency_p99_ms=1000)
    strict = TenantPolicy(tenant_id="t1", latency_p99_ms=100)

    loose_check = next(
        c for c in validator.validate_all(alt, gpu, dep, loose, profile)
        if c.name == "latency_slo"
    )
    strict_check = next(
        c for c in validator.validate_all(alt, gpu, dep, strict, profile)
        if c.name == "latency_slo"
    )

    assert loose_check.result.value == "pass"
    assert strict_check.result.value == "fail"


def test_delayed_effects():
    """Cold alternatives must carry a startup delay and a reduced hourly cost."""
    captured: list[tuple[str, int, str, str, str]] = []
    optimizer = DemandAwareOptimizer(
        {"hysteresis_threshold": 0.5, "min_decision_interval_minutes": 15}
    )
    original = optimizer._evaluator.evaluate

    def capture(alt, gpu, dep, policy, profile=None):
        captured.append((
            alt.gpu_pool_id,
            alt.replica_count,
            alt.warm_policy,
            str(alt.estimated_cost_per_hour),
            alt.cold_start_risk,
        ))
        return original(alt, gpu, dep, policy, profile)

    optimizer._evaluator.evaluate = capture
    _evaluate(optimizer, 10.0, worker_count=2, gpu_cost="4.5")

    cold = [c for c in captured if c[2] == "cold"]
    assert cold, "expected cold alternatives to be evaluated"
    assert all(c[4] and "startup delay" in c[4] for c in cold)

    warm_cost = {
        c[1]: Decimal(c[3])
        for c in captured
        if c[2] == "warm" and c[0] != "__baseline__"
    }
    cold_cost = {
        c[1]: Decimal(c[3])
        for c in captured
        if c[2] == "cold" and c[0] != "__baseline__"
    }
    for replicas, cost in cold_cost.items():
        if replicas in warm_cost and warm_cost[replicas] > 0:
            assert cost < warm_cost[replicas]


def test_min_decision_interval():
    """The minimum decision interval must gate re-decisions."""
    optimizer = DemandAwareOptimizer(
        {"hysteresis_threshold": 0.5, "min_decision_interval_minutes": 15}
    )
    gpus = [_gpu("2.0")]
    policy = TenantPolicy(tenant_id="t1")

    optimizer._last_decisions["d1"] = (8, utcnow_naive())
    within, _, _ = optimizer.evaluate_with_demand(
        _deployment(10), gpus, policy, None, _demand(10.0)[0], _demand(10.0)
    )
    assert within is None

    optimizer._last_decisions["d1"] = (8, utcnow_naive() - timedelta(minutes=20))
    after, _, _ = optimizer.evaluate_with_demand(
        _deployment(10), gpus, policy, None, _demand(10.0)[0], _demand(10.0)
    )
    assert after is not None


def test_unbounded_migration_prevented():
    """The simulator must not report unbounded migrations."""
    scenario = ScenarioLoader().load_from_dict(class_j_spike(seed=42, gpu_count=8))
    result = FleetSimulator(seed=42).run(scenario)

    assert result.state.value == "completed"
    assert result.migration_events == 0
    assert result.scaling_events >= 0
    steps = int(scenario.duration_hours * 3600 / scenario.dt_seconds)
    assert result.scaling_events <= steps * (len(scenario.workloads) + 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
