"""Audit tests for the simulated fleet engine.

All numbers produced here are *simulated*.  The :class:`FleetSimulator` is a
coarse-grained event loop: it does not model queueing, and it does not populate
the latency / queue percentile fields on :class:`SimulationResult` (they remain
``None``).  Tests that would normally assert p50/p95/p99 therefore document the
absence of those metrics instead of inventing them.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from fusion.models.types import (
    DeploymentAlternative,
    SimulationState,
    WorkloadDemand,
)
from fusion.optimizer.historical_replay import HistoricalReplayEngine
from fusion.simulator.engine import FleetSimulator
from fusion.simulator.scenario import ScenarioLoader

from .fleet_scenarios import (
    all_scenarios,
    class_b_bursty,
    fleet_configs,
)


def _run(scenario_dict, seed=42):
    scenario = ScenarioLoader().load_from_dict(scenario_dict)
    return FleetSimulator(seed=seed).run(scenario)


def test_all_workload_classes_run():
    """Every workload class A-O must complete at an 8-GPU fleet size."""
    for label, scenario in all_scenarios(seed=42, gpu_count=8):
        result = _run(scenario, seed=42)
        assert result.state == SimulationState.COMPLETED, f"class {label} did not complete"
        assert result.total_requests >= 0
        assert result.total_cost >= Decimal("0")


def test_all_fleet_sizes():
    """Every configured fleet size must run to completion."""
    for gpu_count, scenario in fleet_configs(seed=42):
        result = _run(scenario, seed=42)
        assert result.state == SimulationState.COMPLETED, f"fleet {gpu_count} did not complete"
        assert result.total_gpu_hours >= 0.0
        assert result.total_gpu_hours <= gpu_count * scenario["duration_hours"] + 1e-9


def test_determinism_across_seeds():
    """Same seed => identical result; different seeds are allowed to differ."""
    scenario = class_b_bursty(seed=42, gpu_count=8)

    r1 = _run(scenario, seed=42)
    r2 = _run(scenario, seed=42)
    assert r1.total_requests == r2.total_requests
    assert r1.completed_requests == r2.completed_requests
    assert r1.slo_violations == r2.slo_violations
    assert r1.total_gpu_hours == r2.total_gpu_hours

    r3 = _run(scenario, seed=1337)
    assert r3.state == SimulationState.COMPLETED
    assert r3.total_requests > 0
    # Different seeds may produce different simulated traces, but must be stable
    # for their own seed.
    r4 = _run(scenario, seed=1337)
    assert r3.total_requests == r4.total_requests


def test_no_gpu_overcommit_in_sim():
    """Allocated GPU-hours can never exceed fleet size x duration."""
    for gpu_count, scenario in fleet_configs(seed=42):
        result = _run(scenario, seed=42)
        bound = gpu_count * scenario["duration_hours"]
        assert result.total_gpu_hours <= bound + 1e-9, (
            f"overcommit for {gpu_count} GPUs: {result.total_gpu_hours} > {bound}"
        )


def test_cost_revenue_reconciliation():
    """total_contribution must reconcile exactly with revenue minus cost."""
    for label, scenario in all_scenarios(seed=42, gpu_count=8):
        result = _run(scenario, seed=42)
        assert result.total_contribution is not None, f"class {label} missing contribution"
        assert result.total_contribution == result.total_revenue - result.total_cost
        assert isinstance(result.total_cost, Decimal)
        assert isinstance(result.total_revenue, Decimal)


def test_slo_violations_tracked():
    """slo_violations must be a non-negative integer bounded by the number of steps."""
    for label, scenario in all_scenarios(seed=42, gpu_count=8):
        result = _run(scenario, seed=42)
        assert isinstance(result.slo_violations, int)
        assert result.slo_violations >= 0
        steps = int(scenario["duration_hours"] * 3600 / 5.0)
        assert result.slo_violations <= steps, f"class {label} violated step bound"


HELD_OUT_CLASSES = {"L", "M", "N", "O"}


def test_held_out_evaluation():
    """Reserve a subset of workload classes as a held-out evaluation set."""
    training = [
        (label, s) for label, s in all_scenarios(seed=42, gpu_count=8)
        if label not in HELD_OUT_CLASSES
    ]
    held_out = [
        (label, s) for label, s in all_scenarios(seed=1337, gpu_count=8)
        if label in HELD_OUT_CLASSES
    ]

    assert len(training) == 11
    assert len(held_out) == 4

    for label, scenario in held_out:
        result = _run(scenario, seed=1337)
        assert result.state == SimulationState.COMPLETED, f"held-out class {label} failed"
        assert result.total_contribution == result.total_revenue - result.total_cost


def _replay_inputs():
    scenario = ScenarioLoader().load_from_dict(class_b_bursty(seed=42, gpu_count=4))
    loader = ScenarioLoader()
    deployments = loader.to_deployments(scenario)
    gpus = loader.to_inventory(scenario)
    policies = loader.to_tenant_policies(scenario)

    demand_trace = {
        dep.deployment_id: [
            WorkloadDemand(deployment_id=dep.deployment_id, requests_per_second=10.0),
            WorkloadDemand(deployment_id=dep.deployment_id, requests_per_second=15.0),
        ]
        for dep in deployments
    }
    alternatives = [
        DeploymentAlternative(
            deployment_id=dep.deployment_id,
            gpu_pool_id=gpus[0].gpu_id,
            gpu_type=gpus[0].gpu_model,
            replica_count=max(1, dep.worker_count),
            estimated_cost_per_hour=gpus[0].resource_cost_per_hour * max(1, dep.worker_count),
        )
        for dep in deployments
    ]
    return deployments, gpus, policies, demand_trace, alternatives


def test_replay_determinism():
    """HistoricalReplayEngine is deterministic for identical inputs."""
    deployments, gpus, policies, demand_trace, alternatives = _replay_inputs()
    engine = HistoricalReplayEngine()

    r1 = engine.replay_and_compare(
        deployments, gpus, policies, {}, demand_trace, alternatives, alternatives,
    )
    r2 = engine.replay_and_compare(
        deployments, gpus, policies, {}, demand_trace, alternatives, alternatives,
    )

    assert r1.state == SimulationState.COMPLETED
    assert r1.total_cost == r2.total_cost
    assert r1.total_gpu_hours == r2.total_gpu_hours
    assert r1.completed_requests == r2.completed_requests
    assert r1.slo_violations == r2.slo_violations
    assert r1.total_contribution == r2.total_contribution


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
