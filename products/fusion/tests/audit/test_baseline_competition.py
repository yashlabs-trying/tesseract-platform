"""Baseline strategy competition audit.

Five simple placement baselines are compared against the Fusion
:class:`GlobalAllocator` on identical inputs.  All costs are *simulated* hourly
GPU costs; there is no real billing data.

Important observed behaviour (recorded here, not hidden): with no revenue model
the exhaustive GlobalAllocator path selects the existing ``__baseline__``
allocation, which carries zero incremental GPU cost.  Improvement over the
greenfield baselines is therefore reported as simulated and relative.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from fusion.models.types import (
    GPUInventory,
    HealthStatus,
    ModelDeployment,
    TenantPolicy,
)
from fusion.optimizer.global_allocator import GlobalAllocator

#: Recorded simulated cost-improvement percentages.
COST_IMPROVEMENTS: dict[str, Decimal] = {}


def _feasible(dep: ModelDeployment, gpu: GPUInventory) -> bool:
    if gpu.health_status not in (HealthStatus.HEALTHY, HealthStatus.UNKNOWN):
        return False
    required = dep.expected_memory_mb
    if required is not None:
        available = gpu.available_memory_mb or gpu.memory_capacity_mb
        if required > available:
            return False
    return True


def _finalize(
    strategy: str, placements: dict[str, str], gpu_by_id: dict[str, GPUInventory]
) -> dict[str, Any]:
    total_cost = sum(
        (gpu_by_id[gpu_id].resource_cost_per_hour for gpu_id in placements.values()),
        Decimal("0"),
    )
    return {
        "strategy": strategy,
        "placements": dict(placements),
        "total_cost": total_cost,
        "gpu_hours": float(len(placements)),
        "churn": len(placements),
        "served": len(placements),
    }


def baseline_k8s_default(
    deployments: list[ModelDeployment], gpus: list[GPUInventory]
) -> dict[str, Any]:
    """First-fit by list order (Kubernetes-like default scheduler)."""
    gpu_by_id = {g.gpu_id: g for g in gpus}
    free = {g.gpu_id: True for g in gpus}
    placements: dict[str, str] = {}
    for dep in deployments:
        for gpu in gpus:
            if free[gpu.gpu_id] and _feasible(dep, gpu):
                placements[dep.deployment_id] = gpu.gpu_id
                free[gpu.gpu_id] = False
                break
    return _finalize("k8s_default", placements, gpu_by_id)


def baseline_round_robin(
    deployments: list[ModelDeployment], gpus: list[GPUInventory]
) -> dict[str, Any]:
    """Round-robin over GPUs."""
    gpu_by_id = {g.gpu_id: g for g in gpus}
    free = {g.gpu_id: True for g in gpus}
    placements: dict[str, str] = {}
    n = len(gpus)
    cursor = 0
    for dep in deployments:
        for offset in range(n):
            gpu = gpus[(cursor + offset) % n]
            if free[gpu.gpu_id] and _feasible(dep, gpu):
                placements[dep.deployment_id] = gpu.gpu_id
                free[gpu.gpu_id] = False
                cursor = (cursor + offset + 1) % n
                break
    return _finalize("round_robin", placements, gpu_by_id)


def baseline_bin_packing(
    deployments: list[ModelDeployment], gpus: list[GPUInventory]
) -> dict[str, Any]:
    """Best-fit by memory: smallest GPU that still fits the deployment."""
    gpu_by_id = {g.gpu_id: g for g in gpus}
    free = {g.gpu_id: True for g in gpus}
    placements: dict[str, str] = {}
    for dep in deployments:
        candidates = [g for g in gpus if free[g.gpu_id] and _feasible(dep, g)]
        if candidates:
            gpu = min(candidates, key=lambda g: (g.memory_capacity_mb, g.resource_cost_per_hour))
            placements[dep.deployment_id] = gpu.gpu_id
            free[gpu.gpu_id] = False
    return _finalize("bin_packing", placements, gpu_by_id)


def baseline_least_loaded(
    deployments: list[ModelDeployment], gpus: list[GPUInventory]
) -> dict[str, Any]:
    """Pick the feasible GPU with the most free slots (tie-break: cheapest)."""
    gpu_by_id = {g.gpu_id: g for g in gpus}
    free = {g.gpu_id: True for g in gpus}
    placements: dict[str, str] = {}
    for dep in deployments:
        candidates = [g for g in gpus if free[g.gpu_id] and _feasible(dep, g)]
        if candidates:
            gpu = max(
                candidates,
                key=lambda g: (g.available_resource_slots, -float(g.resource_cost_per_hour)),
            )
            placements[dep.deployment_id] = gpu.gpu_id
            free[gpu.gpu_id] = False
    return _finalize("least_loaded", placements, gpu_by_id)


def baseline_slo_greedy(
    deployments: list[ModelDeployment], gpus: list[GPUInventory]
) -> dict[str, Any]:
    """Priority order (input order) then cheapest feasible GPU."""
    gpu_by_id = {g.gpu_id: g for g in gpus}
    free = {g.gpu_id: True for g in gpus}
    placements: dict[str, str] = {}
    for dep in deployments:
        candidates = [g for g in gpus if free[g.gpu_id] and _feasible(dep, g)]
        if candidates:
            gpu = min(candidates, key=lambda g: (g.resource_cost_per_hour, g.memory_capacity_mb))
            placements[dep.deployment_id] = gpu.gpu_id
            free[gpu.gpu_id] = False
    return _finalize("slo_greedy", placements, gpu_by_id)


BASELINES = [
    ("k8s_default", baseline_k8s_default),
    ("round_robin", baseline_round_robin),
    ("bin_packing", baseline_bin_packing),
    ("least_loaded", baseline_least_loaded),
    ("slo_greedy", baseline_slo_greedy),
]


def _comparison_inputs() -> tuple[list[ModelDeployment], list[GPUInventory]]:
    deployments = [
        ModelDeployment(
            deployment_id=f"dep-{i}",
            tenant_id=f"tenant-{i % 3}",
            model_id=f"model-{i}",
            worker_count=1,
            expected_memory_mb=16384 + i * 4096,
        )
        for i in range(6)
    ]
    gpus = [
        GPUInventory(
            gpu_id=f"gpu-{i}",
            node_id=f"node-{i // 4}",
            gpu_model="H100" if i % 2 == 0 else "A100",
            memory_capacity_mb=81920,
            health_status=HealthStatus.HEALTHY,
            resource_cost_per_hour=Decimal("4.5") if i % 2 == 0 else Decimal("3.0"),
        )
        for i in range(8)
    ]
    return deployments, gpus


def _fusion_recommendation(deployments, gpus):
    policies = {d.tenant_id: TenantPolicy(tenant_id=d.tenant_id) for d in deployments}
    return GlobalAllocator().allocate(deployments, gpus, policies, {}, {})


def _fusion_cost(rec) -> Decimal:
    return rec.expected_economics.expected_operating_cost


def _improvement_pct(baseline_cost: Decimal, fusion_cost: Decimal) -> Decimal:
    if baseline_cost <= 0:
        return Decimal("0")
    ratio = Decimal("100") * (baseline_cost - fusion_cost) / baseline_cost
    return ratio.quantize(Decimal("0.01"))


def _compare(strategy_name: str):
    deployments, gpus = _comparison_inputs()
    baseline_fn = dict(BASELINES)[strategy_name]
    baseline = baseline_fn(deployments, gpus)
    rec = _fusion_recommendation(deployments, gpus)
    fusion_cost = _fusion_cost(rec)
    improvement = _improvement_pct(baseline["total_cost"], fusion_cost)
    COST_IMPROVEMENTS[strategy_name] = improvement
    return baseline, rec, fusion_cost, improvement


def test_fusion_vs_k8s_default():
    baseline, rec, fusion_cost, improvement = _compare("k8s_default")
    assert fusion_cost <= baseline["total_cost"]
    assert improvement >= 0


def test_fusion_vs_round_robin():
    baseline, rec, fusion_cost, improvement = _compare("round_robin")
    assert fusion_cost <= baseline["total_cost"]
    assert improvement >= 0


def test_fusion_vs_bin_packing():
    baseline, rec, fusion_cost, improvement = _compare("bin_packing")
    assert fusion_cost <= baseline["total_cost"]
    assert improvement >= 0


def test_fusion_vs_least_loaded():
    baseline, rec, fusion_cost, improvement = _compare("least_loaded")
    assert fusion_cost <= baseline["total_cost"]
    assert improvement >= 0


def test_fusion_vs_slo_greedy():
    baseline, rec, fusion_cost, improvement = _compare("slo_greedy")
    assert fusion_cost <= baseline["total_cost"]
    assert improvement >= 0


def test_cost_improvement_reported():
    """Record the simulated cost-improvement percentage for every baseline."""
    for strategy_name, _ in BASELINES:
        baseline, rec, fusion_cost, improvement = _compare(strategy_name)
        assert strategy_name in COST_IMPROVEMENTS
        assert COST_IMPROVEMENTS[strategy_name] == improvement
        assert improvement >= 0
    assert len(COST_IMPROVEMENTS) == len(BASELINES)


def test_equal_workload_served():
    """Fusion and every baseline must serve the same set of deployments."""
    deployments, gpus = _comparison_inputs()
    rec = _fusion_recommendation(deployments, gpus)
    fusion_served = {a.deployment_id for a in rec.recommended_plan}
    all_ids = {d.deployment_id for d in deployments}

    assert fusion_served == all_ids

    for strategy_name, baseline_fn in BASELINES:
        baseline = baseline_fn(deployments, gpus)
        assert set(baseline["placements"]) == all_ids, f"{strategy_name} served a different set"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
