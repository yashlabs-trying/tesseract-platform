"""Audit: fault injection at the optimizer / adapter level.

Fusion is advisory-only, so a "fault" can never cause an unsafe mutation of a
cluster.  The audit instead verifies that faults produce *bounded* behaviour:
clean exceptions, empty/fallback plans, correct accounting and no overcommit.

Every test asserts at least one of:

* no unsafe allocation (never more replicas than slots);
* bounded behaviour (fallback to baseline or an empty plan);
* no exception leaks (faults either raise a documented error or are handled);
* correct accounting (revenue is never fabricated, costs reconcile).
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fusion.models.types import (
    GPUInventory,
    HealthStatus,
    ModelDeployment,
    OptimizationScenario,
    TenantPolicy,
)
from fusion.optimizer.engine import OptimizationEngine
from fusion.optimizer.global_allocator import GlobalAllocator


# --------------------------------------------------------------------------- #
# Builders / invariants
# --------------------------------------------------------------------------- #
def make_gpu(
    gpu_id: str = "gpu-0",
    slots: int = 2,
    cost: str = "4.5",
    model: str = "H100",
    memory: int = 81920,
    health: HealthStatus = HealthStatus.HEALTHY,
    available_memory: int | None = None,
) -> GPUInventory:
    return GPUInventory(
        gpu_id=gpu_id,
        node_id=f"node-{gpu_id}",
        gpu_model=model,
        memory_capacity_mb=memory,
        available_memory_mb=available_memory,
        available_resource_slots=slots,
        health_status=health,
        resource_cost_per_hour=Decimal(cost),
    )


def make_deployment(
    dep_id: str = "d0",
    tenant: str = "t0",
    worker_count: int = 1,
    memory: int | None = 1024,
    model_id: str = "llama-8b",
) -> ModelDeployment:
    return ModelDeployment(
        deployment_id=dep_id,
        tenant_id=tenant,
        model_id=model_id,
        worker_count=worker_count,
        expected_memory_mb=memory,
    )


def policy(tenant: str = "t0", **kwargs) -> TenantPolicy:
    return TenantPolicy(tenant_id=tenant, **kwargs)


def gpu_usage(rec) -> dict:
    usage: dict = {}
    for alt in rec.recommended_plan:
        if alt.gpu_pool_id == "__baseline__":
            continue
        usage[alt.gpu_pool_id] = usage.get(alt.gpu_pool_id, 0) + alt.replica_count
    return usage


def assert_no_overcommit(rec, gpus) -> None:
    slots = {g.gpu_id: g.available_resource_slots for g in gpus}
    for gpu_id, used in gpu_usage(rec).items():
        assert gpu_id in slots, f"recommendation references unknown GPU {gpu_id}"
        assert used <= slots[gpu_id], f"overcommit on {gpu_id}: {used} > {slots[gpu_id]}"


def referenced_gpus(rec) -> set:
    return {a.gpu_pool_id for a in rec.recommended_plan if a.gpu_pool_id != "__baseline__"}


# --------------------------------------------------------------------------- #
# Faults
# --------------------------------------------------------------------------- #
def test_scheduler_crash_simulation():
    """An exception mid-optimization must surface cleanly, not corrupt state."""
    gpus = [make_gpu("gpu-0")]
    dep = make_deployment("d1", "t1", 1, 1024)

    engine = OptimizationEngine()

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated scheduler crash")

    engine._candidate_generator.generate_alternatives = boom

    with pytest.raises(RuntimeError, match="simulated scheduler crash"):
        engine.optimize([dep], gpus, {"t1": policy("t1")}, {}, {})

    # A fresh engine (and a fresh call) must still work — no global corruption.
    recovered = OptimizationEngine().optimize([dep], gpus, {"t1": policy("t1")}, {}, {})
    assert recovered is not None
    assert_no_overcommit(recovered, gpus)


def test_missing_metrics_provider():
    """With no profiles and no demand the optimizer must still return a plan."""
    gpus = [make_gpu("gpu-0")]
    dep = make_deployment("d1", "t1", 1, 1024)

    rec = OptimizationEngine().optimize([dep], gpus, {"t1": policy("t1")}, {}, {})
    assert rec is not None
    assert rec.confidence == 0.3  # neither profiles nor demands present
    assert_no_overcommit(rec, gpus)

    # The global allocator must also degrade gracefully.
    alloc = GlobalAllocator().allocate([dep], gpus, {"t1": policy("t1")}, {}, {})
    assert alloc is not None
    assert_no_overcommit(alloc, gpus)


def test_stale_telemetry():
    """Stale ``observed_at`` timestamps must not crash the optimizer.

    Documented behaviour: Fusion currently does *not* age out stale inventory;
    it uses whatever the adapter supplied.  The audit pins that behaviour and
    requires the recommendation to stay within the observed GPU set.
    """
    stale = datetime(2020, 1, 1, 0, 0, 0)
    gpu = make_gpu("gpu-0").model_copy(
        update={"observed_at": stale, "refresh_timestamp": stale}
    )
    dep = make_deployment("d1", "t1", 1, 1024)

    rec = OptimizationEngine().optimize([dep], [gpu], {"t1": policy("t1")}, {}, {})
    assert rec is not None
    assert referenced_gpus(rec) <= {"gpu-0"}
    assert_no_overcommit(rec, [gpu])


def test_corrupted_config():
    """Invalid configuration must fail cleanly rather than silently misbehave."""
    with pytest.raises(ValidationError):
        OptimizationScenario(scenario_id="bad", risk_tolerance="not-a-number")

    with pytest.raises(ValidationError):
        OptimizationScenario(scenario_id="bad", scenario_horizon_hours="not-an-int")

    # A nonsensical (but typed) allocator config must still yield a bounded,
    # safe plan. After FIX it falls back to the greedy path instead of
    # silently returning an empty plan.
    gpus = [make_gpu("gpu-0")]
    dep = make_deployment("d1", "t1", 1, 1024)
    allocator = GlobalAllocator({"max_exhaustive_combinations": 0})
    rec = allocator.allocate([dep], gpus, {"t1": policy("t1")}, {}, {})
    assert len(rec.recommended_plan) <= 1
    assert_no_overcommit(rec, gpus)


def test_incorrect_gpu_availability():
    """A GPU claiming zero slots must never be allocated to."""
    gpu = make_gpu("gpu-0", slots=0)
    dep = make_deployment("d1", "t1", 1, 1024)

    rec = OptimizationEngine().optimize([dep], [gpu], {"t1": policy("t1")}, {}, {})
    assert "gpu-0" not in referenced_gpus(rec)
    assert_no_overcommit(rec, [gpu])

    alloc = GlobalAllocator().allocate([dep], [gpu], {"t1": policy("t1")}, {}, {})
    assert_no_overcommit(alloc, [gpu])


def test_resource_exhaustion():
    """Requesting more than is available must fall back, never overcommit."""
    gpu = make_gpu("gpu-0", slots=1, memory=1024)
    dep = make_deployment("d1", "t1", worker_count=8, memory=4096)

    rec = OptimizationEngine().optimize([dep], [gpu], {"t1": policy("t1")}, {}, {})
    assert referenced_gpus(rec) == set()
    assert_no_overcommit(rec, [gpu])


def test_unschedulable_workload():
    """A workload with no compatible GPU must fall back to the baseline."""
    dep = make_deployment("d1", "t1", 1, 1024)

    unhealthy = make_gpu("gpu-0", health=HealthStatus.UNHEALTHY)
    rec = OptimizationEngine().optimize([dep], [unhealthy], {"t1": policy("t1")}, {}, {})
    assert referenced_gpus(rec) == set()

    empty = OptimizationEngine().optimize([dep], [], {"t1": policy("t1")}, {}, {})
    assert referenced_gpus(empty) == set()
    assert empty.recommended_plan, "baseline fallback expected"


def test_partial_allocation_failure():
    """Feasible workloads get GPUs; infeasible workloads fall back."""
    gpu_ok = make_gpu("gpu-ok", slots=2, memory=8192)
    gpu_small = make_gpu("gpu-small", slots=1, memory=512)

    d_ok = make_deployment("d-ok", "t1", worker_count=1, memory=1024)
    d_big = make_deployment("d-big", "t2", worker_count=4, memory=100000)

    policies = {"t1": policy("t1"), "t2": policy("t2")}
    rec = OptimizationEngine().optimize([d_ok, d_big], [gpu_ok, gpu_small], policies, {}, {})

    placement = {a.deployment_id: a.gpu_pool_id for a in rec.recommended_plan}
    assert placement["d-ok"] == "gpu-ok"
    assert placement["d-big"] == "__baseline__"
    assert_no_overcommit(rec, [gpu_ok, gpu_small])


def test_node_disappears():
    """Shrinking inventory between calls must be handled without overcommit."""
    dep = make_deployment("d1", "t1", 1, 1024)
    policies = {"t1": policy("t1")}
    engine = OptimizationEngine()

    before = engine.optimize([dep], [make_gpu("gpu-0"), make_gpu("gpu-1")], policies, {}, {})
    assert_no_overcommit(before, [make_gpu("gpu-0"), make_gpu("gpu-1")])

    after_gpus = [make_gpu("gpu-0")]
    after = engine.optimize([dep], after_gpus, policies, {}, {})
    assert "gpu-1" not in referenced_gpus(after)
    assert referenced_gpus(after) <= {"gpu-0"}
    assert_no_overcommit(after, after_gpus)


def test_duplicate_leader_activity():
    """Two independent allocators on identical input must agree and stay safe."""
    gpus = [make_gpu("gpu-0"), make_gpu("gpu-1", cost="3.0")]
    deps = [make_deployment("d1", "t1", 1, 1024), make_deployment("d2", "t2", 1, 1024)]
    policies = {"t1": policy("t1"), "t2": policy("t2")}

    first = GlobalAllocator().allocate(deps, gpus, policies, {}, {})
    second = GlobalAllocator().allocate(deps, gpus, policies, {}, {})

    def sig(rec):
        return sorted(
            (a.deployment_id, a.gpu_pool_id, a.replica_count)
            for a in rec.recommended_plan
        )

    assert sig(first) == sig(second)
    assert_no_overcommit(first, gpus)
    assert_no_overcommit(second, gpus)


def test_invalid_gpu_memory_negative():
    """Negative GPU memory must be rejected at the model boundary."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        make_gpu("gpu-neg", slots=2, memory=-1)


def test_unknown_model_no_profile():
    """An unknown model with no performance profile must not fabricate revenue."""
    gpu = make_gpu("gpu-0", slots=2)
    dep = make_deployment("d1", "t1", 1, 1024, model_id="experimental-unknown")

    rec = OptimizationEngine().optimize([dep], [gpu], {"t1": policy("t1")}, {}, {})

    assert rec is not None
    assert rec.confidence <= 0.5
    assert rec.expected_economics.expected_revenue == Decimal("0")
    for alt in rec.recommended_plan:
        assert alt.estimated_revenue_per_hour is None
    assert_no_overcommit(rec, [gpu])
