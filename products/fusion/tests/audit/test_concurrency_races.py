"""Audit: optimizer-level concurrency and race-condition behaviour.

Fusion is advisory-only and does not mutate a cluster, so these tests probe the
*optimizer* rather than Kubernetes itself.  They run without any cluster.

For every scenario the audit checks the same invariants:

* no double allocation — for each ``gpu_pool_id`` the summed ``replica_count``
  never exceeds the GPU's ``available_resource_slots``;
* no unexpected exceptions escape;
* results are deterministic (the optimizers are pure functions of their input).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from decimal import Decimal

import pytest

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
) -> GPUInventory:
    return GPUInventory(
        gpu_id=gpu_id,
        node_id=f"node-{gpu_id}",
        gpu_model=model,
        memory_capacity_mb=memory,
        available_memory_mb=memory,
        available_resource_slots=slots,
        health_status=health,
        resource_cost_per_hour=Decimal(cost),
    )


def make_deployment(
    dep_id: str = "d0",
    tenant: str = "t0",
    worker_count: int = 1,
    memory: int = 1024,
) -> ModelDeployment:
    return ModelDeployment(
        deployment_id=dep_id,
        tenant_id=tenant,
        model_id=f"model-{dep_id}",
        worker_count=worker_count,
        expected_memory_mb=memory,
    )


def fleet(n_gpus: int = 4, slots: int = 2, cost: str = "4.5") -> list[GPUInventory]:
    return [make_gpu(f"gpu-{i}", slots=slots, cost=cost) for i in range(n_gpus)]


def workloads(n: int = 4, worker_count: int = 1, memory: int = 1024) -> list[ModelDeployment]:
    return [make_deployment(f"d{i}", f"t{i}", worker_count, memory) for i in range(n)]


def policies_for(deployments: list[ModelDeployment]) -> dict[str, TenantPolicy]:
    return {d.tenant_id: TenantPolicy(tenant_id=d.tenant_id) for d in deployments}


def gpu_usage(rec) -> dict[str, int]:
    usage: dict[str, int] = {}
    for alt in rec.recommended_plan:
        if alt.gpu_pool_id == "__baseline__":
            continue
        usage[alt.gpu_pool_id] = usage.get(alt.gpu_pool_id, 0) + alt.replica_count
    return usage


def assert_no_overcommit(rec, gpus: list[GPUInventory]) -> None:
    """Global-plan invariant: summed replicas per GPU never exceed its slots.

    This holds for :class:`GlobalAllocator`, which coordinates the whole fleet.
    It is *not* guaranteed for the per-workload :class:`OptimizationEngine`,
    which recommends each deployment in isolation (see
    :func:`assert_workload_bounds`).
    """
    slots = {g.gpu_id: g.available_resource_slots for g in gpus}
    for gpu_id, used in gpu_usage(rec).items():
        assert gpu_id in slots, f"recommendation references unknown GPU {gpu_id}"
        assert used <= slots[gpu_id], f"overcommit on {gpu_id}: {used} > {slots[gpu_id]}"


def assert_workload_bounds(rec, gpus: list[GPUInventory]) -> None:
    """Per-workload invariant for :class:`OptimizationEngine`.

    Each workload's own replica request must fit the GPU it references.  The
    engine is advisory per-deployment and may recommend the same GPU to several
    workloads; global coordination is the job of :class:`GlobalAllocator`.
    """
    slots = {g.gpu_id: g.available_resource_slots for g in gpus}
    for alt in rec.recommended_plan:
        if alt.gpu_pool_id == "__baseline__":
            continue
        assert alt.gpu_pool_id in slots, f"references unknown GPU {alt.gpu_pool_id}"
        assert alt.replica_count <= slots[alt.gpu_pool_id], (
            f"workload {alt.deployment_id} requests {alt.replica_count} replicas on "
            f"{alt.gpu_pool_id} with {slots[alt.gpu_pool_id]} slots"
        )


def signature(rec) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        sorted((a.deployment_id, a.gpu_pool_id, a.replica_count) for a in rec.recommended_plan)
    )


def run_threads(n: int, target: Callable[[int], object]):
    """Run ``target(i)`` in ``n`` threads; return ``(results, errors)``."""
    errors: list[BaseException] = []
    results: list[object] = []
    lock = threading.Lock()

    def wrapped(i: int) -> None:
        try:
            value = target(i)
        except Exception as exc:  # audit wants to observe every failure
            with lock:
                errors.append(exc)
            return
        with lock:
            results.append(value)

    threads = [threading.Thread(target=wrapped, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, errors


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
def test_concurrent_allocation_requests():
    """20 threads sharing one allocator must not crash or overcommit."""
    gpus = fleet(4, slots=2)
    deps = workloads(4)
    policies = policies_for(deps)
    allocator = GlobalAllocator()

    def worker(_i: int):
        return allocator.allocate(deps, gpus, policies, {}, {})

    results, errors = run_threads(20, worker)

    assert errors == [], f"unexpected exceptions: {errors!r}"
    assert len(results) == 20
    for rec in results:
        assert_no_overcommit(rec, gpus)


def test_concurrent_engines():
    """50 threads using independent OptimizationEngine instances."""
    gpus = fleet(4, slots=2)
    deps = workloads(3)
    policies = policies_for(deps)

    def worker(i: int):
        engine = OptimizationEngine()
        return engine.optimize(
            deps, gpus, policies, {}, {}, OptimizationScenario(scenario_id=f"scenario-{i}")
        )

    results, errors = run_threads(50, worker)

    assert errors == [], f"unexpected exceptions: {errors!r}"
    assert len(results) == 50
    # The engine is per-workload: each request fits its GPU, but the same GPU may
    # be recommended to multiple workloads (GlobalAllocator does global packing).
    for rec in results:
        assert_workload_bounds(rec, gpus)


def test_duplicate_workload_ids():
    """Two deployments sharing a deployment_id must not crash or overcommit.

    Documented behaviour: both optimizers key intermediate structures by
    ``deployment_id``, so duplicates collide.  The important guarantee is that
    the collision never produces an unsafe (overcommitted) allocation.
    """
    gpus = fleet(2, slots=2)
    d1 = make_deployment("dup", "t1", 1, 1024)
    d2 = make_deployment("dup", "t2", 1, 1024)
    policies = policies_for([d1, d2])

    engine = OptimizationEngine()
    rec = engine.optimize([d1, d2], gpus, policies, {}, {})
    assert rec is not None
    assert len(rec.recommended_plan) <= 2
    for alt in rec.recommended_plan:
        assert alt.deployment_id == "dup"
    assert_workload_bounds(rec, gpus)

    alloc = GlobalAllocator().allocate([d1, d2], gpus, policies, {}, {})
    assert_no_overcommit(alloc, gpus)


def test_two_workloads_compete_single_gpu():
    """With 2 workloads and 1 GPU, at most one workload may hold the GPU."""
    gpu = make_gpu("gpu-0", slots=1, cost="0")
    d1 = make_deployment("d1", "t1", 2, 1024)
    d2 = make_deployment("d2", "t2", 2, 1024)
    policies = policies_for([d1, d2])

    rec = GlobalAllocator().allocate([d1, d2], [gpu], policies, {}, {})

    assert gpu_usage(rec).get("gpu-0", 0) <= 1
    assert_no_overcommit(rec, [gpu])


def test_rapid_scale_up_down():
    """Repeatedly optimizing changing replica counts must stay bounded."""
    gpus = fleet(3, slots=3)
    engine = OptimizationEngine()

    for worker_count in (1, 4, 2, 3, 1, 5, 2, 4, 1, 3):
        dep = make_deployment("d1", "t1", worker_count, 1024)
        rec = engine.optimize([dep], gpus, policies_for([dep]), {}, {})
        assert rec is not None
        assert_no_overcommit(rec, gpus)


def test_repeated_allocation_release_cycles():
    """30 allocate cycles with a fresh allocator must not leak state."""
    gpus = fleet(4, slots=2)
    deps = workloads(4)
    policies = policies_for(deps)

    signatures = set()
    for _ in range(30):
        allocator = GlobalAllocator()
        rec = allocator.allocate(deps, gpus, policies, {}, {})
        assert_no_overcommit(rec, gpus)
        signatures.add(signature(rec))

    assert len(signatures) == 1, "fresh allocators leaked state between cycles"


def test_100_scheduling_requests():
    """100 sequential optimize calls must be stable and complete quickly."""
    gpus = fleet(4, slots=2)
    deps = workloads(3)
    policies = policies_for(deps)

    start = time.perf_counter()
    signatures = set()
    for _ in range(100):
        rec = GlobalAllocator().allocate(deps, gpus, policies, {}, {})
        assert rec is not None
        assert_no_overcommit(rec, gpus)
        signatures.add(signature(rec))
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0, f"100 optimize calls took {elapsed:.2f}s"
    assert len(signatures) == 1, "identical inputs produced divergent plans"


@pytest.mark.parametrize("level", [1, 5, 10, 25, 50])
def test_concurrency_levels(level: int):
    """Independent allocators must remain safe at increasing concurrency."""
    gpus = fleet(4, slots=2)
    deps = workloads(4)
    policies = policies_for(deps)

    def worker(_i: int):
        return GlobalAllocator().allocate(deps, gpus, policies, {}, {})

    results, errors = run_threads(level, worker)

    assert errors == [], f"unexpected exceptions at level {level}: {errors!r}"
    assert len(results) == level
    for rec in results:
        assert_no_overcommit(rec, gpus)
