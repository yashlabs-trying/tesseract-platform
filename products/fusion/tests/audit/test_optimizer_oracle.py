"""Audit: compare Fusion's GlobalAllocator against an independent brute-force oracle.

These tests deliberately use a separate exhaustive oracle (``tests/audit/oracle.py``)
so that Fusion's own optimizer logic is not used to validate itself.

Behaviour after FIX (optimizer v0.3.0)
--------------------------------------
* The baseline (``__baseline__``) is a reference point only and is never a
  candidate. Every workload that *can* be placed must be placed on a real GPU.
* The objective is: maximise served replicas (weighted by tenant priority),
  then maximise (revenue - cost). With no revenue model this reduces to
  "serve everything feasible at minimum cost".
* Unhealthy GPUs are offered as candidates and rejected explicitly by the
  constraint validator, so they surface in ``rejected_alternatives``.
* An explicit ``__unplaced__`` option lets the search return a partial plan
  when physical capacity is genuinely insufficient.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from fusion.models.types import (
    GPUInventory,
    HealthStatus,
    ModelDeployment,
    TenantPolicy,
)
from fusion.optimizer.global_allocator import GlobalAllocator
from fusion.policies.validator import ConstraintValidator
from tests.audit.oracle import oracle_optimal_contribution


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_gpu(
    gpu_id: str,
    memory_mb: int = 81920,
    cost_per_hour: str = "4.5",
    slots: int = 1,
    health: HealthStatus = HealthStatus.HEALTHY,
    gpu_model: str = "H100",
) -> GPUInventory:
    """Construct a GPUInventory with an explicit available_memory_mb."""
    return GPUInventory(
        gpu_id=gpu_id,
        node_id=f"node-{gpu_id}",
        gpu_model=gpu_model,
        memory_capacity_mb=memory_mb,
        available_memory_mb=memory_mb,
        available_resource_slots=slots,
        health_status=health,
        resource_cost_per_hour=Decimal(cost_per_hour),
    )


def make_deployment(
    deployment_id: str,
    tenant_id: str = "t1",
    model_id: str = "m1",
    worker_count: int = 1,
    expected_memory_mb: int | None = None,
) -> ModelDeployment:
    return ModelDeployment(
        deployment_id=deployment_id,
        tenant_id=tenant_id,
        model_id=model_id,
        worker_count=worker_count,
        expected_memory_mb=expected_memory_mb,
    )


def make_policy(tenant_id: str = "t1", priority: int = 0) -> TenantPolicy:
    return TenantPolicy(tenant_id=tenant_id, priority=priority)


def fusion_total_cost(rec) -> Decimal:
    """Total operating cost Fusion reports for its recommended plan."""
    return rec.expected_economics.expected_operating_cost


def plan_signature(plan) -> tuple:
    return tuple(
        (
            a.deployment_id,
            a.gpu_pool_id,
            a.replica_count,
            str(a.estimated_cost_per_hour),
        )
        for a in plan
    )


def build_choices(dep: ModelDeployment, gpus: list[GPUInventory]) -> list[tuple]:
    """Mirror ``CandidateGenerator``'s option set for the oracle.

    Each option is a 4-tuple ``(gpu_id_or_None, replicas, cost, mem_per_replica)``
    as required by the oracle implementation. Placement is REQUIRED (no ``None``
    baseline option) to match the allocator's "serve every feasible workload"
    semantics.
    """
    choices: list[tuple] = []
    for gpu in gpus:
        if gpu.health_status != HealthStatus.HEALTHY:
            continue
        for replicas in (
            dep.worker_count,
            max(1, dep.worker_count - 1),
            dep.worker_count + 1,
            max(1, dep.worker_count - 2),
        ):
            choices.append(
                (
                    gpu.gpu_id,
                    replicas,
                    gpu.resource_cost_per_hour * Decimal(str(replicas)),
                    dep.expected_memory_mb or 0,
                )
            )
    return choices


def oracle_for(deps: list[ModelDeployment], gpus: list[GPUInventory]):
    choices = [build_choices(d, gpus) for d in deps]
    return oracle_optimal_contribution(deps, gpus, choices)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_single_workload_single_gpu():
    """One workload, one GPU: Fusion must place it on the GPU."""
    dep = make_deployment("d1", worker_count=1)
    gpu = make_gpu("gpu-0", cost_per_hour="4.5", slots=2)

    rec = GlobalAllocator().allocate([dep], [gpu], {"t1": make_policy()}, {}, {})

    assert len(rec.recommended_plan) == 1
    assert rec.recommended_plan[0].gpu_pool_id == "gpu-0"


def test_optimality_gap_small():
    """3 workloads / 3 GPUs: Fusion cost must equal the oracle's true optimum."""
    deps = [make_deployment(f"d{i}", worker_count=1) for i in range(3)]
    gpus = [
        make_gpu(f"gpu-{i}", cost_per_hour=str(2 + i), slots=2)
        for i in range(3)
    ]
    policies = {"t1": make_policy()}

    rec = GlobalAllocator().allocate(deps, gpus, policies, {}, {})
    fusion_cost = fusion_total_cost(rec)

    oracle_cost, oracle_choice, n = oracle_for(deps, gpus)

    assert n > 0
    assert oracle_cost is not None
    gap = fusion_cost - oracle_cost
    # Fusion's exhaustive search must match the oracle's forced-placement optimum.
    assert gap == Decimal("0"), f"optimality gap {gap} (fusion={fusion_cost}, oracle={oracle_cost})"
    # All three workloads are served.
    assert len(rec.recommended_plan) == 3


def test_infeasible_all_rejected():
    """Workload needs more memory than any GPU: verify no recommendation."""
    dep = make_deployment("d1", worker_count=1, expected_memory_mb=1_000_000)
    gpu = make_gpu("gpu-0", memory_mb=81920, slots=2)

    rec = GlobalAllocator().allocate([dep], [gpu], {"t1": make_policy()}, {}, {})

    assert len(rec.recommended_plan) == 0


def test_no_gpu_overcommit():
    """Sum of replica counts must never exceed physical GPU slots."""
    deps = [make_deployment(f"d{i}", worker_count=1) for i in range(3)]
    gpus = [make_gpu("gpu-0", slots=2), make_gpu("gpu-1", slots=1)]
    policies = {"t1": make_policy()}

    rec = GlobalAllocator().allocate(deps, gpus, policies, {}, {})

    slots = {g.gpu_id: g.available_resource_slots for g in gpus}
    usage: dict[str, int] = {}
    for alt in rec.recommended_plan:
        if alt.gpu_pool_id in slots:
            usage[alt.gpu_pool_id] = usage.get(alt.gpu_pool_id, 0) + alt.replica_count
    for gpu_id, used in usage.items():
        assert used <= slots[gpu_id]
    # 3 workloads, 3 total slots -> all three served.
    assert len(rec.recommended_plan) == 3


def test_determinism():
    """Same input run 10 times must yield the same plan (not the same uuid)."""
    deps = [make_deployment(f"d{i}", worker_count=1) for i in range(3)]
    gpus = [make_gpu(f"gpu-{i}", slots=2) for i in range(3)]
    policies = {"t1": make_policy()}

    signatures = set()
    for _ in range(10):
        rec = GlobalAllocator().allocate(deps, gpus, policies, {}, {})
        signatures.add(plan_signature(rec.recommended_plan))

    assert len(signatures) == 1


def test_cheapest_feasible_selection():
    """2 GPUs; the cheap one has insufficient memory: the expensive one wins."""
    dep = make_deployment("d1", worker_count=1, expected_memory_mb=5000)
    cheap = make_gpu("gpu-cheap", memory_mb=1000, cost_per_hour="2", slots=2)
    expensive = make_gpu("gpu-expensive", memory_mb=81920, cost_per_hour="5", slots=2)
    gpus = [cheap, expensive]
    policies = {"t1": make_policy()}

    # Independent feasibility check: cheap fails memory, expensive passes.
    validator = ConstraintValidator()
    from fusion.optimizer.candidate import CandidateGenerator

    gen = CandidateGenerator()
    cheap_alts = gen.generate_alternatives(dep, [cheap])
    exp_alts = gen.generate_alternatives(dep, [expensive])
    cheap_alt = next(a for a in cheap_alts if a.gpu_pool_id == "gpu-cheap")
    exp_alt = next(a for a in exp_alts if a.gpu_pool_id == "gpu-expensive")
    cheap_checks = validator.validate_all(cheap_alt, cheap, dep, make_policy())
    exp_checks = validator.validate_all(exp_alt, expensive, dep, make_policy())
    assert any(c.result == "fail" for c in cheap_checks)
    assert not any(c.result == "fail" for c in exp_checks)

    # Fusion's actual recommendation must select the only feasible GPU.
    rec = GlobalAllocator().allocate([dep], gpus, policies, {}, {})
    assert rec.recommended_plan[0].gpu_pool_id == "gpu-expensive"


def test_priority_respected_when_capacity_limited():
    """2 tenants, 1 GPU slot: the higher-priority tenant must win the GPU."""
    high = make_deployment("d-high", tenant_id="t-high", worker_count=1)
    low = make_deployment("d-low", tenant_id="t-low", worker_count=1)
    gpu = make_gpu("gpu-0", slots=1)
    policies = {
        "t-high": make_policy("t-high", priority=10),
        "t-low": make_policy("t-low", priority=0),
    }

    rec = GlobalAllocator().allocate([high, low], [gpu], policies, {}, {})

    by_dep = {a.deployment_id: a for a in rec.recommended_plan}
    assert by_dep["d-high"].gpu_pool_id == "gpu-0"
    assert "d-low" not in by_dep
