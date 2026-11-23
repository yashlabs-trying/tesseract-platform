"""Audit: boundary and edge-case behaviour of the Fusion optimizer.

Each test asserts the *observed* behaviour of the current implementation and
documents it.  Several tests record findings that are surprising (e.g. the
baseline alternative is always feasible, the engine does not enforce global
capacity, and pydantic does not reject negative resource values).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fusion.models.types import (
    ConstraintResult,
    EvidenceType,
    GPUInventory,
    HealthStatus,
    ModelDeployment,
    PerformanceProfile,
    TenantPolicy,
)
from fusion.optimizer.engine import OptimizationEngine


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_gpu(
    gpu_id: str = "gpu-0",
    memory_mb: int = 81920,
    cost_per_hour: str = "4.5",
    slots: int = 1,
    health: HealthStatus = HealthStatus.HEALTHY,
    available_memory_mb: int | None = None,
) -> GPUInventory:
    return GPUInventory(
        gpu_id=gpu_id,
        node_id=f"node-{gpu_id}",
        gpu_model="H100",
        memory_capacity_mb=memory_mb,
        available_memory_mb=available_memory_mb,
        available_resource_slots=slots,
        health_status=health,
        resource_cost_per_hour=Decimal(cost_per_hour),
    )


def make_deployment(
    deployment_id: str = "d1",
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


def make_policy(tenant_id: str = "t1", **kwargs) -> TenantPolicy:
    return TenantPolicy(tenant_id=tenant_id, **kwargs)


def plan_signature(plan) -> tuple:
    return tuple(
        (a.deployment_id, a.gpu_pool_id, a.replica_count, str(a.estimated_cost_per_hour))
        for a in plan
    )


# --------------------------------------------------------------------------- #
# Cluster / workload cardinality
# --------------------------------------------------------------------------- #
def test_empty_cluster_no_gpus():
    """No GPUs at all: the baseline plan is still emitted.

    Observation: the candidate generator always emits a ``__baseline__``
    alternative, so the engine recommends the current plan rather than failing.
    """
    rec = OptimizationEngine().optimize(
        [make_deployment()], [], {"t1": make_policy()}, {}, {}
    )
    assert len(rec.recommended_plan) == 1
    assert rec.recommended_plan[0].gpu_pool_id == "__baseline__"
    assert rec.expected_economics.expected_operating_cost == Decimal("0")


def test_empty_workload():
    """No deployments: no recommendation entries."""
    rec = OptimizationEngine().optimize([], [make_gpu(slots=2)], {}, {}, {})
    assert rec.recommended_plan == []
    assert rec.baseline_plan == []


def test_single_gpu_resource():
    """A single, sufficiently large GPU is actually selected by the engine."""
    rec = OptimizationEngine().optimize(
        [make_deployment()],
        [make_gpu(slots=2, cost_per_hour="4.5")],
        {"t1": make_policy()},
        {},
        {},
    )
    assert len(rec.recommended_plan) == 1
    assert rec.recommended_plan[0].gpu_pool_id == "gpu-0"
    # After FIX: the cheapest satisfying placement (worker_count replicas) wins.
    assert rec.recommended_plan[0].replica_count == 1


# --------------------------------------------------------------------------- #
# Capacity boundaries
# --------------------------------------------------------------------------- #
def test_all_resources_occupied():
    """available_resource_slots=0: every GPU-backed candidate is rejected.

    Observation: the engine falls back to the baseline plan and records the
    GPU as rejected with a ``gpu_count`` failure.
    """
    rec = OptimizationEngine().optimize(
        [make_deployment(worker_count=1)],
        [make_gpu(slots=0)],
        {"t1": make_policy()},
        {},
        {},
    )
    assert "gpu-0" in rec.rejected_alternatives
    assert "gpu_count" in rec.rejected_reasons["gpu-0"]
    assert rec.recommended_plan[0].gpu_pool_id == "__baseline__"


def test_gpu_requirement_exceeds_capacity():
    """replica_count > total slots: rejected by the ``gpu_count`` constraint."""
    rec = OptimizationEngine().optimize(
        [make_deployment(worker_count=1)],
        [make_gpu(slots=1)],
        {"t1": make_policy()},
        {},
        {},
    )
    assert "gpu-0" in rec.rejected_alternatives
    assert "gpu_count" in rec.rejected_reasons["gpu-0"]


def test_gpu_memory_exceeds_capacity():
    """Deployment memory > GPU memory: rejected by ``memory_capacity``."""
    rec = OptimizationEngine().optimize(
        [make_deployment(worker_count=1, expected_memory_mb=100_000)],
        [make_gpu(memory_mb=81920, slots=2)],
        {"t1": make_policy()},
        {},
        {},
    )
    assert "gpu-0" in rec.rejected_alternatives
    assert "memory_capacity" in rec.rejected_reasons["gpu-0"]


def test_zero_memory_capacity():
    """A GPU advertising 0 MB cannot host a workload that declares memory."""
    rec = OptimizationEngine().optimize(
        [make_deployment(worker_count=1, expected_memory_mb=100)],
        [make_gpu(memory_mb=0, slots=2)],
        {"t1": make_policy()},
        {},
        {},
    )
    assert "gpu-0" in rec.rejected_alternatives
    assert "memory_capacity" in rec.rejected_reasons["gpu-0"]


# --------------------------------------------------------------------------- #
# Validation boundaries
# --------------------------------------------------------------------------- #
def test_invalid_negative_values():
    """pydantic validation boundaries.

    After FIX: negative ``memory_capacity_mb``, ``available_memory_mb``,
    ``available_resource_slots`` and negative ``worker_count`` are REJECTED
    (``ge=0`` constraints), in addition to the
    ``available_memory_mb <= memory_capacity_mb`` invariant.
    """
    with pytest.raises(ValidationError):
        make_gpu(memory_mb=-1)

    with pytest.raises(ValidationError):
        make_deployment(worker_count=-5)

    with pytest.raises(ValidationError):
        GPUInventory(
            gpu_id="bad",
            node_id="n",
            gpu_model="H100",
            memory_capacity_mb=50,
            available_memory_mb=100,
        )

    with pytest.raises(ValidationError):
        GPUInventory(
            gpu_id="bad2",
            node_id="n",
            gpu_model="H100",
            memory_capacity_mb=50,
            available_resource_slots=-1,
        )


# --------------------------------------------------------------------------- #
# Missing / conflicting data
# --------------------------------------------------------------------------- #
def test_missing_metrics():
    """No profiles and no demands: confidence is the minimum (0.3)."""
    rec = OptimizationEngine().optimize(
        [make_deployment()],
        [make_gpu(slots=2)],
        {"t1": make_policy()},
        {},
        {},
    )
    assert rec.confidence == 0.3


def test_duplicate_workload_ids():
    """Two deployments sharing a deployment_id are both processed.

    Observation: the engine does not de-duplicate; it emits one recommended
    entry per input deployment.
    """
    deps = [make_deployment("d1"), make_deployment("d1")]
    rec = OptimizationEngine().optimize(
        deps, [make_gpu(slots=2)], {"t1": make_policy()}, {}, {}
    )
    assert len(rec.recommended_plan) == 2
    assert all(a.deployment_id == "d1" for a in rec.recommended_plan)


def test_conflicting_requests_same_id():
    """Two different models under the same deployment_id are both processed.

    Observation: the engine treats them as independent entries and does not
    detect the identity conflict.
    """
    deps = [
        make_deployment("d1", model_id="m1", worker_count=1),
        make_deployment("d1", model_id="m2", worker_count=1),
    ]
    rec = OptimizationEngine().optimize(
        deps, [make_gpu(slots=2)], {"t1": make_policy()}, {}, {}
    )
    assert len(rec.recommended_plan) == 2
    assert {a.deployment_id for a in rec.recommended_plan} == {"d1"}


def test_unknown_model_no_profile():
    """No performance profile: ``engine_compatibility`` is UNKNOWN, not FAIL.

    Observation: UNKNOWN constraints are surfaced as warnings and do not block
    the recommendation.
    """
    rec = OptimizationEngine().optimize(
        [make_deployment()],
        [make_gpu(slots=2)],
        {"t1": make_policy()},
        {},
        {},
    )
    assert len(rec.recommended_plan) == 1
    assert rec.recommended_plan[0].gpu_pool_id == "gpu-0"
    assert rec.constraint_checks.get("gpu-0") == ConstraintResult.UNKNOWN
    assert rec.risk_explanation is not None
    assert "UNKNOWN" in rec.risk_explanation


def test_impossible_slo():
    """Latency target below the achievable profile latency: rejected."""
    profile = PerformanceProfile(
        profile_id="p1",
        model_id="m1",
        hardware_id="H100",
        candidate_id="d1",
        engine_version="vllm",
        ttft_p99_ms=500.0,
        evidence_type=EvidenceType.MEASURED,
    )
    rec = OptimizationEngine().optimize(
        [make_deployment()],
        [make_gpu(slots=2)],
        {"t1": make_policy(latency_p99_ms=10)},
        {"d1": profile},
        {},
    )
    assert "gpu-0" in rec.rejected_alternatives
    assert "latency_slo" in rec.rejected_reasons["gpu-0"]


def test_unavailable_node():
    """Unhealthy GPUs must be rejected explicitly, never recommended.

    After FIX: unhealthy GPUs are offered as candidates and rejected by the
    constraint validator, so they appear in ``rejected_alternatives`` with a
    health reason instead of being silently dropped.
    """
    rec = OptimizationEngine().optimize(
        [make_deployment()],
        [make_gpu(health=HealthStatus.UNHEALTHY)],
        {"t1": make_policy()},
        {},
        {},
    )
    assert "gpu-0" in rec.rejected_alternatives
    assert "gpu_health" in rec.rejected_reasons["gpu-0"]
    # The unhealthy GPU must never be the recommended placement.
    assert all(
        a.gpu_pool_id != "gpu-0" for a in rec.recommended_plan
    )


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
def test_concurrent_allocation_requests():
    """Basic thread-safety smoke test: identical inputs yield identical plans."""
    engine = OptimizationEngine()
    deps = [make_deployment(f"d{i}") for i in range(3)]
    gpus = [make_gpu(f"gpu-{i}", slots=2) for i in range(3)]
    policies = {"t1": make_policy()}

    def run(_: int):
        return engine.optimize(deps, gpus, policies, {}, {})

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(16)))

    assert all(r is not None for r in results)
    signatures = {plan_signature(r.recommended_plan) for r in results}
    assert len(signatures) == 1
