"""Audit: single-workload behaviour of ``OptimizationEngine``.

These tests pin down the engine's contract for one workload and document the
observed behaviour (including that the engine always promotes a feasible
GPU-backed alternative over the baseline, regardless of contribution sign).
"""
from __future__ import annotations

from decimal import Decimal

from fusion.models.types import (
    ConstraintResult,
    GPUInventory,
    HealthStatus,
    ModelDeployment,
    TenantPolicy,
)
from fusion.optimizer.candidate import CandidateGenerator
from fusion.optimizer.engine import OptimizationEngine


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_gpu(
    gpu_id: str = "gpu-0",
    memory_mb: int = 81920,
    cost_per_hour: str = "4.5",
    slots: int = 2,
    health: HealthStatus = HealthStatus.HEALTHY,
) -> GPUInventory:
    return GPUInventory(
        gpu_id=gpu_id,
        node_id=f"node-{gpu_id}",
        gpu_model="H100",
        memory_capacity_mb=memory_mb,
        available_memory_mb=memory_mb,
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


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_baseline_and_recommended_are_different_objects():
    """Baseline and recommended entries are distinct object instances."""
    engine = OptimizationEngine()
    rec = engine.optimize(
        [make_deployment()],
        [make_gpu(slots=2)],
        {"t1": TenantPolicy(tenant_id="t1")},
        {},
        {},
    )

    assert len(rec.recommended_plan) == 1
    assert len(rec.baseline_plan) == 1
    assert rec.recommended_plan[0] is not rec.baseline_plan[0]
    assert rec.baseline_plan[0].gpu_pool_id == "__baseline__"
    assert rec.recommended_plan[0].gpu_pool_id == "gpu-0"


def test_contribution_margin_populated():
    """ExpectedOutcome contribution margins are populated (never None)."""
    engine = OptimizationEngine()
    rec = engine.optimize(
        [make_deployment()],
        [make_gpu(slots=2)],
        {"t1": TenantPolicy(tenant_id="t1")},
        {},
        {},
    )

    assert rec.expected_economics.expected_contribution_margin is not None
    assert rec.baseline_economics.expected_contribution_margin is not None
    assert (
        rec.expected_economics.expected_contribution_margin
        == rec.expected_economics.expected_revenue
        - rec.expected_economics.expected_operating_cost
    )


def test_unknown_constraints_do_not_block():
    """UNKNOWN constraints are warnings; they must not remove an alternative."""
    engine = OptimizationEngine()
    rec = engine.optimize(
        [make_deployment()],
        [make_gpu(slots=2)],
        {"t1": TenantPolicy(tenant_id="t1")},
        {},
        {},
    )

    # No profile / no memory => several UNKNOWN checks.
    assert len(rec.recommended_plan) == 1
    assert rec.recommended_plan[0].gpu_pool_id == "gpu-0"
    assert rec.constraint_checks.get("gpu-0") == ConstraintResult.UNKNOWN
    assert rec.risk_explanation is not None
    assert "UNKNOWN" in rec.risk_explanation


def test_fail_constraints_do_block():
    """FAIL constraints reject the alternative (it lands in rejected_alternatives)."""
    engine = OptimizationEngine()
    # GPU has zero available slots -> every placement needs >= 1 slot and fails.
    rec = engine.optimize(
        [make_deployment(worker_count=1)],
        [make_gpu(slots=0)],
        {"t1": TenantPolicy(tenant_id="t1")},
        {},
        {},
    )

    assert rec.rejected_alternatives != []
    assert "gpu-0" in rec.rejected_alternatives
    assert "gpu_count" in rec.rejected_reasons["gpu-0"]
    # Nothing placeable -> the recommendation falls back to the baseline.
    assert rec.recommended_plan[0].gpu_pool_id == "__baseline__"


def test_revenue_is_never_fabricated():
    """Revenue must be None unless explicit billing data is supplied."""
    engine = OptimizationEngine()
    rec = engine.optimize(
        [make_deployment()],
        [make_gpu(slots=2)],
        {"t1": TenantPolicy(tenant_id="t1")},
        {},
        {},
    )

    for alt in rec.recommended_plan:
        assert alt.estimated_revenue_per_hour is None
    for alt in rec.baseline_plan:
        assert alt.estimated_revenue_per_hour is None
    for plan in rec.alternative_plans:
        for alt in plan:
            assert alt.estimated_revenue_per_hour is None

    assert rec.expected_economics.expected_revenue == Decimal("0")
    assert rec.baseline_economics.expected_revenue == Decimal("0")


def test_memory_check_uses_per_replica_accounting():
    """The candidate's expected_memory_mb scales with the replica count."""
    dep = make_deployment(worker_count=1, expected_memory_mb=1000)
    gpu = make_gpu(slots=2)

    alts = CandidateGenerator().generate_alternatives(dep, [gpu])
    gpu_alts = [a for a in alts if a.gpu_pool_id == "gpu-0"]

    # The candidate set includes the natural 1:1 placement and scale-up.
    by_replicas = {a.replica_count: a for a in gpu_alts}
    assert 1 in by_replicas
    assert by_replicas[1].expected_memory_mb == 1000
    assert 2 in by_replicas
    assert by_replicas[2].expected_memory_mb == 2000
