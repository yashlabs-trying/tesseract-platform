"""Security and abuse audit for Tesseract Fusion.

Fusion is explicitly an **advisory-only** control plane (see
``docs/architecture.md`` and ``docs/limits.md``): production
authentication/authorization is not implemented, and no API endpoint mutates
cluster state.  These tests pin that posture down and probe the API and
optimizer for abuse resistance.

Where the observed behaviour does not match the desired security property the
test is marked ``xfail(strict=False)`` and the finding is documented in the
reason string.
"""
from __future__ import annotations

import json
import os
import tempfile
from decimal import Decimal

import pydantic
import pytest
from fastapi.testclient import TestClient

from fusion.api.app import create_app
from fusion.config.settings import Settings
from fusion.models.types import (
    DeploymentAlternative,
    GPUInventory,
    HealthStatus,
    ModelDeployment,
    TenantPolicy,
)
from fusion.optimizer import candidate as candidate_mod
from fusion.optimizer import demand_aware as demand_aware_mod
from fusion.optimizer import engine as engine_mod
from fusion.optimizer import evaluator as evaluator_mod
from fusion.optimizer import global_allocator as global_allocator_mod
from fusion.optimizer.candidate import CandidateGenerator
from fusion.optimizer.engine import OptimizationEngine
from fusion.optimizer.global_allocator import GlobalAllocator
from fusion.policies import validator as validator_mod
from fusion.policies.validator import ConstraintValidator

# --------------------------------------------------------------------------- #
# Constants / helpers
# --------------------------------------------------------------------------- #
#: POST endpoints the service is allowed to expose: imports, evaluations and
#: simulations only.  None of these mutate cluster state.
ALLOWED_POST_PATHS = {
    "/v1/inventory/import",
    "/v1/workloads/import",
    "/v1/performance-profiles/import",
    "/v1/economics/evaluate",
    "/v1/optimization/evaluate",
    "/v1/simulations",
    "/v1/forecast/evaluate",
}

#: Verbs that would indicate an endpoint capable of mutating infrastructure.
MUTATION_VERBS = (
    "scale", "delete", "remove", "apply", "execute", "mutate", "restart",
    "evict", "drain", "resize", "provision", "terminate", "migrate",
    "reconcile", "remediate",
)

#: Tokens that would indicate arbitrary code execution in the optimizer.
DANGEROUS_TOKENS = (
    "eval(", "exec(", "os.system", "subprocess", "__import__(",
    "pickle.loads", "compile(",
)

DANGEROUS_MODULES = (
    engine_mod,
    global_allocator_mod,
    candidate_mod,
    evaluator_mod,
    demand_aware_mod,
    validator_mod,
)


def make_gpu(
    gpu_id: str = "gpu-0",
    slots: int = 4,
    memory_mb: int = 81920,
    cost: str = "2.0",
    gpu_model: str = "H100",
) -> GPUInventory:
    return GPUInventory(
        gpu_id=gpu_id,
        node_id=f"node-{gpu_id}",
        gpu_model=gpu_model,
        memory_capacity_mb=memory_mb,
        available_memory_mb=memory_mb,
        available_resource_slots=slots,
        health_status=HealthStatus.HEALTHY,
        resource_cost_per_hour=Decimal(cost),
    )


def make_deployment(
    deployment_id: str = "d1",
    tenant_id: str = "t1",
    worker_count: int = 1,
    expected_memory_mb: int | None = None,
) -> ModelDeployment:
    return ModelDeployment(
        deployment_id=deployment_id,
        tenant_id=tenant_id,
        model_id="m1",
        worker_count=worker_count,
        expected_memory_mb=expected_memory_mb,
    )


def make_alt(deployment_id: str, gpu_pool_id: str, replicas: int) -> DeploymentAlternative:
    return DeploymentAlternative(
        deployment_id=deployment_id,
        gpu_pool_id=gpu_pool_id,
        gpu_type="H100",
        replica_count=replicas,
        estimated_cost_per_hour=Decimal("4.0"),
    )


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


# --------------------------------------------------------------------------- #
# 1-2. Documented defaults
# --------------------------------------------------------------------------- #
def test_no_auth_by_default_documented(monkeypatch):
    """Authentication is disabled by default (documented limitation)."""
    monkeypatch.delenv("FUSION_ENABLE_AUTH", raising=False)
    settings = Settings()
    assert settings.enable_auth is False
    # No credential is required out of the box.
    monkeypatch.delenv("FUSION_API_KEY", raising=False)
    assert Settings().api_key is None


def test_advisory_mode_default(monkeypatch):
    """Fusion never mutates cluster state unless explicitly opted in."""
    monkeypatch.delenv("FUSION_ADVISORY_MODE", raising=False)
    assert Settings().advisory_mode is True


# --------------------------------------------------------------------------- #
# 3. No mutation endpoints
# --------------------------------------------------------------------------- #
def test_no_mutation_endpoints(client):
    """Every POST endpoint is an import/evaluate/simulate, never a mutation."""
    app = client.app
    post_paths = set()
    for route in app.routes:
        methods = set(getattr(route, "methods", ()) or ())
        if "POST" in methods:
            post_paths.add(route.path)
        # No state-mutating HTTP verbs are exposed at all.
        assert not (methods & {"PUT", "PATCH", "DELETE"}), route.path

    assert post_paths, "expected the service to expose POST endpoints"
    assert post_paths <= ALLOWED_POST_PATHS, (
        f"unexpected POST endpoints: {post_paths - ALLOWED_POST_PATHS}"
    )
    for path in post_paths:
        lowered = path.lower()
        assert not any(verb in lowered for verb in MUTATION_VERBS), (
            f"mutating route detected: {path}"
        )


# --------------------------------------------------------------------------- #
# 4. Input size handling
# --------------------------------------------------------------------------- #
def test_api_input_size_handling(client):
    """A large JSON body is either handled or rejected, never hangs the server."""
    payload = [
        {
            "gpu_id": f"gpu-{i}",
            "node_id": "n0",
            "gpu_model": "H100",
            "memory_capacity_mb": 81920,
        }
        for i in range(5000)
    ]
    assert len(json.dumps(payload)) > 300_000

    response = client.post("/v1/inventory/import", json=payload)
    assert response.status_code in (200, 413, 422)

    # The service must remain responsive afterwards.
    assert client.get("/healthz").status_code == 200


# --------------------------------------------------------------------------- #
# 5. Malformed workload specs
# --------------------------------------------------------------------------- #
def test_malformed_workload_spec(client):
    """A structurally invalid body is rejected with 4xx, not 500."""
    # ``/v1/workloads/import`` expects a list; a mapping is rejected by the
    # request validator.
    response = client.post("/v1/workloads/import", json={"model_id": "m1"})
    assert response.status_code in (400, 422)
    assert response.status_code != 500


def test_malformed_workload_content_rejected(client):
    """A workload missing required fields must not produce a 500."""
    response = client.post("/v1/workloads/import", json=[{"model_id": "m1"}])
    assert response.status_code in (400, 422)


# --------------------------------------------------------------------------- #
# 6. Tenant isolation
# --------------------------------------------------------------------------- #
def test_tenant_isolation_in_models():
    """One tenant's policy failure must not affect another tenant's plan."""
    engine = OptimizationEngine()
    deployments = [
        make_deployment("d1", "t1"),
        make_deployment("d2", "t2"),
    ]
    gpus = [make_gpu("g0", slots=4), make_gpu("g1", slots=4)]
    policies = {
        "t1": TenantPolicy(tenant_id="t1", max_concurrent_workers=0),
        "t2": TenantPolicy(tenant_id="t2"),
    }

    rec = engine.optimize(deployments, gpus, policies, {}, {})
    by_deployment = {alt.deployment_id: alt for alt in rec.recommended_plan}

    assert set(by_deployment) == {"d1", "d2"}
    # t1 is quota-blocked, so its plan falls back to the no-GPU baseline.
    assert by_deployment["d1"].gpu_pool_id == "__baseline__"
    # t2 is completely unaffected by t1's policy.
    assert by_deployment["d2"].gpu_pool_id != "__baseline__"


# --------------------------------------------------------------------------- #
# 7. Quota enforcement
# --------------------------------------------------------------------------- #
def test_quota_enforcement():
    """``max_concurrent_workers`` is enforced in aggregate and per deployment."""
    alloc = GlobalAllocator()
    gpus = [make_gpu("g0", slots=8)]
    gpu_map = {g.gpu_id: g for g in gpus}
    deployments = [make_deployment("d1", "t1", 2), make_deployment("d2", "t1", 2)]
    policy = TenantPolicy(tenant_id="t1", max_concurrent_workers=2)
    policies = {"t1": policy}
    alt_one = make_alt("d1", "g0", 2)
    alt_two = make_alt("d2", "g0", 2)

    # Aggregate: 2 + 2 replicas exceeds the tenant quota of 2.
    feasible, _ = alloc._check_global_feasibility(
        [alt_one, alt_two], deployments, gpus, gpu_map, policies, {}
    )
    assert feasible is False

    # A single allocation within quota is feasible.
    feasible_one, _ = alloc._check_global_feasibility(
        [alt_one], deployments, gpus, gpu_map, policies, {}
    )
    assert feasible_one is True

    # Per-deployment: a deployment needing more workers than the quota fails.
    validator = ConstraintValidator()
    check = next(
        c
        for c in validator.validate_all(
            alt_two, gpus[0], make_deployment("d2", "t1", 3), policy, None
        )
        if c.name == "tenant_quota"
    )
    assert check.result.value == "fail"


# --------------------------------------------------------------------------- #
# 8. Priority abuse
# --------------------------------------------------------------------------- #
def test_priority_abuse():
    """A high-priority tenant still cannot exceed physical GPU capacity."""
    alloc = GlobalAllocator()
    gpus = [make_gpu("g0", slots=4)]
    gpu_map = {g.gpu_id: g for g in gpus}
    deployments = [make_deployment(f"d{i}", "t0", worker_count=2) for i in range(20)]
    policies = {"t0": TenantPolicy(tenant_id="t0", priority=1000)}
    all_alts = {
        d.deployment_id: CandidateGenerator().generate_alternatives(d, gpus)
        for d in deployments
    }

    rec = alloc._greedy_allocate(
        deployments, gpus, gpu_map, policies, {}, all_alts, []
    )

    used: dict[str, int] = {}
    for alt in rec.recommended_plan:
        if alt.gpu_pool_id == "__baseline__":
            continue
        used[alt.gpu_pool_id] = used.get(alt.gpu_pool_id, 0) + alt.replica_count

    assert used.get("g0", 0) <= gpus[0].available_resource_slots
    assert rec.provenance.get("engine") == "greedy_global"


# --------------------------------------------------------------------------- #
# 9. Untrusted scheduling request
# --------------------------------------------------------------------------- #
def test_untrusted_scheduling_request():
    """A GPU advertising impossible free memory is rejected."""
    with pytest.raises(pydantic.ValidationError):
        GPUInventory(
            gpu_id="evil",
            node_id="n0",
            gpu_model="H100",
            memory_capacity_mb=100,
            available_memory_mb=999,
        )

    # The same untrusted request through the API must not succeed.
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/inventory/import",
        json=[
            {
                "gpu_id": "evil",
                "node_id": "n0",
                "gpu_model": "H100",
                "memory_capacity_mb": 100,
                "available_memory_mb": 999,
            }
        ],
    )
    assert response.status_code >= 400


# --------------------------------------------------------------------------- #
# 10. No arbitrary code execution
# --------------------------------------------------------------------------- #
def test_no_arbitrary_code_execution():
    """The optimizer never evaluates its inputs as code."""
    import inspect

    for module in DANGEROUS_MODULES:
        source = inspect.getsource(module)
        for token in DANGEROUS_TOKENS:
            assert token not in source, f"{module.__name__} contains {token!r}"

    # Data that *looks* like code is treated as an opaque string.
    marker = os.path.join(tempfile.gettempdir(), "fusion_pwned_marker")
    if os.path.exists(marker):
        os.remove(marker)
    evil = f"__import__('os').system('touch {marker}')"

    engine = OptimizationEngine()
    rec = engine.optimize(
        [make_deployment("d1", "t1", expected_memory_mb=40960)],
        [make_gpu("g0", slots=4, gpu_model=evil)],
        {"t1": TenantPolicy(tenant_id="t1")},
        {},
        {},
    )
    assert rec is not None
    assert not os.path.exists(marker)


# --------------------------------------------------------------------------- #
# 11. Secrets are not exposed in settings repr
# --------------------------------------------------------------------------- #
def test_secrets_not_logged(monkeypatch):
    """A configured API key must not appear in ``repr(Settings())``."""
    monkeypatch.delenv("FUSION_API_KEY", raising=False)
    assert Settings().api_key is None
    settings = Settings(api_key="super-secret-value")
    assert "super-secret-value" not in repr(settings)


# --------------------------------------------------------------------------- #
# 12. Resource quotas
# --------------------------------------------------------------------------- #
def test_resource_quota_respected():
    """A tenant exceeding ``resource_quotas`` must be rejected."""
    policy = TenantPolicy(
        tenant_id="t1",
        resource_quotas={"gpu": 1},
        max_concurrent_workers=None,
    )
    # The quota is stored faithfully ...
    assert policy.resource_quotas == {"gpu": 1}

    validator = ConstraintValidator()
    checks = validator.validate_all(
        make_alt("d1", "g0", 4),
        make_gpu("g0", slots=8),
        make_deployment("d1", "t1", worker_count=4),
        policy,
        None,
    )
    # ... but it must also cause a constraint failure.
    assert any(c.result.value == "fail" and "quota" in c.name for c in checks)
