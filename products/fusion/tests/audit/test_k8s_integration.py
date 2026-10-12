"""Audit: Kubernetes integration for the advisory-only inventory adapter.

Fusion does not run a Kubernetes controller.  The Kubernetes adapter
(:class:`fusion.adapters.kubernetes.KubernetesInventoryAdapter`) reads nodes and
turns them into :class:`GPUInventory` records.  These tests verify that:

* the client can be initialised when a cluster is present;
* the adapter parses real cluster state into inventory records;
* the adapter fails cleanly (``RuntimeError``) when no cluster is reachable;
* recommendations are always derived from *observed* ``gpu_id`` values.

Tests that need a live cluster are marked with :data:`requires_k8s` and skip
gracefully otherwise.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from fusion.adapters.kubernetes import KubernetesInventoryAdapter
from fusion.models.types import (
    GPUInventory,
    HealthStatus,
    ModelDeployment,
    TenantPolicy,
)
from fusion.optimizer.engine import OptimizationEngine

from .k8s_helpers import (
    allocatable_resources,
    count_allocatable,
    k8s_available,
    list_nodes,
    requires_k8s,
)


def _fixture_inventory() -> list[GPUInventory]:
    """Synthetic stand-in for a real cluster inventory (no cluster required)."""
    return [
        GPUInventory(
            gpu_id="node-a-gpu-0",
            cluster_id="kubernetes",
            node_id="node-a",
            vendor="nvidia",
            gpu_model="H100",
            memory_capacity_mb=81920,
            available_memory_mb=81920,
            available_resource_slots=2,
            health_status=HealthStatus.HEALTHY,
            resource_cost_per_hour=Decimal("4.5"),
            resource_cost_source="kubernetes",
        ),
        GPUInventory(
            gpu_id="node-b-gpu-0",
            cluster_id="kubernetes",
            node_id="node-b",
            vendor="nvidia",
            gpu_model="A100",
            memory_capacity_mb=81920,
            available_memory_mb=81920,
            available_resource_slots=2,
            health_status=HealthStatus.HEALTHY,
            resource_cost_per_hour=Decimal("3.0"),
            resource_cost_source="kubernetes",
        ),
    ]


# --------------------------------------------------------------------------- #
# Client / adapter
# --------------------------------------------------------------------------- #
@requires_k8s
def test_k8s_client_available():
    """The official client must be importable and constructible."""
    from kubernetes import client

    assert client.CoreV1Api() is not None
    assert k8s_available() is True


@requires_k8s
def test_adapter_reads_nodes():
    """The adapter must parse nodes into a list of GPUInventory records."""
    inventory = KubernetesInventoryAdapter().get_inventory()
    assert isinstance(inventory, list)
    for item in inventory:
        assert isinstance(item, GPUInventory)
        assert item.gpu_id
        assert item.node_id
        assert item.cluster_id == "kubernetes"


def test_adapter_handles_no_cluster():
    """Without a cluster the adapter must raise RuntimeError, not leak errors."""
    if k8s_available():
        pytest.skip("a kubernetes cluster is reachable; no-cluster path not exercised")

    with pytest.raises(RuntimeError):
        KubernetesInventoryAdapter().get_inventory()


@requires_k8s
def test_node_resource_reporting():
    """Allocatable cpu/memory (and GPU counts) must be readable per node."""
    nodes = list_nodes()
    assert nodes, "cluster reported no nodes"

    resources = allocatable_resources()
    assert set(resources) == {node.metadata.name for node in nodes}

    for name, allocatable in resources.items():
        assert isinstance(allocatable, dict)
        assert "cpu" in allocatable, f"node {name} has no allocatable cpu"
        assert "memory" in allocatable, f"node {name} has no allocatable memory"

    assert count_allocatable("nvidia.com/gpu") >= 0


@requires_k8s
def test_pod_lifecycle():
    """A probe pod can be created and then deleted."""
    from kubernetes import client
    from kubernetes.client.rest import ApiException

    v1 = client.CoreV1Api()
    name = "fusion-audit-probe"
    namespace = "default"
    body = client.V1Pod(
        metadata=client.V1ObjectMeta(name=name, labels={"app": "fusion-audit"}),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="probe",
                    image="busybox:1.36",
                    command=["sh", "-c", "sleep 60"],
                )
            ],
            restart_policy="Never",
        ),
    )

    created = False
    try:
        try:
            v1.create_namespaced_pod(namespace=namespace, body=body, _request_timeout=10)
            created = True
        except ApiException as exc:
            if exc.status == 409:
                pytest.skip("probe pod already exists")
            if exc.status in (401, 403):
                pytest.skip(f"insufficient RBAC to create pods: {exc.status}")
            raise

        pod = v1.read_namespaced_pod(name=name, namespace=namespace, _request_timeout=10)
        assert pod.metadata.name == name
    finally:
        if created:
            try:
                v1.delete_namespaced_pod(name=name, namespace=namespace, _request_timeout=10)
            except ApiException:
                pass


@requires_k8s
def test_scheduling_events_readable():
    """Scheduling events must be readable (for diagnostics only)."""
    from kubernetes import client
    from kubernetes.client.rest import ApiException

    v1 = client.CoreV1Api()
    try:
        events = v1.list_event_for_all_namespaces(limit=10, _request_timeout=10)
    except ApiException as exc:
        if exc.status in (401, 403):
            pytest.skip(f"insufficient RBAC to list events: {exc.status}")
        raise

    assert isinstance(events.items, list)


@requires_k8s
def test_rbac_readonly_sufficient():
    """Read-only operations must be sufficient for the advisory adapter."""
    from kubernetes import client

    v1 = client.CoreV1Api()
    assert v1.list_node(_request_timeout=10).items is not None
    assert v1.list_namespaced_pod("default", _request_timeout=10).items is not None
    assert v1.list_namespace(_request_timeout=10).items is not None


# --------------------------------------------------------------------------- #
# Recommendation provenance
# --------------------------------------------------------------------------- #
def test_recommendation_uses_observed_state():
    """A recommendation must only ever reference observed gpu_id values."""
    observed = []
    if k8s_available():
        observed = KubernetesInventoryAdapter().get_inventory()
    # A CPU-only cluster legitimately reports zero GPUs; fall back to the
    # fixture inventory so the provenance check still exercises the optimizer.
    if not observed:
        observed = _fixture_inventory()

    assert observed, "expected at least one observed GPU (or fixture)"

    observed_ids = {gpu.gpu_id for gpu in observed}
    deployment = ModelDeployment(
        deployment_id="audit-dep",
        tenant_id="audit-tenant",
        model_id="llama-8b",
        worker_count=1,
        expected_memory_mb=1024,
    )

    rec = OptimizationEngine().optimize(
        [deployment],
        observed,
        {"audit-tenant": TenantPolicy(tenant_id="audit-tenant")},
        {},
        {},
    )

    referenced = {
        alt.gpu_pool_id
        for alt in rec.recommended_plan
        if alt.gpu_pool_id != "__baseline__"
    }
    assert referenced <= observed_ids, (
        f"recommendation referenced unobserved GPUs: {referenced - observed_ids}"
    )
