"""Helpers for the optional Kubernetes-backed audit tests.

Fusion is advisory-only: the Kubernetes adapter *reads* cluster state and never
mutates it.  Every helper here is therefore read-only.  When no cluster (or no
``kubernetes`` package) is available the helpers degrade gracefully and the
tests that depend on them are skipped via :data:`requires_k8s`.
"""
from __future__ import annotations

import pytest

#: Result of the last availability probe, cached so collection is cheap.
_K8S_AVAILABLE: bool | None = None


def k8s_available() -> bool:
    """Return ``True`` only if a live cluster is reachable with read access.

    The probe is cached after the first call.  Any failure (missing package,
    missing kube-config, unreachable API server, RBAC denial) yields ``False``.
    """
    global _K8S_AVAILABLE
    if _K8S_AVAILABLE is not None:
        return _K8S_AVAILABLE
    try:
        from kubernetes import client, config

        config.load_kube_config()
        client.CoreV1Api().list_node(_request_timeout=5)
        _K8S_AVAILABLE = True
    except Exception:
        _K8S_AVAILABLE = False
    return _K8S_AVAILABLE


requires_k8s = pytest.mark.skipif(
    not k8s_available(), reason="no kubernetes cluster available"
)


def _core_v1():
    """Build a ``CoreV1Api`` client, preferring out-of-cluster config."""
    from kubernetes import client, config

    try:
        config.load_kube_config()
    except Exception:
        config.load_incluster_config()
    return client.CoreV1Api()


def list_nodes() -> list:
    """Return the raw ``V1Node`` objects visible to the current identity."""
    return _core_v1().list_node(_request_timeout=10).items


def node_names() -> list[str]:
    """Return the names of all nodes in the cluster."""
    return [node.metadata.name for node in list_nodes()]


def allocatable_resources() -> dict[str, dict[str, str]]:
    """Return ``{node_name: allocatable}`` for every node.

    ``allocatable`` is the scheduler-visible capacity (which may be smaller
    than ``capacity`` on a busy node), so it is the correct source for an
    advisory recommendation.
    """
    result: dict[str, dict[str, str]] = {}
    for node in list_nodes():
        result[node.metadata.name] = dict(node.status.allocatable or {})
    return result


def count_allocatable(resource: str = "nvidia.com/gpu") -> int:
    """Sum an allocatable resource across all nodes (best effort)."""
    total = 0
    for allocatable in allocatable_resources().values():
        try:
            total += int(allocatable.get(resource, 0))
        except (TypeError, ValueError):
            continue
    return total


def list_pods_by_namespace(namespace: str = "default") -> list:
    """Return the ``V1Pod`` objects in ``namespace``."""
    return _core_v1().list_namespaced_pod(namespace, _request_timeout=10).items
