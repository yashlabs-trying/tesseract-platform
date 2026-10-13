"""Ignite -> Platform adapter.

Ignite owns the Kubernetes side of the lifecycle: a ``ReplicaGroup`` CRD
(``kubbernetd.io/v1``) whose operator drives a model-server ``Deployment`` and
scales it (including scale-to-zero / wake). This adapter:

* builds the ``ReplicaGroup`` and ``Deployment`` manifests from the platform
  deployment contract (pure, testable functions);
* creates/deletes them via the Kubernetes API when a client is available;
* polls readiness, which the coordinator requires before it routes traffic.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from integrations.base import (
    DeploymentHandle,
    IntegrationError,
    IntegrationUnavailable,
)
from tesseract_platform.models import (
    PrecisionProfileRef,
    Reservation,
    WorkloadRequest,
)

_DNS_RE = re.compile(r"[^a-z0-9-]+")


def dns_name(value: str, max_length: int = 52) -> str:
    name = _DNS_RE.sub("-", value.lower()).strip("-")
    return name[:max_length] or "model"


def build_replicagroup(
    request: WorkloadRequest,
    deployment_id: str,
    *,
    namespace: str = "tesseract",
    precision: Optional[PrecisionProfileRef] = None,
    target_deployment: Optional[str] = None,
    wake_slo_seconds: int = 30,
    cold_start_timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Build an Ignite ``ReplicaGroup`` manifest (matches deploy/kubbernetd/crd.yaml)."""
    model = request.model
    replicas = max(1, request.placement.replicas or 1)
    gpu_per_worker = max(1, request.placement.min_gpu_count)
    precision_value = precision.method if precision else model.precision
    target = target_deployment or deployment_id
    return {
        "apiVersion": "kubbernetd.io/v1",
        "kind": "ReplicaGroup",
        "metadata": {
            "name": dns_name(deployment_id),
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/managed-by": "tesseract-platform",
                "tesseract.io/workload": dns_name(request.workload_id),
            },
        },
        "spec": {
            "model": {
                "name": model.model_id,
                "engine": "vLLM",
                "precision": precision_value,
                "tensorParallel": gpu_per_worker,
                "pipelineParallel": 1,
                "gpuPerWorker": gpu_per_worker,
                "workers": replicas,
            },
            "targetRef": {"kind": "Deployment", "name": target},
            "wakeSLO": wake_slo_seconds,
            "coldStartTimeout": cold_start_timeout_seconds,
            "sleepPolicy": {"sleepDepth": "full", "idleTimeout": 300},
        },
    }


def build_model_deployment(
    request: WorkloadRequest,
    deployment_id: str,
    *,
    namespace: str = "tesseract",
    image: str = "vllm/vllm-openai:latest",
    port: int = 8000,
) -> dict[str, Any]:
    """Build the model-server ``Deployment`` that a ReplicaGroup targets."""
    model = request.model
    replicas = max(1, request.placement.replicas or 1)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": dns_name(deployment_id),
            "namespace": namespace,
            "labels": {"app.kubernetes.io/managed-by": "tesseract-platform"},
        },
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": dns_name(deployment_id)}},
            "template": {
                "metadata": {"labels": {"app": dns_name(deployment_id)}},
                "spec": {
                    "containers": [
                        {
                            "name": "model-server",
                            "image": image,
                            "args": ["--model", model.model_id, "--port", str(port)],
                            "ports": [{"containerPort": port}],
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": port},
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5,
                            },
                        }
                    ]
                },
            },
        },
    }


class IgniteAdapter:
    """Ignite :class:`DeploymentProvider` implementation."""

    def __init__(
        self,
        *,
        namespace: str = "tesseract",
        kubeconfig: Optional[str] = None,
        in_cluster: bool = False,
        image: str = "vllm/vllm-openai:latest",
        client: Any = None,
        poll_interval: float = 2.0,
    ) -> None:
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.in_cluster = in_cluster
        self.image = image
        self._client = client
        self.poll_interval = poll_interval

    # ---------------------------------------------------------------- k8s client

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from kubernetes import client as k8s_client  # type: ignore
            from kubernetes import config as k8s_config  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise IntegrationUnavailable(
                "kubernetes python client is not installed"
            ) from exc
        if self.in_cluster:
            k8s_config.load_incluster_config()
        elif self.kubeconfig:
            k8s_config.load_kube_config(config_file=self.kubeconfig)
        else:
            k8s_config.load_kube_config()
        self._client = k8s_client
        return self._client

    # ---------------------------------------------------------------- provider

    def deploy(
        self,
        request: WorkloadRequest,
        reservation: Reservation,
        precision: Optional[PrecisionProfileRef] = None,
    ) -> DeploymentHandle:
        deployment_id = f"wl-{request.workload_id}"
        replicagroup = build_replicagroup(
            request,
            deployment_id,
            namespace=self.namespace,
            precision=precision,
        )
        model_deployment = build_model_deployment(
            request, deployment_id, namespace=self.namespace, image=self.image
        )
        try:
            client = self._ensure_client()
        except IntegrationUnavailable:
            raise
        except Exception as exc:  # pragma: no cover
            raise IntegrationError(f"failed to initialise Kubernetes client: {exc}") from exc

        custom = client.CustomObjectsApi()
        apps = client.AppsV1Api()
        try:
            apps.create_namespaced_deployment(self.namespace, model_deployment)
        except Exception as exc:  # already exists -> idempotent reuse
            if not _is_conflict(exc):
                raise IntegrationError(f"Ignite deployment create failed: {exc}") from exc
        try:
            custom.create_namespaced_custom_object(
                group="kubbernetd.io",
                version="v1",
                namespace=self.namespace,
                plural="replicagroups",
                body=replicagroup,
            )
        except Exception as exc:
            if not _is_conflict(exc):
                raise IntegrationError(f"Ignite ReplicaGroup create failed: {exc}") from exc

        return DeploymentHandle(
            deployment_id=dns_name(deployment_id),
            endpoint=self._endpoint(request.workload_id),
            replicas=request.placement.replicas or 1,
            namespace=self.namespace,
            precision_profile_ref=precision.profile_ref if precision else None,
        )

    def await_ready(self, deployment_id: str, timeout_seconds: float) -> bool:
        import time

        deadline = time.monotonic() + timeout_seconds
        client = self._ensure_client()
        custom = client.CustomObjectsApi()
        while time.monotonic() < deadline:
            try:
                obj = custom.get_namespaced_custom_object(
                    group="kubbernetd.io",
                    version="v1",
                    namespace=self.namespace,
                    plural="replicagroups",
                    name=dns_name(deployment_id),
                )
            except Exception as exc:
                if _is_not_found(exc):
                    raise IntegrationError(
                        f"ReplicaGroup {deployment_id} disappeared while waiting for readiness"
                    ) from exc
                time.sleep(self.poll_interval)
                continue
            status = obj.get("status", {}) or {}
            phase = str(status.get("phase", "")).upper()
            ready = int(status.get("readyReplicas", 0) or 0)
            if phase in ("RUNNING", "WARMING") and ready > 0:
                return True
            if phase in ("ERROR",):
                return False
            time.sleep(self.poll_interval)
        return False

    def release(self, deployment_id: str) -> None:
        client = self._ensure_client()
        custom = client.CustomObjectsApi()
        apps = client.AppsV1Api()
        try:
            custom.delete_namespaced_custom_object(
                group="kubbernetd.io",
                version="v1",
                namespace=self.namespace,
                plural="replicagroups",
                name=dns_name(deployment_id),
            )
        except Exception as exc:
            if not _is_not_found(exc):
                raise IntegrationError(f"Ignite ReplicaGroup delete failed: {exc}") from exc
        try:
            apps.delete_namespaced_deployment(dns_name(deployment_id), self.namespace)
        except Exception as exc:
            if not _is_not_found(exc):
                raise IntegrationError(f"Ignite Deployment delete failed: {exc}") from exc

    def _endpoint(self, workload_id: str) -> str:
        return f"/v1/workloads/{workload_id}"


def _is_conflict(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    return status == 409 or "AlreadyExists" in str(exc)


def _is_not_found(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    return status == 404 or "NotFound" in str(exc)
