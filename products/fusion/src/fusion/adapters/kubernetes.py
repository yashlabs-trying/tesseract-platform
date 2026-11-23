from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from fusion.models.types import GPUInventory, HealthStatus, IsolationMode, SharingPolicy
from fusion._time import utcnow_naive


class KubernetesInventoryAdapter:
    def __init__(self, kube_config: Optional[str] = None, in_cluster: bool = False):
        self._kube_config = kube_config
        self._in_cluster = in_cluster
        self._initialized = False
        self._v1 = None

    def _init_client(self):
        if self._initialized:
            return
        try:
            import kubernetes
            if self._in_cluster:
                kubernetes.config.load_incluster_config()
            elif self._kube_config:
                kubernetes.config.load_kube_config(self._kube_config)
            else:
                kubernetes.config.load_kube_config()
            self._v1 = kubernetes.client.CoreV1Api()
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kubernetes client: {e}")

    def get_inventory(self) -> List[GPUInventory]:
        self._init_client()
        inventory = []
        try:
            nodes = self._v1.list_node()
            for node in nodes.items:
                gpu_count = self._extract_gpu_count(node)
                gpu_model = self._extract_gpu_model(node)
                node_name = node.metadata.name

                for i in range(gpu_count):
                    inventory.append(GPUInventory(
                        gpu_id=f"{node_name}-gpu-{i}",
                        cluster_id="kubernetes",
                        node_id=node_name,
                        vendor="nvidia",
                        gpu_model=gpu_model or "unknown",
                        memory_capacity_mb=self._extract_gpu_memory(node),
                        available_resource_slots=1,
                        health_status=self._parse_health(node),
                        resource_cost_per_hour=Decimal("0"),
                        resource_cost_source="kubernetes",
                        observed_at=utcnow_naive(),
                    ))
        except Exception as e:
            raise RuntimeError(f"Failed to list Kubernetes nodes: {e}")
        return inventory

    def _extract_gpu_count(self, node) -> int:
        try:
            capacity = node.status.capacity or {}
            return int(capacity.get("nvidia.com/gpu", 0))
        except (AttributeError, ValueError):
            return 0

    def _extract_gpu_model(self, node) -> Optional[str]:
        labels = node.metadata.labels or {}
        for key in ["gpu-type", "nvidia.com/gpu.model", "gpu.model", "accelerator"]:
            if key in labels:
                return labels[key]
        return None

    def _extract_gpu_memory(self, node) -> int:
        labels = node.metadata.labels or {}
        try:
            return int(labels.get("nvidia.com/gpu.memory", "0"))
        except ValueError:
            return 0

    def _parse_health(self, node) -> HealthStatus:
        for condition in node.status.conditions or []:
            if condition.type == "Ready":
                return HealthStatus.HEALTHY if condition.status == "True" else HealthStatus.UNHEALTHY
        return HealthStatus.UNKNOWN
