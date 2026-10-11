from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
from fusion.models.types import GPUInventory, TenantPolicy


class KAIAdapter:
    def __init__(self, base_url: Optional[str] = None):
        self._base_url = base_url

    def get_placement_options(self, workload: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    def get_queue_state(self) -> List[Dict[str, Any]]:
        return []

    def get_quotas(self) -> List[Dict[str, Any]]:
        return []

    def check_placement_feasibility(self, gpu: GPUInventory, policy: TenantPolicy) -> bool:
        return True

    def get_cluster_capacity(self) -> Dict[str, Any]:
        return {"total_gpus": 0, "allocated_gpus": 0, "available_gpus": 0}
