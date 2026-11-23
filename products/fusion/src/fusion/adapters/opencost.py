from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
import urllib.request
from urllib.parse import quote


class OpenCostAdapter:
    def __init__(self, base_url: str = "http://localhost:9001", timeout: int = 30):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def get_allocation(self, window: str = "1h") -> List[Dict[str, Any]]:
        url = f"{self._base_url}/model/allocation?window={quote(window)}"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
            return data.get("data", []) if isinstance(data, dict) else data
        except Exception as e:
            raise RuntimeError(f"OpenCost allocation error: {e}")

    def get_cluster_costs(self, window: str = "24h") -> List[Dict[str, Any]]:
        return self.get_allocation(window)

    def get_gpu_costs(self, window: str = "1h") -> List[Dict[str, Any]]:
        allocations = self.get_allocation(window)
        gpu_costs = []
        for entry in allocations:
            if isinstance(entry, dict):
                props = entry.get("properties", {})
                if "gpu" in str(props.get("service", "")).lower() or "gpu" in str(props.get("labels", {})):
                    gpu_costs.append(entry)
        return gpu_costs
