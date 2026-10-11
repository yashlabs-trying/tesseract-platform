from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime


class DynamoMetricsAdapter:
    def __init__(self, base_url: Optional[str] = None):
        self._base_url = base_url

    def get_planner_decisions(self) -> List[Dict[str, Any]]:
        return []

    def get_scaling_events(self) -> List[Dict[str, Any]]:
        return []

    def get_forward_pass_metrics(self) -> List[Dict[str, Any]]:
        return []

    def normalize_metrics(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for entry in raw:
            normalized.append({
                "timestamp": entry.get("timestamp", datetime.utcnow().isoformat()),
                "metric": entry.get("name", "unknown"),
                "value": entry.get("value", 0),
                "labels": entry.get("labels", {}),
            })
        return normalized
