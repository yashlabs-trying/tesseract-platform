from __future__ import annotations
import json
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal
from fusion.models.types import GPUInventory, HealthStatus, IsolationMode, SharingPolicy, Currency


class FixtureInventoryAdapter:
    def __init__(self, fixture_path: Optional[str] = None):
        self._fixture_path = fixture_path
        self._inventory: List[GPUInventory] = []

    def load_from_file(self, path: str) -> List[GPUInventory]:
        p = Path(path)
        if p.suffix in (".yaml", ".yml"):
            with open(p) as f:
                data = yaml.safe_load(f)
        elif p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {p.suffix}")

        inventory = []
        for entry in data.get("gpus", data.get("inventory", [])):
            inventory.append(self._parse_entry(entry))
        self._inventory = inventory
        return inventory

    def load_from_dict(self, data: List[Dict[str, Any]]) -> List[GPUInventory]:
        inventory = [self._parse_entry(entry) for entry in data]
        self._inventory = inventory
        return inventory

    def get_inventory(self) -> List[GPUInventory]:
        if not self._inventory and self._fixture_path:
            return self.load_from_file(self._fixture_path)
        return self._inventory

    def _parse_entry(self, entry: Dict[str, Any]) -> GPUInventory:
        return GPUInventory(
            gpu_id=entry.get("gpu_id", entry.get("id", "unknown")),
            cluster_id=entry.get("cluster_id", "default"),
            node_id=entry.get("node_id", "node-0"),
            vendor=entry.get("vendor", "nvidia"),
            gpu_model=entry.get("gpu_model", entry.get("model", "unknown")),
            memory_capacity_mb=entry.get("memory_capacity_mb", entry.get("memory_mb", 0)),
            available_memory_mb=entry.get("available_memory_mb"),
            available_resource_slots=entry.get("available_resource_slots", 1),
            interconnect=entry.get("interconnect"),
            topology=entry.get("topology"),
            supported_isolation_modes=[
                IsolationMode(m) for m in entry.get("supported_isolation_modes", ["exclusive"])
            ],
            health_status=HealthStatus(entry.get("health_status", "unknown")),
            sharing_policy=SharingPolicy(entry.get("sharing_policy", "exclusive_only")),
            location_zone=entry.get("location_zone"),
            resource_cost_per_hour=Decimal(str(entry.get("resource_cost_per_hour", entry.get("cost_per_hour", 0)))),
            currency=Currency(entry.get("currency", "USD")),
            resource_cost_source=entry.get("resource_cost_source", "fixture"),
            observed_at=datetime.fromisoformat(entry["observed_at"]) if "observed_at" in entry else datetime.utcnow(),
            refresh_timestamp=datetime.utcnow(),
        )
