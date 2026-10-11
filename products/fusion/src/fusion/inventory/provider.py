from __future__ import annotations
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Protocol, Any
from fusion.models.types import GPUInventory, HealthStatus, IsolationMode, SharingPolicy, Currency


class InventoryProvider(Protocol):
    def get_inventory(self) -> List[GPUInventory]:
        ...


class MetricsProvider(Protocol):
    def get_metrics(self, query: str, time_range: Optional[timedelta] = None) -> List[Dict[str, Any]]:
        ...


class CostProvider(Protocol):
    def get_costs(self, window: Optional[timedelta] = None) -> List[Dict[str, Any]]:
        ...


class InventoryService:
    def __init__(self, provider: InventoryProvider, stale_threshold_seconds: int = 600):
        self._provider = provider
        self._cache: List[GPUInventory] = []
        self._cache_time: Optional[datetime] = None
        self._stale_threshold = stale_threshold_seconds

    def get_inventory(self, force_refresh: bool = False) -> List[GPUInventory]:
        now = datetime.utcnow()
        if not force_refresh and self._cache_time is not None:
            age = (now - self._cache_time).total_seconds()
            if age < self._stale_threshold:
                return self._cache
        inventory = self._provider.get_inventory()
        self._cache = inventory
        self._cache_time = now
        return inventory

    def get_available_gpus(self, force_refresh: bool = False) -> List[GPUInventory]:
        return [g for g in self.get_inventory(force_refresh) if g.health_status == HealthStatus.HEALTHY]

    def get_gpu_by_id(self, gpu_id: str) -> Optional[GPUInventory]:
        inv = self.get_inventory()
        for g in inv:
            if g.gpu_id == gpu_id:
                return g
        return None

    def get_gpus_by_model(self, model: str) -> List[GPUInventory]:
        return [g for g in self.get_inventory() if g.gpu_model == model]

    def get_total_capacity_by_model(self) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for g in self.get_inventory():
            if g.health_status == HealthStatus.HEALTHY:
                result[g.gpu_model] = result.get(g.gpu_model, 0) + g.available_resource_slots
        return result

    def invalidate_cache(self):
        self._cache_time = None

    @property
    def cache_age_seconds(self) -> Optional[float]:
        if self._cache_time is None:
            return None
        return (datetime.utcnow() - self._cache_time).total_seconds()
