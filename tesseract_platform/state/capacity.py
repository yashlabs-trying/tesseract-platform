"""Capacity accounting ledger.

Enforces the platform invariant, per resource pool::

    committed + active_reserved <= total_usable_capacity

* ``active_reserved`` — capacity held by in-flight reservations (PENDING..
  RESERVED/DEPLOYING).
* ``committed`` — capacity held by ACTIVE deployments.

Units are a single abstract "capacity unit" (e.g. GPU slots). Callers must use
the same unit consistently; the platform uses ``Capacity.gpu_count``.

The ledger is process-local and thread-safe. For multi-replica control planes a
shared datastore must back it (documented in docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import threading
from typing import Optional


class CapacityExhausted(RuntimeError):
    """Raised when a reservation would exceed total usable capacity."""


class CapacityLedger:
    def __init__(self, total_usable_capacity: int) -> None:
        if total_usable_capacity < 0:
            raise ValueError("total_usable_capacity must be >= 0")
        self._total = total_usable_capacity
        self._active: dict[str, int] = {}
        self._committed: dict[str, int] = {}
        self._lock = threading.RLock()

    @property
    def total_usable_capacity(self) -> int:
        return self._total

    @property
    def active_reserved(self) -> int:
        with self._lock:
            return sum(self._active.values())

    @property
    def committed(self) -> int:
        with self._lock:
            return sum(self._committed.values())

    @property
    def used(self) -> int:
        return self.active_reserved + self.committed

    @property
    def available(self) -> int:
        return self._total - self.used

    def invariant_ok(self) -> bool:
        return self.used <= self._total

    def reserve(self, owner_id: str, count: int) -> None:
        """Hold ``count`` units for ``owner_id`` (idempotent per owner)."""
        if count < 0:
            raise ValueError("count must be >= 0")
        with self._lock:
            if owner_id in self._active or owner_id in self._committed:
                return  # idempotent: already accounted
            if self.used + count > self._total:
                raise CapacityExhausted(
                    f"insufficient capacity for {owner_id!r}: "
                    f"used={self.used} + requested={count} > total={self._total}"
                )
            self._active[owner_id] = count

    def commit(self, owner_id: str) -> None:
        """Move an active reservation to committed (deployment became ACTIVE)."""
        with self._lock:
            count = self._active.pop(owner_id, None)
            if count is None:
                if owner_id in self._committed:
                    return
                raise KeyError(f"no active reservation for {owner_id!r}")
            self._committed[owner_id] = count

    def release(self, owner_id: str) -> None:
        """Release capacity held by ``owner_id`` (idempotent)."""
        with self._lock:
            self._active.pop(owner_id, None)
            self._committed.pop(owner_id, None)

    def owner_count(self, owner_id: str) -> Optional[int]:
        with self._lock:
            if owner_id in self._active:
                return self._active[owner_id]
            return self._committed.get(owner_id)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "total_usable_capacity": self._total,
                "active_reserved": sum(self._active.values()),
                "committed": sum(self._committed.values()),
                "available": self._total - sum(self._active.values()) - sum(self._committed.values()),
                "active_owners": sorted(self._active),
                "committed_owners": sorted(self._committed),
                "invariant_ok": (
                    sum(self._active.values()) + sum(self._committed.values()) <= self._total
                ),
            }
