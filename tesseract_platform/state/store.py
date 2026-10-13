"""Durable state store backed by SQLite.

The store is deliberately small and dependency-free (stdlib ``sqlite3``). It
provides:

* idempotent operation creation keyed by ``idempotency_key``;
* optimistic concurrency (revision) on reservations;
* an append-only lifecycle/audit log;
* telemetry persistence.

It is safe for the single-process control plane; a real deployment can swap in
Postgres by implementing the same surface.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, Sequence

from tesseract_platform.models import (
    DeploymentRecord,
    LifecycleTransition,
    Operation,
    Reservation,
    TelemetryEvent,
)


class DuplicateOperation(RuntimeError):
    """Raised when an operation with the same idempotency key already exists."""


class StaleRevision(RuntimeError):
    """Raised when an optimistic-concurrency update sees an outdated revision."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, dsn: str = "sqlite:///./var/platform.db"):
        self._lock = threading.RLock()
        self._path = self._resolve_path(dsn)
        if self._path not in (":memory:", ""):
            parent = os.path.dirname(os.path.abspath(self._path))
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    @staticmethod
    def _resolve_path(dsn: str) -> str:
        if dsn.startswith("sqlite:///"):
            return dsn[len("sqlite:///") :]
        if dsn.startswith("sqlite://"):
            return dsn[len("sqlite://") :]
        return dsn

    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            finally:
                cur.close()

    def _init_schema(self) -> None:
        with self._cursor() as cur:
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id    TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    workload_id     TEXT NOT NULL,
                    state           TEXT NOT NULL,
                    payload         TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reservations (
                    reservation_id TEXT PRIMARY KEY,
                    operation_id   TEXT NOT NULL,
                    workload_id    TEXT NOT NULL,
                    state          TEXT NOT NULL,
                    revision       INTEGER NOT NULL,
                    expires_at     TEXT,
                    payload        TEXT NOT NULL,
                    created_at     TEXT NOT NULL,
                    updated_at     TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id  TEXT PRIMARY KEY,
                    operation_id   TEXT NOT NULL,
                    reservation_id TEXT,
                    workload_id    TEXT NOT NULL,
                    state          TEXT NOT NULL,
                    payload        TEXT NOT NULL,
                    created_at     TEXT NOT NULL,
                    updated_at     TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lifecycle (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type    TEXT NOT NULL,
                    entity_id      TEXT NOT NULL,
                    operation_id   TEXT NOT NULL,
                    from_state     TEXT,
                    to_state       TEXT NOT NULL,
                    reason         TEXT,
                    actor          TEXT,
                    correlation_id TEXT,
                    timestamp      TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id   TEXT,
                    event          TEXT NOT NULL,
                    detail         TEXT,
                    timestamp      TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS telemetry (
                    event_id       TEXT PRIMARY KEY,
                    workload_id    TEXT NOT NULL,
                    correlation_id TEXT,
                    payload        TEXT NOT NULL,
                    timestamp      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_res_state ON reservations(state);
                CREATE INDEX IF NOT EXISTS idx_dep_state ON deployments(state);
                CREATE INDEX IF NOT EXISTS idx_tel_workload ON telemetry(workload_id);
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ operations

    def create_operation(self, op: Operation) -> Operation:
        """Insert an operation; if the idempotency key exists, return the existing one."""
        existing = self.get_operation_by_idempotency(op.idempotency_key)
        if existing is not None:
            return existing
        with self._cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO operations "
                    "(operation_id, idempotency_key, workload_id, state, payload, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        op.operation_id,
                        op.idempotency_key,
                        op.workload_id,
                        op.state.value,
                        op.model_dump_json(),
                        op.created_at.isoformat(),
                        op.updated_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                pass
        return self.get_operation_by_idempotency(op.idempotency_key) or op

    def get_operation(self, operation_id: str) -> Optional[Operation]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
        return Operation.model_validate_json(row["payload"]) if row else None

    def get_operation_by_idempotency(self, key: str) -> Optional[Operation]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM operations WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return Operation.model_validate_json(row["payload"]) if row else None

    def save_operation(self, op: Operation) -> None:
        op.updated_at = datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE operations SET state = ?, payload = ?, updated_at = ? WHERE operation_id = ?",
                (op.state.value, op.model_dump_json(), op.updated_at.isoformat(), op.operation_id),
            )

    # ------------------------------------------------------------------ reservations

    def save_reservation(self, res: Reservation, expected_revision: Optional[int] = None) -> None:
        res.updated_at = datetime.now(timezone.utc)
        with self._cursor() as cur:
            if expected_revision is None:
                cur.execute(
                    "INSERT OR REPLACE INTO reservations "
                    "(reservation_id, operation_id, workload_id, state, revision, expires_at, payload, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        res.reservation_id,
                        res.operation_id,
                        res.workload_id,
                        res.state.value,
                        res.revision,
                        res.expires_at.isoformat() if res.expires_at else None,
                        res.model_dump_json(),
                        res.created_at.isoformat(),
                        res.updated_at.isoformat(),
                    ),
                )
            else:
                cur.execute(
                    "UPDATE reservations SET state = ?, revision = ?, expires_at = ?, payload = ?, updated_at = ? "
                    "WHERE reservation_id = ? AND revision = ?",
                    (
                        res.state.value,
                        res.revision,
                        res.expires_at.isoformat() if res.expires_at else None,
                        res.model_dump_json(),
                        res.updated_at.isoformat(),
                        res.reservation_id,
                        expected_revision,
                    ),
                )
                if cur.rowcount == 0:
                    raise StaleRevision(
                        f"reservation {res.reservation_id} revision mismatch "
                        f"(expected {expected_revision})"
                    )

    def get_reservation(self, reservation_id: str) -> Optional[Reservation]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
        return Reservation.model_validate_json(row["payload"]) if row else None

    def list_reservations_by_state(self, states: Sequence[str]) -> list[Reservation]:
        if not states:
            return []
        placeholders = ",".join("?" for _ in states)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT payload FROM reservations WHERE state IN ({placeholders})",
                tuple(states),
            ).fetchall()
        return [Reservation.model_validate_json(r["payload"]) for r in rows]

    # ------------------------------------------------------------------ deployments

    def save_deployment(self, dep: DeploymentRecord, touch: bool = True) -> None:
        if touch:
            dep.updated_at = datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO deployments "
                "(deployment_id, operation_id, reservation_id, workload_id, state, payload, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    dep.deployment_id,
                    dep.operation_id,
                    dep.reservation_id,
                    dep.workload_id,
                    dep.state.value,
                    dep.model_dump_json(),
                    dep.created_at.isoformat(),
                    dep.updated_at.isoformat(),
                ),
            )

    def get_deployment(self, deployment_id: str) -> Optional[DeploymentRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM deployments WHERE deployment_id = ?", (deployment_id,)
            ).fetchone()
        return DeploymentRecord.model_validate_json(row["payload"]) if row else None

    def list_deployments_by_state(self, states: Sequence[str]) -> list[DeploymentRecord]:
        if not states:
            return []
        placeholders = ",".join("?" for _ in states)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT payload FROM deployments WHERE state IN ({placeholders})",
                tuple(states),
            ).fetchall()
        return [DeploymentRecord.model_validate_json(r["payload"]) for r in rows]

    # ------------------------------------------------------------------ lifecycle / audit

    def record_transition(self, t: LifecycleTransition) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO lifecycle "
                "(entity_type, entity_id, operation_id, from_state, to_state, reason, actor, correlation_id, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    t.entity_type,
                    t.entity_id,
                    t.operation_id,
                    t.from_state,
                    t.to_state,
                    t.reason,
                    t.actor,
                    t.correlation_id,
                    t.timestamp.isoformat(),
                ),
            )

    def list_transitions(self, operation_id: str) -> list[LifecycleTransition]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM lifecycle WHERE operation_id = ? ORDER BY id", (operation_id,)
            ).fetchall()
        return [
            LifecycleTransition(
                entity_type=r["entity_type"],
                entity_id=r["entity_id"],
                operation_id=r["operation_id"],
                from_state=r["from_state"],
                to_state=r["to_state"],
                reason=r["reason"] or "",
                actor=r["actor"] or "coordinator",
                correlation_id=r["correlation_id"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
            )
            for r in rows
        ]

    def record_audit(self, operation_id: Optional[str], event: str, detail: Optional[dict] = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO audit (operation_id, event, detail, timestamp) VALUES (?, ?, ?, ?)",
                (operation_id, event, json.dumps(detail or {}), _now()),
            )

    def list_audit(self, operation_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit WHERE operation_id = ? ORDER BY id", (operation_id,)
            ).fetchall()
        return [
            {
                "event": r["event"],
                "detail": json.loads(r["detail"]) if r["detail"] else {},
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ telemetry

    def record_telemetry(self, event: TelemetryEvent) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO telemetry (event_id, workload_id, correlation_id, payload, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.workload_id,
                    event.correlation_id,
                    event.model_dump_json(),
                    event.timestamp.isoformat(),
                ),
            )

    def list_telemetry(self, workload_id: str) -> list[TelemetryEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM telemetry WHERE workload_id = ? ORDER BY timestamp",
                (workload_id,),
            ).fetchall()
        return [TelemetryEvent.model_validate_json(r["payload"]) for r in rows]
