"""Telemetry ingestion and aggregation.

Integrity rules enforced here:

* failed/incomplete requests (``success is False``) never contribute to
  successful throughput or latency percentiles;
* every event must carry a ``correlation_id`` preserved across the lifecycle;
* events are validated against the ``tesseract.platform.telemetry.v1`` contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from pydantic import BaseModel

from tesseract_platform.contracts import validate_or_raise
from tesseract_platform.models import TelemetryEvent
from tesseract_platform.state import StateStore


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile (nearest-rank fallback for small samples)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


class TelemetrySummary(BaseModel):
    workload_id: str
    total_events: int = 0
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    itl_p50_ms: Optional[float] = None
    itl_p95_ms: Optional[float] = None
    request_latency_p50_ms: Optional[float] = None
    request_latency_p95_ms: Optional[float] = None
    request_latency_p99_ms: Optional[float] = None
    cold_start_p95_ms: Optional[float] = None
    readiness_p95_ms: Optional[float] = None
    replica_count_max: int = 0
    tokens_output_total: int = 0
    first_event: Optional[datetime] = None
    last_event: Optional[datetime] = None


class TelemetryCollector:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def record(self, event: TelemetryEvent) -> TelemetryEvent:
        validate_or_raise("telemetry", event.model_dump(mode="json", exclude_none=True))
        self.store.record_telemetry(event)
        return event

    def ingest(self, payload: dict) -> TelemetryEvent:
        validate_or_raise("telemetry", payload)
        event = TelemetryEvent.model_validate(payload)
        self.store.record_telemetry(event)
        return event

    def aggregate(self, workload_id: str) -> TelemetrySummary:
        events = self.store.list_telemetry(workload_id)
        summary = TelemetrySummary(workload_id=workload_id, total_events=len(events))
        if not events:
            return summary

        requests = [e for e in events if e.event_type == "request"]
        successful = [e for e in requests if e.success is not False]
        failed = [e for e in requests if e.success is False]

        summary.request_count = len(requests)
        summary.success_count = len(successful)
        summary.failure_count = len(failed)
        summary.error_rate = (len(failed) / len(requests)) if requests else 0.0

        def collect(field: str, pool: Sequence[TelemetryEvent]) -> list[float]:
            out = []
            for event in pool:
                value = getattr(event, field)
                if value is not None:
                    out.append(float(value))
            return out

        # Latency percentiles use successful requests only.
        summary.ttft_p50_ms = percentile(collect("ttft_ms", successful), 50)
        summary.ttft_p95_ms = percentile(collect("ttft_ms", successful), 95)
        summary.ttft_p99_ms = percentile(collect("ttft_ms", successful), 99)
        summary.itl_p50_ms = percentile(collect("itl_ms", successful), 50)
        summary.itl_p95_ms = percentile(collect("itl_ms", successful), 95)
        summary.request_latency_p50_ms = percentile(collect("request_latency_ms", successful), 50)
        summary.request_latency_p95_ms = percentile(collect("request_latency_ms", successful), 95)
        summary.request_latency_p99_ms = percentile(collect("request_latency_ms", successful), 99)

        # Lifecycle percentiles.
        summary.cold_start_p95_ms = percentile(
            collect("cold_start_ms", [e for e in events if e.event_type == "cold_start"]), 95
        )
        summary.readiness_p95_ms = percentile(
            collect("readiness_ms", [e for e in events if e.event_type == "readiness"]), 95
        )

        timestamps = [e.timestamp for e in events if e.timestamp]
        if timestamps:
            summary.first_event = min(timestamps)
            summary.last_event = max(timestamps)
            duration = (summary.last_event - summary.first_event).total_seconds()
            if duration > 0:
                summary.throughput_rps = len(successful) / duration

        summary.replica_count_max = max(
            (e.replica_count for e in events if e.replica_count is not None), default=0
        )
        summary.tokens_output_total = sum(
            (e.tokens_output or 0) for e in successful
        )
        return summary
