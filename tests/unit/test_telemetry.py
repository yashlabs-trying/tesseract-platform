from datetime import datetime, timedelta, timezone

import pytest

from tesseract_platform.models import Evidence, TelemetryEvent
from tesseract_platform.observability import TelemetryCollector
from tesseract_platform.state import StateStore


def _event(collector, store, *, eid, ts, ttft, latency, success, tokens=10):
    collector.record(
        TelemetryEvent(
            event_id=eid,
            correlation_id="corr-1",
            workload_id="wl-1",
            event_type="request",
            timestamp=ts,
            ttft_ms=ttft,
            request_latency_ms=latency,
            tokens_output=tokens,
            success=success,
            evidence=Evidence.measured,
            source="test",
        )
    )


def test_failed_requests_do_not_inflate_throughput():
    store = StateStore("sqlite:///:memory:")
    collector = TelemetryCollector(store)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _event(collector, store, eid="1", ts=t0, ttft=100, latency=1000, success=True)
    _event(collector, store, eid="2", ts=t0 + timedelta(seconds=5), ttft=200, latency=2000, success=True)
    _event(collector, store, eid="3", ts=t0 + timedelta(seconds=10), ttft=300, latency=3000, success=True)
    # Failed requests with tiny latency must not count.
    _event(collector, store, eid="4", ts=t0 + timedelta(seconds=1), ttft=1, latency=1, success=False)
    _event(collector, store, eid="5", ts=t0 + timedelta(seconds=2), ttft=1, latency=1, success=False)

    summary = collector.aggregate("wl-1")
    assert summary.request_count == 5
    assert summary.success_count == 3
    assert summary.failure_count == 2
    assert summary.error_rate == pytest.approx(0.4)
    # 3 successful requests over a 10s window.
    assert summary.throughput_rps == pytest.approx(0.3)
    assert summary.ttft_p50_ms == pytest.approx(200.0)
    assert summary.tokens_output_total == 30
    store.close()


def test_ingest_rejects_unknown_event_type():
    store = StateStore("sqlite:///:memory:")
    collector = TelemetryCollector(store)
    with pytest.raises(ValueError):
        collector.ingest(
            {
                "schema_version": "tesseract.platform.telemetry.v1",
                "event_id": "x",
                "correlation_id": "c",
                "workload_id": "wl-1",
                "event_type": "not_a_real_event",
                "timestamp": "2026-01-01T00:00:00Z",
                "evidence": "measured",
                "source": "test",
            }
        )
    store.close()
