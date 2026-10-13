import pytest

from tesseract_platform.models import (
    Evidence,
    Operation,
    Reservation,
    ReservationState,
    TelemetryEvent,
)
from tesseract_platform.state import StaleRevision, StateStore
from tesseract_platform.testing import make_request


def _operation(op_id="op-1", key="idem-key-0001"):
    return Operation(
        operation_id=op_id,
        idempotency_key=key,
        workload_id="wl-1",
        correlation_id="corr-1",
        request=make_request(idempotency_key=key),
    )


def test_create_operation_is_idempotent():
    store = StateStore("sqlite:///:memory:")
    first = store.create_operation(_operation("op-1", "same-key-0001"))
    second = store.create_operation(_operation("op-2", "same-key-0001"))
    assert first.operation_id == second.operation_id == "op-1"
    store.close()


def test_reservation_optimistic_concurrency():
    store = StateStore("sqlite:///:memory:")
    res = Reservation(
        reservation_id="res-1",
        operation_id="op-1",
        idempotency_key="idem-key-0001",
        workload_id="wl-1",
        state=ReservationState.RESERVING,
        revision=0,
    )
    store.save_reservation(res)
    res.state = ReservationState.RESERVED
    res.revision = 1
    store.save_reservation(res, expected_revision=0)
    res.revision = 2
    with pytest.raises(StaleRevision):
        store.save_reservation(res, expected_revision=0)
    store.close()


def test_lifecycle_and_telemetry_roundtrip():
    store = StateStore("sqlite:///:memory:")
    event = TelemetryEvent(
        event_id="e1",
        correlation_id="corr-1",
        workload_id="wl-1",
        event_type="request",
        ttft_ms=100.0,
        success=True,
        evidence=Evidence.measured,
        source="test",
    )
    store.record_telemetry(event)
    events = store.list_telemetry("wl-1")
    assert len(events) == 1
    assert events[0].ttft_ms == 100.0
    store.close()
