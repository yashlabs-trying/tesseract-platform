from datetime import datetime, timedelta, timezone

from tesseract_platform.config import ControllerMode
from tesseract_platform.models import Evidence, SLOSpec, TelemetryEvent
from tesseract_platform.observability import OptimizationController, TelemetryCollector
from tesseract_platform.state import StateStore


def _seed(collector, ttft=500.0, count=5):
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(count):
        collector.record(
            TelemetryEvent(
                event_id=f"e{i}",
                correlation_id="c",
                workload_id="wl-1",
                event_type="request",
                timestamp=t0 + timedelta(seconds=i),
                ttft_ms=ttft,
                request_latency_ms=ttft * 5,
                success=True,
                evidence=Evidence.measured,
                source="test",
            )
        )


def test_observe_only_never_acts():
    store = StateStore("sqlite:///:memory:")
    collector = TelemetryCollector(store)
    _seed(collector, ttft=500.0)
    controller = OptimizationController(store, collector, mode=ControllerMode.OBSERVE_ONLY)
    decision = controller.evaluate("wl-1", SLOSpec(ttft_p95_ms=100.0))
    assert decision.action == "none"
    assert decision.reason == "ttft_p95_exceeds_objective"
    store.close()


def test_recommendation_mode_recommends():
    store = StateStore("sqlite:///:memory:")
    collector = TelemetryCollector(store)
    _seed(collector, ttft=500.0)
    controller = OptimizationController(store, collector, mode=ControllerMode.RECOMMENDATION)
    decision = controller.evaluate("wl-1", SLOSpec(ttft_p95_ms=100.0))
    assert decision.action == "recommend_reconfigure"
    store.close()


def test_within_hysteresis_no_action():
    store = StateStore("sqlite:///:memory:")
    collector = TelemetryCollector(store)
    _seed(collector, ttft=110.0)
    controller = OptimizationController(store, collector, mode=ControllerMode.RECOMMENDATION)
    decision = controller.evaluate("wl-1", SLOSpec(ttft_p95_ms=100.0))
    assert decision.action == "none"
    assert decision.reason == "within_hysteresis"
    store.close()


def test_controlled_action_cooldown_and_budget():
    store = StateStore("sqlite:///:memory:")
    collector = TelemetryCollector(store)
    _seed(collector, ttft=500.0)
    controller = OptimizationController(
        store, collector, mode=ControllerMode.CONTROLLED_ACTION, cooldown_seconds=120, action_budget=1
    )
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = controller.evaluate("wl-1", SLOSpec(ttft_p95_ms=100.0), now=t0)
    assert first.action == "reconfigure"
    assert first.budget_remaining == 0

    second = controller.evaluate("wl-1", SLOSpec(ttft_p95_ms=100.0), now=t0 + timedelta(seconds=10))
    assert second.action == "recommend_reconfigure"
    assert second.cooldown_active is True

    third = controller.evaluate(
        "wl-1", SLOSpec(ttft_p95_ms=100.0), now=t0 + timedelta(seconds=500)
    )
    assert third.action == "recommend_reconfigure"
    assert third.reason == "action_budget_exhausted"
    store.close()
