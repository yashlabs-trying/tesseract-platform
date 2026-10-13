"""Phases D, F, G, H — CPU harness.

D: reservation/resource accounting invariants (randomized concurrency + faults).
F: concurrency levels + short soak + leak check.
G: chaos/failure recovery matrix.
H: telemetry integrity.

All use the real platform coordinator/store/ledger/telemetry/reconciler with
in-process product fakes. Outputs raw JSON evidence.
"""

from __future__ import annotations

import json
import random
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from integrations.adaptive import DisabledAdaptiveProvider
from integrations.base import IntegrationError
from tesseract_platform.coordinator import Reconciler, WorkloadCoordinator
from tesseract_platform.models import Evidence, OperationState, TelemetryEvent, utcnow
from tesseract_platform.observability import TelemetryCollector
from tesseract_platform.observability.telemetry import percentile
from tesseract_platform.policies import PlatformPolicy
from tesseract_platform.state import CapacityLedger, StateStore
from tesseract_platform.testing import (
    FakeDeploymentProvider,
    FakePlacementProvider,
    FakeProfileProvider,
    make_request,
)

RAW = Path("/home/yashs/tesseract-platform/reports/raw")
RAW.mkdir(parents=True, exist_ok=True)


def _coordinator(store, *, ledger=None, profile=None, placement=None, deployment=None, telemetry=None, policy=None):
    return WorkloadCoordinator(
        store=store,
        policy=policy or PlatformPolicy(reservation_ttl_seconds=900, readiness_timeout_seconds=30),
        profile_provider=profile or FakeProfileProvider(),
        placement_provider=placement or FakePlacementProvider(),
        deployment_provider=deployment or FakeDeploymentProvider(),
        precision_provider=DisabledAdaptiveProvider(),
        telemetry=telemetry or TelemetryCollector(store),
        capacity_ledger=ledger,
    )


class RandomPlacementProvider(FakePlacementProvider):
    def __init__(self, rng, **kw):
        super().__init__(**kw)
        self.rng = rng

    def reserve(self, request, profile, precision=None):
        self.gpu_count = self.rng.choice([1, 2, 3])
        return super().reserve(request, profile, precision)


# --------------------------------------------------------------------------- Phase D
def phase_d() -> dict:
    rng = random.Random(1234)
    total = 20
    store = StateStore("sqlite:///:memory:")
    ledger = CapacityLedger(total)
    placement = RandomPlacementProvider(rng)
    coordinator = _coordinator(store, ledger=ledger, placement=placement)

    def submit(i):
        return coordinator.submit(
            make_request(workload_id=f"d-{i}", idempotency_key=f"d-idem-{i:08d}")
        )

    with ThreadPoolExecutor(max_workers=32) as pool:
        ops = list(pool.map(submit, range(100)))

    active = [o for o in ops if o.state is OperationState.ACTIVE]
    failed = [o for o in ops if o.state is OperationState.FAILED]
    snap = ledger.snapshot()
    # Every ACTIVE op has a reservation + deployment; no duplicate reservations.
    res_ids = [o.reservation_id for o in active]
    invariants = {
        "no_duplicate_reservations": len(res_ids) == len(set(res_ids)),
        "every_active_has_reservation": all(o.reservation_id for o in active),
        "every_active_has_deployment": all(o.deployment_id for o in active),
        "capacity_invariant": snap["invariant_ok"],
        "used_le_total": snap["active_reserved"] + snap["committed"] <= total,
    }

    # Fault injection: fusion failure, ignite failure, readiness timeout.
    faults = {}
    for name, kwargs in (
        ("fusion_unavailable", {"placement": FakePlacementProvider(fail=True)}),
        ("ignite_unavailable", {"deployment": FakeDeploymentProvider(fail=True)}),
        ("readiness_timeout", {"deployment": FakeDeploymentProvider(ready=False)}),
    ):
        s = StateStore("sqlite:///:memory:")
        led = CapacityLedger(4)
        c = _coordinator(s, ledger=led, **kwargs)
        op = c.submit(make_request(workload_id=f"f-{name}", idempotency_key=f"f-idem-{name}"))
        faults[name] = {
            "state": op.state.value,
            "capacity_released": led.used == 0,
            "invariant_ok": led.invariant_ok(),
        }
        s.close()

    # Repeated release is idempotent.
    s = StateStore("sqlite:///:memory:")
    led = CapacityLedger(4)
    c = _coordinator(s, ledger=led)
    op = c.submit(make_request(workload_id="rel", idempotency_key="rel-idem-0001"))
    c.release(op.operation_id)
    c.release(op.operation_id)
    repeated_release = {"used": led.used, "state": c.get(op.operation_id).state.value}
    s.close()

    # Stale reservation expiration via reconciler.
    s = StateStore("sqlite:///:memory:")
    led = CapacityLedger(4)
    c = _coordinator(s, ledger=led)
    from tesseract_platform.models import Operation, Reservation, ReservationState

    req = make_request(workload_id="stale", idempotency_key="stale-idem-0001")
    s.create_operation(
        Operation(operation_id="op-stale", idempotency_key=req.idempotency_key,
                  workload_id=req.workload_id, correlation_id="c", request=req,
                  state=OperationState.RESERVED, reservation_id="res-stale")
    )
    s.save_reservation(Reservation(reservation_id="res-stale", operation_id="op-stale",
        idempotency_key=req.idempotency_key, workload_id=req.workload_id,
        state=ReservationState.RESERVED, revision=1, expires_at=utcnow() - timedelta(seconds=5)))
    actions = Reconciler(c, s, readiness_timeout_seconds=30).reconcile()
    stale = {"actions": actions, "final_state": s.get_operation("op-stale").state.value}
    s.close()

    store.close()
    return {
        "phase": "D",
        "total_capacity": total,
        "active": len(active),
        "failed": len(failed),
        "ledger": snap,
        "invariants": invariants,
        "fault_injection": faults,
        "repeated_release": repeated_release,
        "stale_reservation": stale,
    }


# --------------------------------------------------------------------------- Phase F
def phase_f() -> dict:
    levels = [1, 10, 25, 100, 250, 500]
    results = []
    for level in levels:
        store = StateStore("sqlite:///:memory:")
        telemetry = TelemetryCollector(store)
        coordinator = _coordinator(store, telemetry=telemetry)

        def submit(i, coordinator=coordinator, level=level):
            start = time.perf_counter()
            op = coordinator.submit(
                make_request(workload_id=f"f{level}-{i}", idempotency_key=f"f{level}-idem-{i:08d}")
            )
            return (op.state is OperationState.ACTIVE, (time.perf_counter() - start) * 1000.0)

        with ThreadPoolExecutor(max_workers=min(level, 64)) as pool:
            outcomes = list(pool.map(submit, range(level)))
        latencies = [ms for ok, ms in outcomes if ok]
        results.append({
            "level": level,
            "success": sum(1 for ok, _ in outcomes if ok),
            "failed": sum(1 for ok, _ in outcomes if not ok),
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "p99_ms": percentile(latencies, 99),
        })
        store.close()

    # Short soak: 100 reservation/release cycles + leak check.
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    store = StateStore("sqlite:///:memory:")
    ledger = CapacityLedger(4)
    coordinator = _coordinator(store, ledger=ledger)
    started = time.perf_counter()
    for i in range(100):
        op = coordinator.submit(make_request(workload_id=f"soak-{i}", idempotency_key=f"soak-idem-{i:08d}"))
        if op.state is OperationState.ACTIVE:
            coordinator.release(op.operation_id)
    soak_seconds = time.perf_counter() - started
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    soak = {
        "cycles": 100,
        "seconds": round(soak_seconds, 3),
        "ledger_used_after": ledger.used,
        "rss_kb_before": rss_before,
        "rss_kb_after": rss_after,
        "rss_delta_kb": rss_after - rss_before,
    }
    store.close()
    return {"phase": "F", "levels": results, "soak_100_cycles": soak,
            "note": "30-minute sustained soak NOT RUN in this session (see report)"}


# --------------------------------------------------------------------------- Phase G
def phase_g() -> dict:
    matrix = {}

    def run(name, *, profile=None, placement=None, deployment=None, expect_failed=True):
        s = StateStore("sqlite:///:memory:")
        led = CapacityLedger(4)
        c = _coordinator(s, ledger=led, profile=profile, placement=placement, deployment=deployment)
        op = c.submit(make_request(workload_id=f"g-{name}", idempotency_key=f"g-idem-{name}"))
        audit = s.list_audit(op.operation_id)
        transitions = s.list_transitions(op.operation_id)
        out = {
            "state": op.state.value,
            "invariant_ok": led.invariant_ok(),
            "capacity_released": led.used == 0,
            "has_error": bool(op.error),
            "audit_events": [a["event"] for a in audit],
            "transition_count": len(transitions),
            "terminal": op.state.value in ("ACTIVE", "FAILED", "RELEASED"),
        }
        s.close()
        return out

    matrix["foundry_unavailable"] = run("foundry", profile=FakeProfileProvider(fail=True))
    matrix["fusion_unavailable"] = run("fusion", placement=FakePlacementProvider(fail=True))
    matrix["ignite_unavailable"] = run("ignite", deployment=FakeDeploymentProvider(fail=True))
    matrix["readiness_timeout"] = run("ready", deployment=FakeDeploymentProvider(ready=False))

    # Adaptive unavailable: platform must still succeed (safe default).
    class _UnavailableAdaptive:
        available = False
        enabled = True
        def recommend_profile(self, request, capacity=None):
            raise IntegrationError("adaptive unavailable")
        def validate_profile(self, profile):
            return False
    s = StateStore("sqlite:///:memory:")
    c = WorkloadCoordinator(
        store=s, policy=PlatformPolicy(), profile_provider=FakeProfileProvider(),
        placement_provider=FakePlacementProvider(), deployment_provider=FakeDeploymentProvider(),
        precision_provider=_UnavailableAdaptive(), telemetry=TelemetryCollector(s),
    )
    op = c.submit(make_request(workload_id="g-adaptive", idempotency_key="g-idem-adaptive"))
    matrix["adaptive_unavailable"] = {"state": op.state.value, "precision_ref": op.precision_profile_ref}
    s.close()

    # Duplicate lifecycle event / out-of-order telemetry handled in Phase H.
    return {"phase": "G", "matrix": matrix}


# --------------------------------------------------------------------------- Phase H
def phase_h() -> dict:
    store = StateStore("sqlite:///:memory:")
    collector = TelemetryCollector(store)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # 3 success + 2 failed requests; failed must not inflate throughput.
    for i, success in enumerate([True, True, True, False, False]):
        collector.record(TelemetryEvent(
            event_id=f"e{i}", correlation_id="corr-1", workload_id="h-1", event_type="request",
            timestamp=t0 + timedelta(seconds=i), ttft_ms=100.0 if success else 1.0,
            request_latency_ms=1000.0 if success else 1.0, success=success,
            tokens_output=10, evidence=Evidence.measured, source="test"))
    summary = collector.aggregate("h-1")
    integrity = {
        "request_count": summary.request_count,
        "success_count": summary.success_count,
        "failure_count": summary.failure_count,
        "error_rate": summary.error_rate,
        "throughput_uses_success_only": abs(summary.throughput_rps - 3 / 4.0) < 0.2,
        "ttft_p50_success_only": summary.ttft_p50_ms,
    }
    # Duplicate event id is idempotent (same primary key).
    collector.record(TelemetryEvent(
        event_id="dup", correlation_id="corr-1", workload_id="h-2", event_type="request",
        timestamp=t0, ttft_ms=50.0, success=True, evidence=Evidence.measured, source="test"))
    collector.record(TelemetryEvent(
        event_id="dup", correlation_id="corr-1", workload_id="h-2", event_type="request",
        timestamp=t0, ttft_ms=50.0, success=True, evidence=Evidence.measured, source="test"))
    integrity["duplicate_event_deduped"] = len(store.list_telemetry("h-2")) == 1
    # Out-of-order timestamps still aggregate.
    collector.record(TelemetryEvent(
        event_id="ooo1", correlation_id="c", workload_id="h-3", event_type="request",
        timestamp=t0 + timedelta(seconds=10), ttft_ms=200.0, success=True,
        evidence=Evidence.measured, source="test"))
    collector.record(TelemetryEvent(
        event_id="ooo2", correlation_id="c", workload_id="h-3", event_type="request",
        timestamp=t0, ttft_ms=100.0, success=True, evidence=Evidence.measured, source="test"))
    s3 = collector.aggregate("h-3")
    integrity["out_of_order_handled"] = s3.request_count == 2 and s3.first_event < s3.last_event
    store.close()
    return {"phase": "H", "integrity": integrity}


def main() -> int:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "D": phase_d(),
        "F": phase_f(),
        "G": phase_g(),
        "H": phase_h(),
    }
    out = RAW / "cpu_phases.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out}")
    print("D invariants:", report["D"]["invariants"])
    print("F levels:", [(r["level"], r["success"], r["failed"]) for r in report["F"]["levels"]])
    print("G states:", {k: v.get("state") for k, v in report["G"]["matrix"].items()})
    print("H integrity:", report["H"]["integrity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
