"""Phase J — control-plane performance baseline (CPU, in-process).

Measures platform overhead only. This is NOT model-inference throughput.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from integrations.foundry import map_performance_profile
from integrations.ignite import build_replicagroup
from tesseract_platform.api.app import create_app
from tesseract_platform.config import Settings
from tesseract_platform.coordinator import Reconciler, WorkloadCoordinator
from tesseract_platform.models import Evidence, TelemetryEvent
from tesseract_platform.observability import OptimizationController, TelemetryCollector
from tesseract_platform.observability.telemetry import percentile
from tesseract_platform.policies import PlatformPolicy
from tesseract_platform.runtime import build_platform
from tesseract_platform.state import StateStore
from tesseract_platform.testing import (
    FakeDeploymentProvider,
    FakePlacementProvider,
    FakeProfileProvider,
    make_request,
)

OUT = Path("/home/yashs/tesseract-platform/reports/raw/performance_baseline.json")

PROFILE_DOC = {
    "schema_version": "tesseract.performance-profile.v1",
    "profile_id": "p1", "model_id": "Qwen/Qwen3-0.6B", "model_revision": "main",
    "hardware_id": "l4-1", "engine": "vLLM", "evidence_type": "synthetic",
    "collection_timestamp": "2026-01-01T00:00:00Z", "source_type": "test_fixture",
    "ttft_p95_ms": 50.0, "output_tokens_per_second": 50.0,
}


def measure(fn, n, warmup=20):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "n": n,
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
        "mean_ms": sum(samples) / len(samples),
    }


def main() -> int:
    results = {"phase": "J", "generated_at": datetime.now(timezone.utc).isoformat(),
               "note": "control-plane overhead only; not model inference"}

    results["foundry_profile_mapping"] = measure(lambda: map_performance_profile(PROFILE_DOC), 2000)

    store = StateStore("sqlite:///:memory:")
    req = make_request()
    reservation_id = {"i": 0}
    from tesseract_platform.models import Reservation, ReservationState
    def reserve_txn():
        reservation_id["i"] += 1
        res = Reservation(reservation_id=f"r{reservation_id['i']}", operation_id="o",
                          idempotency_key="idem-key-0001", workload_id="w",
                          state=ReservationState.RESERVING, revision=0)
        store.save_reservation(res)
        store.get_reservation(res.reservation_id)
    results["reservation_transaction"] = measure(reserve_txn, 1000)

    results["ignite_manifest_build"] = measure(lambda: build_replicagroup(req, "wl-1"), 2000)

    coordinator = WorkloadCoordinator(
        store=store, policy=PlatformPolicy(), profile_provider=FakeProfileProvider(),
        placement_provider=FakePlacementProvider(), deployment_provider=FakeDeploymentProvider(),
        telemetry=TelemetryCollector(store))
    counter = {"i": 0}
    def full_submit():
        counter["i"] += 1
        coordinator.submit(make_request(workload_id=f"p-{counter['i']}",
                                        idempotency_key=f"perf-idem-{counter['i']:08d}"))
    results["coordinator_full_lifecycle"] = measure(full_submit, 300)

    def rollback():
        counter["i"] += 1
        op = coordinator.submit(make_request(workload_id=f"r-{counter['i']}",
                                             idempotency_key=f"rb-idem-{counter['i']:08d}"))
        coordinator.rollback(op.operation_id)
    results["rollback_completion"] = measure(rollback, 200)

    collector = TelemetryCollector(store)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(500):
        collector.record(TelemetryEvent(event_id=f"e{i}", correlation_id="c", workload_id="perf",
                                        event_type="request", timestamp=t0, ttft_ms=100.0,
                                        success=True, evidence=Evidence.measured, source="perf"))
    results["telemetry_aggregate_500"] = measure(lambda: collector.aggregate("perf"), 300)

    controller = OptimizationController(store, collector)
    from tesseract_platform.models import SLOSpec
    results["controller_evaluate"] = measure(lambda: controller.evaluate("perf", SLOSpec(ttft_p95_ms=100.0)), 1000)

    reconciler = Reconciler(coordinator, store, readiness_timeout_seconds=30)
    results["reconciler_pass"] = measure(lambda: reconciler.reconcile(), 300)

    platform = build_platform(
        Settings(state_dsn="sqlite:///:memory:"), profile_provider=FakeProfileProvider(),
        placement_provider=FakePlacementProvider(), deployment_provider=FakeDeploymentProvider())
    client = TestClient(create_app(platform))
    api_counter = {"i": 0}
    def api_request():
        api_counter["i"] += 1
        client.post("/v1/workloads", json={
            "workload_id": f"api-{api_counter['i']}",
            "idempotency_key": f"api-idem-{api_counter['i']:08d}",
            "model": {"model_id": "m", "revision": "r"}})
    results["api_request_overhead"] = measure(api_request, 200)
    results["api_healthz"] = measure(lambda: client.get("/healthz"), 500)
    platform.close()
    store.close()

    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT}")
    for k, v in results.items():
        if isinstance(v, dict) and "p50_ms" in v:
            print(f"  {k:32s} p50={v['p50_ms']:.3f}ms p95={v['p95_ms']:.3f}ms p99={v['p99_ms']:.3f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
