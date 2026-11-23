"""Performance and endurance audit for Tesseract Fusion.

These tests measure scheduling latency, scaling behaviour, bounded exhaustive
search, repeated-call memory stability, a short soak, and simulator throughput.
Everything runs against the deterministic optimizer and the simulated fleet --
no real GPU hardware is involved.

The slow tests are marked ``@pytest.mark.slow``.  Run them with::

    pytest tests/audit/test_endurance.py -m slow

Baseline and final process resource usage is recorded by the module-scoped
``record_resource_usage`` fixture (via :mod:`resource` where available) and by
``tracemalloc`` inside the leak test.
"""
from __future__ import annotations

import gc
import os
import statistics
import sys
import time
import tracemalloc
from decimal import Decimal

import pytest

from fusion.models.types import GPUInventory, HealthStatus, ModelDeployment, TenantPolicy
from fusion.optimizer.candidate import CandidateGenerator
from fusion.optimizer.engine import OptimizationEngine
from fusion.optimizer.global_allocator import GlobalAllocator
from fusion.simulator.engine import FleetSimulator
from fusion.simulator.scenario import ScenarioLoader

try:  # pragma: no cover - exercised on POSIX CI
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]


SOAK_SECONDS = float(os.environ.get("FUSION_AUDIT_SOAK_SECONDS", "30"))
LATENCY_BUDGET_SECONDS = 5.0


# --------------------------------------------------------------------------- #
# Resource baseline / final recording
# --------------------------------------------------------------------------- #
def _rss_mb() -> float | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    if sys.platform == "darwin":  # pragma: no cover
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


@pytest.fixture(scope="module", autouse=True)
def record_resource_usage():
    """Record baseline and final resource usage for the audit report."""
    baseline_rss = _rss_mb()
    baseline_wall = time.perf_counter()
    yield
    final_rss = _rss_mb()
    elapsed = time.perf_counter() - baseline_wall
    print(
        f"\n[endurance] resource baseline: rss_mb={baseline_rss} | "
        f"final: rss_mb={final_rss} | wall_s={elapsed:.2f}"
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_gpu(gpu_id: str, slots: int = 4, memory_mb: int = 81920, cost: str = "2.0"):
    return GPUInventory(
        gpu_id=gpu_id,
        node_id=f"node-{gpu_id}",
        gpu_model="H100",
        memory_capacity_mb=memory_mb,
        available_memory_mb=memory_mb,
        available_resource_slots=slots,
        health_status=HealthStatus.HEALTHY,
        resource_cost_per_hour=Decimal(cost),
    )


def make_deployment(
    deployment_id: str, tenant_id: str, worker_count: int = 1, memory_mb: int = 40960
):
    return ModelDeployment(
        deployment_id=deployment_id,
        tenant_id=tenant_id,
        model_id="m1",
        worker_count=worker_count,
        expected_memory_mb=memory_mb,
    )


def make_engine(max_candidates: int = 20) -> OptimizationEngine:
    return OptimizationEngine({"max_candidates_per_workload": max_candidates})


# --------------------------------------------------------------------------- #
# 1. Scheduling latency
# --------------------------------------------------------------------------- #
def test_scheduling_latency():
    """A single optimize call must complete well under the 5s budget."""
    engine = make_engine()
    deployments = [make_deployment(f"d{i}", f"t{i}") for i in range(5)]
    gpus = [make_gpu(f"g{i}") for i in range(8)]

    start = time.perf_counter()
    rec = engine.optimize(deployments, gpus, {}, {}, {})
    elapsed = time.perf_counter() - start

    assert rec is not None
    assert elapsed < LATENCY_BUDGET_SECONDS, f"optimize took {elapsed:.3f}s"


# --------------------------------------------------------------------------- #
# 2. Scaling with fleet size
# --------------------------------------------------------------------------- #
def test_scaling_fleet_sizes():
    """Optimize latency across fleets of 1..64 GPUs stays bounded."""
    engine = make_engine()
    deployments = [make_deployment(f"d{i}", f"t{i}") for i in range(4)]

    timings: dict[int, float] = {}
    for count in (1, 2, 4, 8, 16, 32, 64):
        gpus = [make_gpu(f"g{i}") for i in range(count)]
        start = time.perf_counter()
        rec = engine.optimize(deployments, gpus, {}, {}, {})
        timings[count] = time.perf_counter() - start
        assert rec is not None

    print(f"\n[endurance] fleet-size timings (s): {timings}")
    assert all(t < LATENCY_BUDGET_SECONDS for t in timings.values())


# --------------------------------------------------------------------------- #
# 3. Scaling with workload count
# --------------------------------------------------------------------------- #
def test_scaling_workload_counts():
    """Optimize latency across 1..50 workloads stays bounded."""
    engine = make_engine()
    gpus = [make_gpu(f"g{i}") for i in range(8)]

    timings: dict[int, float] = {}
    for count in (1, 5, 10, 20, 50):
        deployments = [make_deployment(f"d{i}", f"t{i}") for i in range(count)]
        start = time.perf_counter()
        rec = engine.optimize(deployments, gpus, {}, {}, {})
        timings[count] = time.perf_counter() - start
        assert rec is not None

    print(f"\n[endurance] workload-count timings (s): {timings}")
    assert all(t < LATENCY_BUDGET_SECONDS for t in timings.values())


# --------------------------------------------------------------------------- #
# 4. Bounded exhaustive search
# --------------------------------------------------------------------------- #
def test_optimization_runtime_bounded():
    """Exhaustive search must stop at ``max_exhaustive_combinations``."""
    cap = 7
    alloc = GlobalAllocator(
        {"max_exhaustive_combinations": cap, "max_candidates_per_workload": 20}
    )
    gpus = [make_gpu("g0", slots=10), make_gpu("g1", slots=10)]
    gpu_map = {g.gpu_id: g for g in gpus}
    deployments = [make_deployment(f"d{i}", "t0") for i in range(4)]
    policies = {"t0": TenantPolicy(tenant_id="t0")}
    all_alts = {
        d.deployment_id: CandidateGenerator().generate_alternatives(d, gpus)
        for d in deployments
    }

    calls = {"n": 0}
    original = alloc._check_global_feasibility

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    alloc._check_global_feasibility = counted  # type: ignore[method-assign]

    start = time.perf_counter()
    rec = alloc._exhaustive_search(
        deployments, gpus, gpu_map, policies, {}, all_alts, []
    )
    elapsed = time.perf_counter() - start

    assert calls["n"] <= cap, f"evaluated {calls['n']} combinations (cap {cap})"
    assert elapsed < LATENCY_BUDGET_SECONDS
    assert rec is not None


# --------------------------------------------------------------------------- #
# 5. Repeated calls / memory
# --------------------------------------------------------------------------- #
def test_repeated_calls_no_leak():
    """200 optimize calls must not retain unbounded memory."""
    engine = make_engine()
    deployments = [make_deployment(f"d{i}", f"t{i}") for i in range(5)]
    gpus = [make_gpu(f"g{i}") for i in range(8)]

    sizes: list[int] = []
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    for i in range(200):
        rec = engine.optimize(deployments, gpus, {}, {}, {})
        if i in (0, 199):
            sizes.append(sys.getsizeof(rec))
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    retained_growth = current - baseline
    print(
        f"\n[endurance] repeated-call tracemalloc retained="
        f"{retained_growth / 1024:.2f}KB peak={peak / 1024:.2f}KB"
    )

    assert retained_growth < 5 * 1024 * 1024, "retained memory grew beyond 5MB"
    # ``sys.getsizeof`` of the returned recommendation stays bounded.
    assert abs(sizes[-1] - sizes[0]) < 100_000


# --------------------------------------------------------------------------- #
# 6. Short soak
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_soak_short():
    """Run the optimizer in a loop for ~30s; no crash, stable latency."""
    engine = make_engine()
    deployments = [make_deployment(f"d{i}", f"t{i}") for i in range(5)]
    gpus = [make_gpu(f"g{i}") for i in range(8)]

    deadline = time.perf_counter() + SOAK_SECONDS
    latencies: list[float] = []
    while time.perf_counter() < deadline:
        start = time.perf_counter()
        rec = engine.optimize(deployments, gpus, {}, {}, {})
        latencies.append(time.perf_counter() - start)
        assert rec is not None

    assert latencies, "soak loop did not execute"
    assert max(latencies) < LATENCY_BUDGET_SECONDS

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    print(
        f"\n[endurance] soak iterations={len(latencies)} "
        f"p50={p50:.4f}s p95={p95:.4f}s max={max(latencies):.4f}s"
    )
    # Latency must remain stable: p95 not wildly above the median.
    assert p95 < max(p50 * 10.0, 0.5)


# --------------------------------------------------------------------------- #
# 7. Simulation throughput
# --------------------------------------------------------------------------- #
def test_simulation_throughput():
    """Measure how many fleet simulations complete per second."""
    scenario = ScenarioLoader().load_from_dict(
        {
            "name": "throughput-audit",
            "gpu_pools": [
                {"gpu_model": "H100", "count": 4, "memory_capacity_mb": 81920, "cost_per_hour": 4.5}
            ],
            "workloads": [
                {"name": "w1", "model_id": "m1", "tenant_id": "t1", "base_rps": 10.0},
                {"name": "w2", "model_id": "m2", "tenant_id": "t2", "base_rps": 5.0},
            ],
            "duration_hours": 0.1,
        }
    )
    simulator = FleetSimulator(seed=42)
    iterations = 20

    start = time.perf_counter()
    for _ in range(iterations):
        result = simulator.run(scenario)
        assert result.state.value == "completed"
    elapsed = time.perf_counter() - start

    throughput = iterations / elapsed
    print(f"\n[endurance] simulation throughput: {throughput:.2f} sims/sec")
    assert throughput > 0
