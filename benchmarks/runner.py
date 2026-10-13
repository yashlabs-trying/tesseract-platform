#!/usr/bin/env python3
"""Local benchmark harness.

Runs the *real* platform coordinator in-process against fixture providers and
emits machine-readable JSON plus a human-readable Markdown report with full
hardware/commit provenance.

This is a FIXTURE baseline. It measures the platform's orchestration overhead,
not model-server or GPU performance. It must not be reported as production
throughput or as GPU validation.
"""

from __future__ import annotations

import argparse
import json
import platform as py_platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tesseract_platform import __version__  # noqa: E402
from tesseract_platform.config import Settings  # noqa: E402
from tesseract_platform.observability.telemetry import percentile  # noqa: E402
from tesseract_platform.runtime import build_platform  # noqa: E402
from tesseract_platform.testing import (  # noqa: E402
    FakeDeploymentProvider,
    FakePlacementProvider,
    FakeProfileProvider,
    make_request,
)

REGRESSION_THRESHOLD_PCT = 20.0


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        )
    except Exception:
        return "unknown"


def _manifest() -> dict:
    path = ROOT / "platform.manifest.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _run(scenario: dict) -> dict:
    settings = Settings(state_dsn="sqlite:///:memory:")
    platform = build_platform(
        settings,
        profile_provider=FakeProfileProvider(),
        placement_provider=FakePlacementProvider(),
        deployment_provider=FakeDeploymentProvider(),
    )

    concurrency = int(scenario.get("concurrency", 1))
    request_count = int(scenario.get("request_count", 10))

    def one(index: int) -> float:
        request = make_request(
            workload_id=f"bench-{index}",
            idempotency_key=f"bench-key-{index:08d}",
            model_id=scenario.get("model_id", "qwen3-0.6b"),
            replicas=int(scenario.get("replicas", 1)),
            min_gpu_count=int(scenario.get("min_gpu_count", 1)),
            ttft_p95_ms=scenario.get("slo_ttft_p95_ms"),
        )
        started = time.perf_counter()
        op = platform.coordinator.submit(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if op.state.value != "ACTIVE":
            raise RuntimeError(f"operation {op.operation_id} ended in {op.state.value}")
        return elapsed_ms

    started = time.perf_counter()
    latencies: list[float] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, i) for i in range(request_count)]
        for future in futures:
            try:
                latencies.append(future.result())
            except Exception:
                failures += 1
    duration = time.perf_counter() - started
    platform.close()

    successes = len(latencies)
    report = {
        "scenario": scenario.get("name", "unnamed"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform_version": __version__,
        "git_commit": _git_commit(),
        "environment": {
            "python": sys.version.split()[0],
            "implementation": py_platform.python_implementation(),
            "machine": py_platform.machine(),
            "system": py_platform.system(),
            "release": py_platform.release(),
        },
        "pinned_products": {
            key: value.get("pinned_commit")
            for key, value in (_manifest().get("products") or {}).items()
        },
        "model_or_fixture": scenario.get("model_id", "qwen3-0.6b"),
        "runtime": "in-process coordinator + fixture providers",
        "hardware": "none (simulated GPU capacity)",
        "precision": scenario.get("precision", "bf16"),
        "workload": {
            "concurrency": concurrency,
            "request_count": request_count,
            "replicas": scenario.get("replicas", 1),
            "min_gpu_count": scenario.get("min_gpu_count", 1),
        },
        "request_count": request_count,
        "success_count": successes,
        "failure_count": failures,
        "duration_seconds": round(duration, 4),
        "throughput_rps": round(successes / duration, 4) if duration > 0 else 0.0,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
        },
        "ttft_ms": None,
        "itl_ms": None,
        "resource_utilisation": "not collected (in-process)",
        "known_limitations": [
            "Fixture providers only; no model server, no GPU.",
            "Measures orchestration overhead, not inference throughput.",
            "Not a production throughput claim.",
        ],
    }
    return report


def _compare_baseline(report: dict, out_dir: Path) -> dict:
    baseline_path = ROOT / "benchmarks" / "baselines" / f"{report['scenario']}.json"
    if not baseline_path.exists():
        return {"baseline": None, "note": "no baseline recorded"}
    with baseline_path.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    base_p95 = (baseline.get("latency_ms") or {}).get("p95")
    new_p95 = (report.get("latency_ms") or {}).get("p95")
    delta_pct = None
    regression = False
    if base_p95 and new_p95:
        delta_pct = (new_p95 - base_p95) / base_p95 * 100.0
        regression = delta_pct > REGRESSION_THRESHOLD_PCT
    return {
        "baseline": str(baseline_path),
        "baseline_p95_ms": base_p95,
        "current_p95_ms": new_p95,
        "delta_pct": delta_pct,
        "regression_suspected": regression,
        "note": "small samples; not statistically significant",
    }


def _write_markdown(report: dict, comparison: dict, out_dir: Path) -> Path:
    path = out_dir / f"{report['scenario']}.md"
    lat = report["latency_ms"]
    lines = [
        f"# Benchmark: {report['scenario']}",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Platform: {report['platform_version']} @ {report['git_commit'][:10]}",
        f"- Runtime: {report['runtime']}",
        f"- Hardware: {report['hardware']}",
        f"- Model/fixture: {report['model_or_fixture']} ({report['precision']})",
        f"- Concurrency: {report['workload']['concurrency']}",
        f"- Requests: {report['request_count']} ({report['success_count']} ok, {report['failure_count']} failed)",
        f"- Duration: {report['duration_seconds']}s",
        f"- Throughput: {report['throughput_rps']} rps",
        f"- Latency ms: p50={lat['p50']}, p95={lat['p95']}, p99={lat['p99']}",
        "",
        "## Regression detector",
        "",
        f"```json\n{json.dumps(comparison, indent=2)}\n```",
        "",
        "## Known limitations",
        "",
    ]
    lines += [f"- {item}" for item in report["known_limitations"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Tesseract local benchmark harness")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", default="benchmarks/reports")
    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    if not scenario_path.is_absolute():
        scenario_path = ROOT / scenario_path
    with scenario_path.open(encoding="utf-8") as handle:
        scenario = json.load(handle)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _run(scenario)
    comparison = _compare_baseline(report, out_dir)
    report["regression_check"] = comparison

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"{report['scenario']}-{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = _write_markdown(report, comparison, out_dir)

    print(f"JSON report: {json_path}")
    print(f"MD report:   {md_path}")
    print(f"p95 latency: {report['latency_ms']['p95']} ms  failures: {report['failure_count']}")
    if comparison.get("regression_suspected"):
        print("WARNING: possible regression vs baseline (small sample)")
    return 0 if report["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
