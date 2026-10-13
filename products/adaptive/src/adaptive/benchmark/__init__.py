"""Benchmark reporting and GPU test manifests (Milestone 5).

Report generation accepts only measured or validated data (typically imported
from Foundry); GPU test manifests pin model, profile, hardware, workload and
thresholds for reproducible validation on a GPU host.
"""

from __future__ import annotations

from adaptive.benchmark.manifest import (
    GPU_TEST_MANIFEST_SCHEMA,
    GpuHardwareSpec,
    GpuProfileSpec,
    GpuTestManifest,
    GpuThresholds,
    GpuWorkloadSpec,
    load_manifest,
    load_manifests,
)
from adaptive.benchmark.report import (
    BENCHMARK_REPORT_SCHEMA,
    BenchmarkReport,
    ProfileBenchmarkSummary,
    build_benchmark_report,
    write_benchmark_markdown,
    write_benchmark_report,
)

__all__ = [
    "BENCHMARK_REPORT_SCHEMA",
    "GPU_TEST_MANIFEST_SCHEMA",
    "BenchmarkReport",
    "GpuHardwareSpec",
    "GpuProfileSpec",
    "GpuTestManifest",
    "GpuThresholds",
    "GpuWorkloadSpec",
    "ProfileBenchmarkSummary",
    "build_benchmark_report",
    "load_manifest",
    "load_manifests",
    "write_benchmark_markdown",
    "write_benchmark_report",
]
