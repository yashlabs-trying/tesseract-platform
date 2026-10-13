"""Benchmark report generation (Milestone 5).

Reports are built *only* from measured or validated benchmark data (typically
imported from Foundry). Synthetic or estimated inputs are rejected by default,
so a GPU report can never contain fabricated numbers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from adaptive.errors import ConfigurationError
from adaptive.integration.foundry import BenchmarkMeasurement
from adaptive.models.enums import EvidenceType
from adaptive.schemas import write_document

BENCHMARK_REPORT_SCHEMA = "tesseract.adaptive.benchmark-report.v1"


class ProfileBenchmarkSummary(BaseModel):
    """Per-profile summary of a set of measurements."""

    model_config = {"frozen": True}

    profile_id: str
    engine: str = ""
    hardware_gpu_model: str = ""
    ttft_p95_ms: float | None = None
    itl_p95_ms: float | None = None
    throughput_tps: float | None = None
    evidence: EvidenceType = EvidenceType.measured


class BenchmarkReport(BaseModel):
    """A reproducible benchmark report."""

    model_config = {"frozen": True}

    schema_version: str = BENCHMARK_REPORT_SCHEMA
    model_id: str
    generated_by: str = "tesseract-adaptive"
    evidence: EvidenceType = EvidenceType.measured
    measurements: tuple[BenchmarkMeasurement, ...] = Field(default_factory=tuple)
    summaries: tuple[ProfileBenchmarkSummary, ...] = Field(default_factory=tuple)
    notes: tuple[str, ...] = Field(default_factory=tuple)

    def to_markdown(self) -> str:
        lines = [
            f"# Benchmark Report - {self.model_id}",
            "",
            f"Evidence: {self.evidence.value}",
            "",
            "| profile | engine | GPU | TTFT p95 (ms) | ITL p95 (ms) | throughput (tps) |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        for summary in self.summaries:
            lines.append(
                f"| {summary.profile_id} | {summary.engine} | {summary.hardware_gpu_model} "
                f"| {_fmt(summary.ttft_p95_ms)} | {_fmt(summary.itl_p95_ms)} "
                f"| {_fmt(summary.throughput_tps)} |"
            )
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def build_benchmark_report(
    measurements: Iterable[BenchmarkMeasurement],
    *,
    require_measured: bool = True,
    generated_by: str = "tesseract-adaptive",
) -> BenchmarkReport:
    """Build a benchmark report from measurements, rejecting fabricated data."""
    items = list(measurements)
    if not items:
        raise ConfigurationError("cannot build a benchmark report from no measurements")
    if require_measured:
        unmeasured = [
            item.profile_id
            for item in items
            if item.evidence in (EvidenceType.synthetic, EvidenceType.estimated)
        ]
        if unmeasured:
            raise ConfigurationError(
                "benchmark reports require measured/validated data; "
                f"unmeasured profiles: {', '.join(sorted(set(unmeasured)))}"
            )

    by_profile: dict[str, list[BenchmarkMeasurement]] = {}
    for item in items:
        by_profile.setdefault(item.profile_id, []).append(item)

    summaries: list[ProfileBenchmarkSummary] = []
    for profile_id, group in sorted(by_profile.items()):
        first = group[0]
        summaries.append(
            ProfileBenchmarkSummary(
                profile_id=profile_id,
                engine=first.engine,
                hardware_gpu_model=first.hardware_gpu_model,
                ttft_p95_ms=_first_present(item.ttft_p95_ms for item in group),
                itl_p95_ms=_first_present(item.itl_p95_ms for item in group),
                throughput_tps=_first_present(item.throughput_tps for item in group),
                evidence=_weakest_evidence(group),
            )
        )

    evidence = (
        EvidenceType.validated
        if all(item.evidence is EvidenceType.validated for item in items)
        else EvidenceType.measured
    )
    return BenchmarkReport(
        model_id=items[0].model_id,
        generated_by=generated_by,
        evidence=evidence,
        measurements=tuple(items),
        summaries=tuple(summaries),
        notes=("generated from imported benchmark measurements; no values are fabricated",),
    )


def _first_present(values: Iterable[float | None]) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _weakest_evidence(items: Sequence[BenchmarkMeasurement]) -> EvidenceType:
    if any(item.evidence is EvidenceType.estimated for item in items):
        return EvidenceType.estimated
    if any(item.evidence is EvidenceType.measured for item in items):
        return EvidenceType.measured
    return EvidenceType.validated


def write_benchmark_report(report: BenchmarkReport, path: Path | str) -> None:
    """Atomically write a benchmark report as JSON."""
    write_document(report.model_dump(mode="json"), path)


def write_benchmark_markdown(report: BenchmarkReport, path: Path | str) -> None:
    """Atomically write a benchmark report as Markdown."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(report.to_markdown())
    tmp.replace(target)


__all__ = [
    "BENCHMARK_REPORT_SCHEMA",
    "BenchmarkReport",
    "ProfileBenchmarkSummary",
    "build_benchmark_report",
    "write_benchmark_markdown",
    "write_benchmark_report",
]
