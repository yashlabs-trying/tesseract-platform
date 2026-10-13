"""Command line interface for Tesseract Adaptive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from adaptive import __version__
from adaptive.analyzer import ModelAnalyzer
from adaptive.errors import AdaptiveError
from adaptive.logging import configure_logging
from adaptive.models.architecture import ModelArchitecture
from adaptive.models.enums import Granularity, QuantScheme, RuntimeBackend
from adaptive.quantize import (
    baseline_profile,
    compute_storage,
    uniform_profile,
    validate_profile,
)

if TYPE_CHECKING:
    from torch import nn

    from adaptive.quality import TokenDataset

app = typer.Typer(
    name="adaptive",
    help="Autonomous, quality-constrained precision optimization for LLM inference.",
    no_args_is_help=True,
)


def _load(analyzer: ModelAnalyzer, config: Path, model_id: str | None) -> ModelArchitecture:
    return analyzer.analyze_file(config, model_id=model_id)


@app.command()
def version() -> None:
    """Print the Adaptive version."""
    typer.echo(__version__)


@app.command()
def analyze(
    config: Path = typer.Argument(..., exists=True, readable=True, help="HF config.json"),
    model_id: str | None = typer.Option(None, "--model-id"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Analyze a model architecture and report parameter/storage accounting."""
    configure_logging()
    analyzer = ModelAnalyzer()
    try:
        arch = _load(analyzer, config, model_id)
    except AdaptiveError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    families = {k.value: v for k, v in arch.parameter_count_by_family().items()}
    report = {
        "model_id": arch.model_id,
        "model_type": arch.model_type,
        "family": arch.family.value,
        "revision": arch.revision,
        "total_parameters": arch.total_parameters,
        "quantizable_parameters": arch.quantizable_parameters,
        "baseline_dtype": arch.baseline_dtype.value,
        "baseline_weight_bytes": arch.baseline_weight_bytes,
        "num_layers": arch.num_layers,
        "num_tensors": len(arch.layers),
        "parameter_count_by_family": families,
        "structural_fingerprint": arch.structural_fingerprint(),
        "notes": list(arch.notes),
    }
    if json_output:
        typer.echo(json.dumps(report, indent=2))
        return
    typer.echo(f"model:        {arch.model_id} ({arch.family.value})")
    typer.echo(f"parameters:   {arch.total_parameters:,}")
    typer.echo(f"quantizable:  {arch.quantizable_parameters:,}")
    baseline_mb = arch.baseline_weight_bytes / 1e6
    typer.echo(f"baseline:     {arch.baseline_dtype.value} ({baseline_mb:.1f} MB)")
    typer.echo(f"tensors:      {len(arch.layers)}")


@app.command()
def storage(
    config: Path = typer.Argument(..., exists=True, readable=True),
    bits: int = typer.Option(4, "--bits"),
    runtime: RuntimeBackend = typer.Option(RuntimeBackend.torch_eager, "--runtime"),
    group_size: int = typer.Option(128, "--group-size"),
    granularity: Granularity = typer.Option(Granularity.per_group, "--granularity"),
    scheme: QuantScheme = typer.Option(QuantScheme.symmetric, "--scheme"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Report serialized storage for a uniform profile."""
    configure_logging()
    analyzer = ModelAnalyzer()
    try:
        arch = _load(analyzer, config, None)
        profile = uniform_profile(
            arch,
            bits,
            runtime=runtime,
            granularity=granularity,
            group_size=group_size if granularity is Granularity.per_group else None,
            scheme=scheme,
        )
    except AdaptiveError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    baseline = baseline_profile(arch, runtime=runtime)
    base_storage = compute_storage(arch, baseline)
    storage_result = compute_storage(arch, profile)
    issues = validate_profile(arch, profile)
    report = {
        "profile": profile.name,
        "runtime": runtime.value,
        "deployable": profile.deployable,
        "baseline_bytes": base_storage.total_bytes,
        "quantized_bytes": storage_result.total_bytes,
        "compression_ratio": storage_result.compression_ratio,
        "breakdown": storage_result.model_dump(),
        "issues": list(issues),
    }
    if json_output:
        typer.echo(json.dumps(report, indent=2))
        return
    typer.echo(f"profile:      {profile.name}")
    typer.echo(f"baseline:     {base_storage.total_mb:.1f} MB")
    typer.echo(f"quantized:    {storage_result.total_mb:.1f} MB")
    typer.echo(f"compression:  {storage_result.compression_ratio:.2f}x")
    typer.echo(f"deployable:   {profile.deployable}")
    for issue in issues:
        typer.secho(f"  issue: {issue}", fg=typer.colors.YELLOW)


def _reference_context(
    config: Path, model_id: str | None, seed: int = 0
) -> tuple[ModelArchitecture, nn.Module, TokenDataset, TokenDataset]:
    """Build a synthetic reference model + datasets for M2 commands."""
    from adaptive.nets import ReferenceTransformer
    from adaptive.quality import DatasetSplit, make_synthetic_dataset

    arch = ModelAnalyzer().analyze_file(config, model_id=model_id)
    model = ReferenceTransformer.from_architecture(arch, seed=seed)
    calibration = make_synthetic_dataset(
        "calibration",
        DatasetSplit.calibration,
        vocab_size=arch.vocab_size,
        num_sequences=4,
        seq_len=16,
        seed=seed + 1,
    )
    evaluation = make_synthetic_dataset(
        "evaluation",
        DatasetSplit.evaluation,
        vocab_size=arch.vocab_size,
        num_sequences=3,
        seq_len=16,
        seed=seed + 2,
    )
    return arch, model, calibration, evaluation


@app.command()
def sensitivity(
    config: Path = typer.Argument(..., exists=True, readable=True),
    bits: int = typer.Option(4, "--bits"),
    group_size: int = typer.Option(16, "--group-size"),
    top_k: int = typer.Option(5, "--top-k"),
    model_id: str | None = typer.Option(None, "--model-id"),
) -> None:
    """Rank layer sensitivity for a synthetic reference model (no weights downloaded)."""
    configure_logging()
    try:
        from adaptive.sensitivity import SensitivityMethod, analyze_sensitivity

        arch, model, calibration, _ = _reference_context(config, model_id)
        report = analyze_sensitivity(
            arch,
            model,
            calibration.sequences,
            methods=(
                SensitivityMethod.weight_error,
                SensitivityMethod.activation_weighted,
                SensitivityMethod.output_divergence,
            ),
            bits=bits,
            group_size=group_size,
            calibration_fingerprint=calibration.fingerprint(),
        )
    except AdaptiveError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    for method in report.methods:
        typer.echo(f"[{method.value}]")
        for rank, name in enumerate(report.top_k(method, top_k), start=1):
            typer.echo(f"  {rank}. {name}  score={report.score(name, method):.3f}")


@app.command()
def optimize(
    config: Path = typer.Argument(..., exists=True, readable=True),
    tolerance: float = typer.Option(0.05, "--tolerance"),
    group_size: int = typer.Option(16, "--group-size"),
    algorithm: str = typer.Option("greedy", "--algorithm"),
    model_id: str | None = typer.Option(None, "--model-id"),
) -> None:
    """Optimize precision under a quality tolerance using a synthetic reference model."""
    configure_logging()
    try:
        from adaptive.models.optimization import OptimizationConstraints
        from adaptive.optimizer import optimize as run_optimize
        from adaptive.sensitivity import analyze_sensitivity

        arch, model, calibration, evaluation = _reference_context(config, model_id)
        report = analyze_sensitivity(
            arch,
            model,
            calibration.sequences,
            bits=4,
            group_size=group_size,
            calibration_fingerprint=calibration.fingerprint(),
        )
        constraints = OptimizationConstraints(
            quality_tolerance=tolerance, allowed_bits=(4, 8, 16)
        )
        result = run_optimize(
            arch,
            report,
            constraints,
            baseline_metric_value=1.0,
            group_size=group_size,
            algorithm=algorithm,
        )
    except AdaptiveError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    selected = result.selected()
    typer.echo(f"algorithm:  {result.algorithm}")
    typer.echo(f"candidates: {len(result.candidates)}")
    typer.echo(f"pareto:     {', '.join(result.pareto_frontier)}")
    if selected is not None:
        typer.echo(f"selected:   {selected.profile.name} ({selected.profile.profile_id})")
        typer.echo(f"  size:     {selected.storage.total_mb:.1f} MB")
        typer.echo(f"  cost:     {selected.cost.lower_cost:.6f}")
        typer.echo(f"  feasible: {selected.feasible}")
        for violation in selected.violations:
            typer.secho(f"  violation: {violation}", fg=typer.colors.YELLOW)


@app.command()
def controller(
    config: Path = typer.Argument(..., exists=True, readable=True),
    scenario: str = typer.Option("pressure", "--scenario", help="steady|pressure|burst"),
    tolerance: float = typer.Option(0.05, "--tolerance"),
    group_size: int = typer.Option(16, "--group-size"),
    model_id: str | None = typer.Option(None, "--model-id"),
) -> None:
    """Simulate the adaptive controller on a telemetry scenario (no GPU required)."""
    configure_logging()
    try:
        from adaptive.controller import ControllerPolicy, ProfileOption, run_simulation
        from adaptive.optimizer import AnalyticCostModel
        from adaptive.telemetry import (
            memory_pressure_scenario,
            steady_scenario,
            traffic_burst_scenario,
        )

        arch = ModelAnalyzer().analyze_file(config, model_id=model_id)
        cost_model = AnalyticCostModel()
        degradations = {4: 0.05, 8: 0.01, 16: 0.0}
        options = []
        for bits in (4, 8, 16):
            profile = uniform_profile(arch, bits, group_size=group_size)
            cost = cost_model.estimate(arch, profile)
            options.append(
                ProfileOption(
                    profile_id=profile.profile_id,
                    precision_bits=bits,
                    estimated_cost=cost.lower_cost,
                    weight_bytes=cost.weight_bytes,
                    quality_degradation=degradations[bits],
                )
            )
        if scenario == "steady":
            snapshots = steady_scenario()
        elif scenario == "pressure":
            snapshots = memory_pressure_scenario()
        elif scenario == "burst":
            snapshots = traffic_burst_scenario()
        else:
            typer.secho(f"error: unknown scenario {scenario}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        policy = ControllerPolicy(quality_tolerance=tolerance, min_dwell_s=0.0, cooldown_s=0.0)
        result = run_simulation(policy, options, snapshots, automatic=True)
    except AdaptiveError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"scenario:   {scenario}")
    typer.echo(f"switches:   {result.switch_count}")
    typer.echo(f"rollbacks:  {result.rollback_count}")
    typer.echo(f"oscillated: {result.oscillated}")
    typer.echo(f"final:      {result.final_profile_id}")
    for decision in result.decisions:
        typer.echo(f"  t={decision.action.value:<10} {decision.reason}")


if __name__ == "__main__":  # pragma: no cover
    app()
