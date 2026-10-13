"""Command line interface for Tesseract Adaptive."""

from __future__ import annotations

import json
from pathlib import Path

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


if __name__ == "__main__":  # pragma: no cover
    app()
