from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional, List
from decimal import Decimal
import typer
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from fusion.config import get_settings
from fusion.inventory import FixtureInventoryAdapter, InventoryService
from fusion.optimizer.engine import OptimizationEngine
from fusion.simulator.engine import FleetSimulator
from fusion.simulator.scenario import ScenarioLoader, ScenarioDefinition
from fusion.demand.forecaster import DemandForecaster
from fusion.demand.profile import WorkloadProfileBuilder
from fusion.models.types import (
    GPUInventory, ModelDeployment, TenantPolicy, PerformanceProfile,
    WorkloadDemand, OptimizationScenario,
)

app = typer.Typer(name="fusion", help="Tesseract Fusion CLI")
console = Console()
settings = get_settings()


@app.command()
def inventory_import(
    path: str = typer.Argument(..., help="Path to fleet YAML/JSON definition"),
):
    """Import GPU fleet inventory from a file."""
    adapter = FixtureInventoryAdapter(path)
    inventory = adapter.get_inventory()
    console.print(f"[green]Imported {len(inventory)} GPUs from {path}[/green]")
    table = Table(title="GPU Inventory")
    table.add_column("GPU ID")
    table.add_column("Model")
    table.add_column("Memory (MB)")
    table.add_column("Cost/hr")
    table.add_column("Status")
    for gpu in inventory:
        table.add_row(
            gpu.gpu_id, gpu.gpu_model, str(gpu.memory_capacity_mb),
            f"${gpu.resource_cost_per_hour}", gpu.health_status.value,
        )
    console.print(table)


@app.command()
def workload_import(
    path: str = typer.Argument(..., help="Path to traffic CSV or JSON"),
):
    """Import workload data from a file."""
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
    else:
        console.print(f"[red]Unsupported format: {p.suffix}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Imported workload data from {path}[/green]")
    console.print(json.dumps(data, indent=2)[:500])


@app.command()
def profile_import(
    path: str = typer.Argument(..., help="Path to performance profile JSON"),
):
    """Import a performance profile."""
    with open(path) as f:
        data = json.load(f)
    profile = PerformanceProfile(**data)
    console.print(f"[green]Imported profile: {profile.profile_id}[/green]")
    console.print(f"  Model: {profile.model_id} on {profile.hardware_id}")
    console.print(f"  TTFT p50: {profile.ttft_p50_ms}ms, p99: {profile.ttft_p99_ms}ms")
    console.print(f"  Throughput: {profile.throughput_rps} rps")


@app.command()
def simulate(
    path: str = typer.Argument(..., help="Path to scenario YAML/JSON"),
    format: str = typer.Option("text", "--format", "-f", help="Output format"),
):
    """Run a fleet simulation."""
    loader = ScenarioLoader()
    if Path(path).suffix in (".yaml", ".yml"):
        scenario = loader.load_from_yaml(path)
    else:
        scenario = loader.load_from_json(path)

    sim = FleetSimulator(seed=scenario.seed)
    result = sim.run(scenario)

    if format == "json":
        console.print(result.model_dump_json(indent=2))
    else:
        table = Table(title=f"Simulation: {scenario.name}")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Total Requests", str(result.total_requests))
        table.add_row("Completed", str(result.completed_requests))
        table.add_row("SLO Violations", str(result.slo_violations))
        table.add_row("GPU Hours", f"{result.total_gpu_hours:.1f}")
        table.add_row("Total Cost", f"${result.total_cost}")
        table.add_row("Total Revenue", f"${result.total_revenue}")
        table.add_row("Contribution", f"${result.total_contribution or 'N/A'}")
        table.add_row("Scaling Events", str(result.scaling_events))
        console.print(table)


@app.command()
def optimize(
    path: str = typer.Argument(..., help="Path to scenario YAML/JSON"),
    mode: str = typer.Option("advisory", "--mode", "-m", help="Optimization mode"),
    format: str = typer.Option("text", "--format", "-f", help="Output format"),
):
    """Run optimization on a fleet scenario."""
    loader = ScenarioLoader()
    if Path(path).suffix in (".yaml", ".yml"):
        scenario = loader.load_from_yaml(path)
    else:
        scenario = loader.load_from_json(path)

    inventory = loader.to_inventory(scenario)
    deployments = loader.to_deployments(scenario)
    policies = loader.to_tenant_policies(scenario)

    engine = OptimizationEngine()
    recommendation = engine.optimize(
        deployments=deployments,
        gpus=inventory,
        tenant_policies=policies,
        profiles={},
        demands={},
    )

    if format == "json":
        console.print(recommendation.model_dump_json(indent=2))
    else:
        console.print(Markdown(f"# Optimization: {Path(path).stem}"))
        console.print(f"\nMode: **{mode}**")
        console.print(f"Confidence: **{recommendation.confidence:.1%}**")
        console.print(f"Status: **{recommendation.approval_status.value}**")

        if recommendation.rejected_alternatives:
            console.print(f"\n## Rejected Alternatives")
            for alt, reason in recommendation.rejected_reasons.items():
                console.print(f"- {alt}: {reason}")

        table = Table(title="Economics Comparison")
        table.add_column("Metric")
        table.add_column("Baseline")
        table.add_column("Recommended")
        table.add_row("Revenue", f"${recommendation.baseline_economics.expected_revenue}", f"${recommendation.expected_economics.expected_revenue}")
        table.add_row("Cost", f"${recommendation.baseline_economics.expected_operating_cost}", f"${recommendation.expected_economics.expected_operating_cost}")
        table.add_row("Contribution", f"${recommendation.baseline_economics.expected_contribution_margin or 'N/A'}", f"${recommendation.expected_economics.expected_contribution_margin or 'N/A'}")
        console.print(table)


@app.command()
def economics(
    path: str = typer.Argument(..., help="Path to scenario YAML/JSON"),
):
    """Evaluate economics of a fleet scenario."""
    loader = ScenarioLoader()
    if Path(path).suffix in (".yaml", ".yml"):
        scenario = loader.load_from_yaml(path)
    else:
        scenario = loader.load_from_json(path)

    inventory = loader.to_inventory(scenario)
    console.print(f"\n[bold]Fleet Economics: {scenario.name}[/bold]")
    table = Table(title="GPU Pool Costs")
    table.add_column("GPU Model")
    table.add_column("Count")
    table.add_column("Cost/GPU/hr")
    table.add_column("Total/hr")
    totals = {}
    for gpu in inventory:
        if gpu.gpu_model not in totals:
            totals[gpu.gpu_model] = {"count": 0, "cost": Decimal("0")}
        totals[gpu.gpu_model]["count"] += 1
        totals[gpu.gpu_model]["cost"] += gpu.resource_cost_per_hour
    for model, info in totals.items():
        table.add_row(model, str(info["count"]), f"${info['cost'] / info['count']:.2f}", f"${info['cost']:.2f}")
    console.print(table)


@app.command()
def report(
    recommendation_id: str = typer.Argument(...),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format"),
):
    """Print a detailed recommendation report."""
    console.print(f"[yellow]Recommendation {recommendation_id}[/yellow]")
    console.print(f"Format: {format}")
    console.print("\nIn a full deployment, this would retrieve the recommendation from the database.")


@app.command()
def serve():
    """Start the Fusion API server."""
    import uvicorn
    from fusion.api import create_app
    app = create_app(settings)
    console.print(f"[green]Starting Fusion API on {settings.host}:{settings.port}[/green]")
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    app()
