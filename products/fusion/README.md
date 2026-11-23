# Tesseract Fusion

GPU fleet economics and optimization platform.

Fusion determines **which feasible infrastructure decision delivers the greatest expected commercial benefit while honoring every mandatory operational and customer constraint**.

## Architecture

```
┌────────────────────────────────────┐
│         API (FastAPI :8100)        │
├────────────────────────────────────┤
│         CLI (Typer)                │
├────────────────────────────────────┤
│  ┌─────────┐ ┌───────────────┐     │
│  │Optimizer│ │  Simulator    │     │
│  └─────────┘ └───────────────┘     │
│  ┌──────┐ ┌────────┐ ┌──────────┐ │
│  │Cost  │ │Demand  │ │Policy    │ │
│  │Engine│ │Forecast│ │Validator │ │
│  └──────┘ └────────┘ └──────────┘ │
├────────────────────────────────────┤
│  Adapters: K8s, Prom, OpenCost,   │
│           Dynamo, KAI              │
└────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run API server
fusion serve

# Run CLI commands
fusion inventory import fixtures/small-fleet.yaml
fusion simulate fixtures/fleet.yaml
fusion optimize fixtures/fleet.yaml
fusion economics fixtures/fleet.yaml

# Run demo
python examples/demo_report.py
```

## Tests

```bash
pytest tests/ -v
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Health check |
| GET | `/readyz` | Readiness check |
| POST | `/v1/inventory/import` | Import GPU inventory |
| GET | `/v1/inventory` | List inventory |
| POST | `/v1/workloads/import` | Import workloads |
| POST | `/v1/performance-profiles/import` | Import profiles |
| POST | `/v1/optimization/evaluate` | Run optimization |
| POST | `/v1/simulations` | Run simulation |
| POST | `/v1/forecast/evaluate` | Forecast demand |

## License

MIT
