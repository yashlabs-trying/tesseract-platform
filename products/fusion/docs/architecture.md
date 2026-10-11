# Tesseract Fusion Architecture

## Overview

Fusion is a standalone GPU fleet economics and optimization platform. It provides a commercially intelligent layer that helps GPU infrastructure providers maximize profitability while respecting performance, hardware, reliability, and tenant-isolation constraints.

## Architecture Components

```
┌─────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                  │
├─────────────────────────────────────────────────────────┤
│                     CLI (Typer)                          │
├─────────────────────────────────────────────────────────┤
│                     Optimization Engine                  │
│  ┌────────┐ ┌────────┐ ┌──────────┐ ┌───────────────┐  │
│  │Inventory│ │Demand  │ │Economics │ │Policy          │  │
│  │Service  │ │Forecast│ │Engine    │ │Validator       │  │
│  └────────┘ └────────┘ └──────────┘ └───────────────┘  │
│  ┌────────┐ ┌────────┐ ┌──────────┐ ┌───────────────┐  │
│  │Candidate│ │Alter.  │ │Optim.    │ │Recomm.        │  │
│  │Generator│ │Eval.   │ │Engine    │ │Generator      │  │
│  └────────┘ └────────┘ └──────────┘ └───────────────┘  │
├─────────────────────────────────────────────────────────┤
│                     Fleet Simulator                     │
├─────────────────────────────────────────────────────────┤
│                External Adapter Layer                    │
│  ┌──────────┐ ┌──────────┐ ┌───────┐ ┌─────┐ ┌─────┐  │
│  │Kubernetes│ │Prometheus│ │OpenCost│ │Dynamo│ │KAI  │  │
│  └──────────┘ └──────────┘ └───────┘ └─────┘ └─────┘  │
└─────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Single control-plane service** - No microservices needed for v0.1
2. **ADVISORY mode by default** - Never mutates cluster state without approval
3. **Deterministic optimization** - Transparent, reproducible candidate evaluation
4. **SQLite first** - PostgreSQL for production via connection string
5. **Protocol-based adapters** - Pluggable data sources

## Data Flow

1. Import inventory (fixtures, Kubernetes, or manual)
2. Import workloads and performance profiles
3. Run optimization → generate deployment candidates
4. Validate constraints → filter infeasible alternatives
5. Evaluate economics → rank by contribution margin
6. Generate recommendation with evidence and risk analysis
7. Output via API, CLI, or Markdown report

## Integration Points

### Upstream (consumed by Fusion)
- Kubernetes API (read-only inventory)
- Prometheus (GPU metrics and telemetry)
- OpenCost (cost allocation data)
- NVIDIA Dynamo (Planner metrics)
- KAI Scheduler (placement and capability data)

### Downstream (Fusion outputs)
- Tesseract Foundry (performance profiles)
- Tesseract Ignite (capacity lifecycle decisions)

All integrations are adapter-based and versioned. Missing upstream dependencies fall back to fixture data.
