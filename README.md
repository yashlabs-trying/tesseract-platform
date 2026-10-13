# Tesseract Platform

Unified orchestration and optimization layer connecting three Tesseract
products into **one functioning system**:

| Product | Repository | Role |
|---|---|---|
| **Foundry** | `yashlabs-trying/tesseract-foundry` | Profiling & performance measurement |
| **Fusion** | `yashlabs-trying/tesseract-fusion` | GPU placement & capacity reservation |
| **Ignite** | `yashlabs-trying/kubbernetd-saver` | Kubernetes deployment, cold-start & routing |
| **Adaptive** *(future)* | `yashlabs-trying/tesseract-adaptive` | Adaptive quantization — optional, disabled |

The platform owns the **workload lifecycle**: it turns a customer request into a
profiled workload, obtains a placement and reservation from Fusion, drives a
deployment through Ignite, routes traffic only to ready endpoints, collects
runtime telemetry back into Foundry, and reconciles/rolls back on failure.

```
CUSTOMER REQUEST → PLATFORM → Foundry(profile) → Fusion(placement+reservation)
                → Ignite(deploy/wake/route) → MODEL SERVER → Foundry(telemetry)
                → OPTIMIZATION CONTROLLER (observe → recommend → act)
```

## Status

Milestone 1 (audit + contracts) and Milestone 2 (real integration, coordinator,
reservation lifecycle, telemetry, rollback) are implemented and covered by
unit, contract and in-process integration tests. Kubernetes and GPU validation
are **pending** — see `docs/MAC_VALIDATION.md` and `docs/GPU_VALIDATION.md`.

The pinned product revisions live in [`platform.manifest.yaml`](platform.manifest.yaml).

## Quick start

```bash
pip install -e ".[test]"
pytest
```

Run the control-plane API:

```bash
tesseract-platform serve            # or: uvicorn tesseract_platform.api.app:app
```

## Layout

```
tesseract_platform/     Control plane: coordinator, reservation state machine,
                        durable state store, policies, observability/controller, API.
integrations/           Versioned adapters: foundry, fusion, ignite, adaptive.
contracts/              JSON Schema contracts shared across products.
deployments/            docker-compose, Kubernetes, Helm, kind.
benchmarks/             Benchmark scenarios, baselines and reports.
tests/                  unit, contracts, integration, e2e, concurrency, chaos, performance.
docs/                   Architecture, integration contracts, audit, validation plans.
```

> **Naming note.** The control-plane package is `tesseract_platform/`, not
> `platform/`, because a top-level package named `platform` would shadow the
> Python standard-library `platform` module (imported by httpx/uvicorn).

## Design principles

- **No product code is vendored.** Integration is through versioned contracts
  and pinned checkouts.
- **Never assume** Foundry emits placement or Ignite accepts reservations —
  those contracts are defined here.
- **Durable state machine**, not a single database transaction, spans Fusion and
  Kubernetes. Failures trigger explicit compensating actions.
- **Adaptive is optional.** The platform passes contract tests proving it works
  with Adaptive absent.

## License

MIT
