# Operations Guide

## Install

```bash
pip install -e ".[test]"
```

## Configure

Copy `.env.example` to `.env` and set product URLs and credentials. No secrets
are ever committed. Key variables:

| Variable | Meaning |
|---|---|
| `TESSERACT_STATE_DSN` | durable state (`sqlite:///./var/platform.db`) |
| `TESSERACT_FOUNDRY_URL` | Foundry base URL |
| `TESSERACT_FUSION_URL` | Fusion base URL |
| `TESSERACT_IGNITE_KUBECONFIG` | kubeconfig used to drive the `ReplicaGroup` CRD |
| `TESSERACT_IGNITE_NAMESPACE` | namespace for deployments (default `tesseract`) |
| `TESSERACT_CONTROLLER_MODE` | `observe_only` (default) / `recommendation` / `controlled_action` |
| `TESSERACT_CONTROLLER_ACTION_BUDGET` | max controller actions (default 0) |
| `TESSERACT_ADAPTIVE_ENABLED` | enable the optional Adaptive provider (default false) |

## Run

```bash
tesseract-platform serve              # control-plane API (default :8088)
tesseract-platform reconcile          # one reconciliation pass
tesseract-platform validate-contracts # validate vendored schemas
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/workloads` | submit a workload (idempotent) |
| GET | `/v1/workloads/{operation_id}` | inspect an operation |
| GET | `/v1/workloads/{operation_id}/lifecycle` | transition audit trail |
| POST | `/v1/workloads/{operation_id}/release` | graceful release |
| POST | `/v1/workloads/{operation_id}/rollback` | rollback / compensation |
| POST | `/v1/telemetry` | ingest a telemetry event |
| GET | `/v1/telemetry/{workload_id}/summary` | aggregated metrics |
| POST | `/v1/reconcile` | run reconciliation |
| POST | `/v1/controller/evaluate` | evaluate the optimization controller |
| GET | `/healthz`, `/readyz` | liveness / readiness (readiness lists wired products) |

## Operating the controller safely

1. Start in `observe_only` and watch `/v1/controller/evaluate` decisions.
2. Move to `recommendation` to surface recommendations without acting.
3. Only enable `controlled_action` after safety and rollback tests pass, and set
   a small `TESSERACT_CONTROLLER_ACTION_BUDGET` with a cooldown.

## Backup and state

The SQLite database (`var/platform.db`) holds operations, the reservation
ledger, the lifecycle/audit log and telemetry. Back it up before upgrades. The
store interface is swappable for Postgres.

## Troubleshooting

* **Operation `FAILED` with `reservation_expired`** — the reconciler released a
  reservation that was not committed in time. Check Fusion/Ignite latency.
* **`ReadinessTimeout`** — Ignite did not report ready replicas. Inspect the
  `ReplicaGroup` status and the model `Deployment`.
* **`/readyz` shows a product `false`** — that adapter is not configured; the
  platform will apply conservative defaults (Foundry) or fail the operation
  (Fusion/Ignite) rather than guess.
* **`integration_unavailable` audit events** — the corresponding product was
  unreachable; the operation either used conservative defaults or failed safely.
