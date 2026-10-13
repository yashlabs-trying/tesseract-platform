# Integration Contracts

Every cross-product exchange is a versioned JSON Schema under `contracts/`,
validated in-process by `tesseract_platform.contracts`. Python models in
`tesseract_platform/models.py` mirror the schemas.

| Contract | `$id` | Producer | Consumer |
|---|---|---|---|
| `workload.schema.json` | `tesseract.platform.workload.v1` | client → platform | coordinator |
| `reservation.schema.json` | `tesseract.platform.reservation.v1` | platform | coordinator/reconciler |
| `deployment.schema.json` | `tesseract.platform.deployment.v1` | platform | Ignite |
| `telemetry.schema.json` | `tesseract.platform.telemetry.v1` | Ignite / Foundry | platform |
| `lifecycle.schema.json` | `tesseract.platform.lifecycle.v1` | platform | audit |
| `rollback.schema.json` | `tesseract.platform.rollback.v1` | platform | operators |
| `precision_profile.schema.json` | `tesseract.platform.precision-profile.v1` | Adaptive | platform |
| `foundry-performance-profile.schema.json` | `tesseract.performance-profile.v1` | Foundry | platform (vendored copy) |

## Integration A — Foundry → Platform

Foundry produces `tesseract.performance-profile.v1` documents
(`schemas/tesseract/performance-profile-v1.json` in the Foundry repo).
`integrations/foundry/adapter.py::map_performance_profile` maps them to a
platform `WorkloadProfile` **without changing the evidence level**.

### Evidence vocabulary

The platform adopts Foundry's vocabulary everywhere:

`tesseract.platform.Evidence = {synthetic, estimated, measured, validated}`

Fusion's richer vocabulary is collapsed on import:

| Fusion `EvidenceType` | Platform `Evidence` |
|---|---|
| `measured` | `measured` |
| `estimated` | `estimated` |
| `simulated` | `synthetic` |
| `extrapolated` | `estimated` |
| `unknown` | `estimated` |

### Field mapping (Foundry → platform)

| Foundry field | Platform field |
|---|---|
| `output_tokens_per_second` | `WorkloadProfile.output_tokens_per_second` |
| `completed_requests_per_second` | `WorkloadProfile.completed_requests_per_second` |
| `input_tokens_per_second` | `WorkloadProfile.input_tokens_per_second` |
| `evidence_type` | `WorkloadProfile.evidence` |
| `workload_concurrency` | `WorkloadProfile.concurrency` |
| `ttft_p95_ms`, `itl_p95_ms`, `request_latency_p95_ms` | same |

> Adaptive's `integration/foundry.py` reads a `throughput_tps` key that Foundry
> does not emit. The platform treats `output_tokens_per_second` as the
> throughput signal and keeps `completed_requests_per_second` separate; this
> discrepancy is flagged for reconciliation with the Adaptive repository.

### Live retrieval

`FoundryAdapter.fetch_profile_document(profile_id)` calls a configurable
endpoint template (default `/v1/performance-profiles/{profile_id}`). Foundry's
current API does not publish that route; until it does, the adapter is used in
**document-injection** mode (documents supplied by export, fixtures or tests).
This is an explicit, documented gap — the adapter never invents an endpoint
response.

## Integration B — Fusion → Ignite (reservation protocol)

Fusion returns a `FusionRecommendation`. `map_recommendation` converts the
`recommended_plan` into a `PlacementResult` (slots + capacity). The platform
then owns the reservation lifecycle (see `docs/ARCHITECTURE.md`).

* Fusion's `execution_eligibility=false` → `IntegrationError` → operation
  `FAILED`; **no** capacity is reserved.
* Fusion is advisory and has no native release endpoint today;
  `FusionAdapter.release` is a logical no-op and the platform's durable ledger
  is authoritative. When Fusion exposes a release route it is used automatically
  (`release_path` configurable).

Ignite is driven via the `ReplicaGroup` CRD (`kubbernetd.io/v1`), whose schema
is mirrored in `integrations/ignite/adapter.py::build_replicagroup`. A model
`Deployment` is created alongside it (`build_model_deployment`).

State mapping:

| Platform state | Ignite / Kubernetes signal |
|---|---|
| `DEPLOYING` | `ReplicaGroup` created, `Deployment` created |
| `READY` | `ReplicaGroup.status.phase ∈ {RUNNING, WARMING}` and `readyReplicas > 0` |
| `ACTIVE` | endpoint routable |
| `RELEASING`/`RELEASED` | `ReplicaGroup` + `Deployment` deleted |

## Integration C — Ignite → Platform telemetry

Ignite emits `tesseract.platform.telemetry.v1` events (cold start, deployment
startup, readiness, queue wait, request, scale event, release). The platform's
`TelemetryCollector`:

* validates every event against the contract;
* preserves `correlation_id` across the lifecycle;
* **excludes failed requests** (`success is False`) from throughput and latency
  percentiles while still counting them in the error rate.

## Integration D — Foundry → Optimization controller

`OptimizationController.evaluate(workload_id, slo)` reads aggregated telemetry
and compares measured TTFT p95 to the objective with hysteresis. Modes and
safety rails are documented in `docs/ARCHITECTURE.md`.

## Versioning policy

* Schema `$id` is immutable; breaking changes require a new `.vN`.
* `schema_version` is a required, constant field on every document.
* The platform refuses documents whose `schema_version` it does not recognise.
