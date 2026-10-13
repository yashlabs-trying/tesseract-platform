# 02 — Live Integration

Evidence: `reports/raw/live_integration.json`. Real services were started with
`uvicorn` (Foundry `foundry.api:app` on :8001, Fusion `fusion.api.app:create_app`
on :8002) and driven over HTTP with `httpx`.

## Foundry — real HTTP service

| Endpoint | Status | Notes |
|---|---|---|
| `GET /healthz` | 200 | |
| `GET /metrics` | 200 | |
| `POST /v1/schemas/validate` | 200 | performance-profile validated |
| `POST /v1/workloads/validate` | 200 | workload validated |
| `POST /v1/services/recommend` | **500** | defect **F-01** |
| `POST /v1/profiling/jobs` | **500** | defect **F-02** |

Foundry's **library** interface works and was used successfully: candidate
generation (`foundry.candidates.get_provider`), the fixture profiler
(`foundry.benchmarking.get_runner`), and `foundry.schemas.export_performance_profile`
produced a valid `tesseract.performance-profile.v1` document with
`evidence_type="synthetic"` (honest fixture labelling).

## Fusion — real HTTP service

| Endpoint | Status |
|---|---|
| `GET /healthz` | 200 |
| `POST /v1/inventory/import` | 200 |
| `GET /v1/inventory` | 200 |
| `POST /v1/workloads/import` | 200 |
| `GET /v1/workloads` | 200 |
| `POST /v1/optimization/evaluate` | 200 |
| `GET /v1/optimization/recommendations/{id}` | 200 |

Fusion returns a real `FusionRecommendation` with a `recommended_plan`
(`execution_eligibility=false`, `approval_status=advisory_only`).

## Platform lifecycle (real coordinator + real Fusion over HTTP)

Workload `phase-b-wl`, model `Qwen/Qwen3-0.6B`:

```
PROFILING -> RESERVED -> DEPLOYING -> ACTIVE
```

* Final state **ACTIVE**; reservation **ACTIVE**; deployment **ACTIVE**.
* Profile evidence: **synthetic** (fixture), source `test_fixture` — not presented
  as measured.
* Ignite `ReplicaGroup` + `Deployment` manifests built and released.
* Correlation IDs and full transition history recorded.

## Defect found and fixed during Phase B

**P-01 (platform, fixed):** `FusionAdapter.map_recommendation` rejected
advisory-only recommendations (`execution_eligibility=false`). The platform is
the execution authority, so advisory plans must be accepted. Default changed to
`require_execution_eligible=False`; regression test
`test_map_recommendation_accepts_advisory_plan` added. Platform tests: 63 → pass.

## Explicit boundaries

* **Ignite Kubernetes deployment is NOT exercised here.** There is no cluster on
  this host, so the deployment provider builds the real Ignite manifests and
  stubs readiness at the Kubernetes boundary. This is reported, not hidden.
  Real Ignite deployment is Phase E (NOT TESTED).
* Foundry's two broken HTTP endpoints (F-01, F-02) are product defects; the
  platform integrates through Foundry's supported library interface instead.
  Product repositories were **not modified** (per campaign decision).

## Verdict

| Component | Real interface integrated |
|---|---|
| Foundry | Library: YES. HTTP: partial (2 endpoints 500) |
| Fusion | YES (all endpoints) |
| Platform | YES |
| Ignite | Manifests only (K8s boundary stubbed) |

**Full-stack integration: PARTIAL PASS** — Foundry+Fusion+Platform live; Ignite
Kubernetes lifecycle pending Phase E.
