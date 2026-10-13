# Full-Stack E2E

## Real cross-product integration (CPU, live services)
Evidence: `reports/raw/live_integration.json` (re-run after the Foundry fixes).

* **Foundry** real HTTP service (`foundry.api:app` @ :8001): `/healthz` 200,
  `/metrics` 200, `/v1/schemas/validate` 200, `/v1/workloads/validate` 200,
  **`/v1/services/recommend` 200** (was 500), **`/v1/profiling/jobs` 200** (was 500).
  Library path (candidate generation + fixture profiler + `export_performance_profile`)
  produces a valid `tesseract.performance-profile.v1` (synthetic).
* **Fusion** real HTTP service (`fusion.api.app:create_app` @ :8002): inventory,
  workloads, `/v1/optimization/evaluate`, recommendation — all 200.
* **Platform** coordinator + real `FusionAdapter` over HTTP: full lifecycle
  `PROFILING → RESERVED → DEPLOYING → ACTIVE`, reservation/deployment ACTIVE,
  Ignite manifests built, release verified, correlation IDs preserved.

## Platform test suite
`81 passed` (unit + contract + integration + security).

## Kubernetes end-to-end
**NOT TESTED** on this host (no cluster). On the Ignite side, the GitHub-hosted
`k8s-e2e` suite now passes scenarios 1–7 of 9 (see `IGNITE_PR1_STATUS.md`); 8–9
remain and the required check is still red.

## Verdict
* Real cross-product integration (Foundry+Fusion+Platform): **PASS**.
* Full-stack **Kubernetes** E2E: **NOT TESTED** (host) / **partial** (Ignite CI).
