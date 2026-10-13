# Release Candidate Scorecard

Release candidate branch: `release/pre-gpu-candidate` (platform).
Pinned product revisions in `platform.manifest.yaml`.

| Gate | Required | Result |
|---|---|---|
| Foundry HTTP defects | FIXED | **PASS** — F-01/F-02/F-03 fixed on `fix/pre-gpu-api-blockers`, PR #1 CI green |
| Foundry regression | PASS | **PASS** — 47 passed, 4 skipped (GPU) |
| Fusion regression | PASS | **PASS** — 183 passed, 6 skipped (K8s) |
| Ignite regression | PASS | **PASS** — 85 passed |
| Adaptive regression | PASS | **PASS** — 254 passed, 2 skipped (CUDA) |
| Platform regression | PASS | **PASS** — 81 passed |
| Real cross-product integration | PASS | **PASS (CPU)** — live Foundry+Fusion+Platform |
| Mac Kubernetes lifecycle | PASS | **NOT TESTED** — Mac unprovisioned |
| Reservation consistency | PASS | **PASS** — invariant holds under randomized concurrency |
| Concurrency and chaos | PASS | **PASS (CPU)**; K8s 30-min soak NOT RUN |
| Adaptive quality safety | PASS | **PASS** |
| Security | PASS | **PASS** — 0 High/Medium; mypy/ruff clean |
| Ignite required CI | PASS | **FAIL** — `k8s-e2e` red (7/9 E2E pass; 8–9 fail) |
| GPU benchmark harness | READY | **PASS (prepared)** |
| NVIDIA hardware validation | NOT TESTED | **NOT TESTED** |

## Readiness verdicts (independent)

1. **Component software correctness — PASS** (all suites green; Foundry HTTP fixed).
2. **Full-stack integration — PARTIAL** (Foundry+Fusion+Platform live; Ignite K8s
   E2E partial; host K8s NOT TESTED).
3. **Mac M5 Kubernetes validation — NOT TESTED.**
4. **NVIDIA GPU production readiness — NOT TESTED.**

## Release-candidate status
**BLOCKED** on the Ignite `k8s-e2e` required check (E2E 8–9) and on the Mac
Kubernetes lifecycle. Not merged to `main`.
