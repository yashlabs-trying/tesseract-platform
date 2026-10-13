# Release Candidate Scorecard

Release candidate branch: `release/pre-gpu-candidate` (platform).
Pinned product revisions in `platform.manifest.yaml`.

| Gate | Required | Result |
|---|---|---|
| Foundry HTTP defects | FIXED | **PASS** — F-01/F-02/F-03 fixed; PR #1 CLEAN (CI pass) |
| Foundry regression | PASS | **PASS** — 47 passed, 4 skipped (GPU) |
| Fusion regression | PASS | **PASS** — 183 passed, 6 skipped (K8s) |
| Ignite regression | PASS | **PASS** — 88 passed |
| Adaptive regression | PASS | **PASS** — 254 passed, 2 skipped (CUDA) |
| Platform regression | PASS | **PASS** — 81 passed |
| Real cross-product integration | PASS | **PASS (CPU)** — live Foundry+Fusion+Platform |
| Mac Kubernetes lifecycle | PASS | **NOT TESTED** — Mac unprovisioned |
| Reservation consistency | PASS | **PASS** — invariant holds under randomized concurrency |
| Concurrency and chaos | PASS | **PASS (CPU)**; 30-min K8s soak NOT RUN |
| Adaptive quality safety | PASS | **PASS** |
| Security | PASS | **PASS** — 0 High/Medium; mypy/ruff clean |
| Ignite required CI | PASS | **PASS** — `k8s-e2e` green (9/9 E2E scenarios); PR #1 CLEAN/MERGEABLE |
| GPU benchmark harness | READY | **PASS (prepared)** |
| NVIDIA hardware validation | NOT TESTED | **NOT TESTED** |

## Readiness verdicts (independent)

1. **Component software correctness — PASS** (all five suites green; Foundry HTTP
   fixed; Ignite E2E green).
2. **Full-stack integration — PASS (CPU/CI)**. Live Foundry+Fusion+Platform
   lifecycle ACTIVE; Ignite Kubernetes E2E passes all 9 scenarios in GitHub CI.
3. **Mac M5 Kubernetes validation — NOT TESTED** (Mac unprovisioned).
4. **NVIDIA GPU production readiness — NOT TESTED.**

## Release-candidate status
**Software gates PASS.** The only remaining pre-GPU items are environmental:
run the Kubernetes lifecycle on the provisioned Mac and the 30-minute soak. Not
merged to `main` (awaiting authorization).
