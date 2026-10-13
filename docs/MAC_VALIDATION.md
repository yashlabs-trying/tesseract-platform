# Mac M5 Validation

**Status: PENDING.** This document is a plan; no Mac/Apple-Silicon Kubernetes
run has been performed yet. Nothing here is claimed as validated.

## Target environment

* Apple Silicon Mac (M5), macOS 26.6.2 (observed at `192.168.1.4`).
* Local Kubernetes via `kind` (ARM64 images) or `k3s`.
* The platform and product services run as ARM64 containers.

## Scope

1. **Product compatibility** — package versions, schema compatibility, API
   contracts, Kubernetes resources, auth, RBAC, controller startup, service
   discovery, shutdown/cleanup.
2. **Full request lifecycle** — cold, warm, concurrent, multi-model requests;
   placement, reservation, deploy, readiness, routing, measurement, release,
   rollback.
3. **Concurrency** — 1, 5, 25, 100, 250, 500 concurrent requests with small
   fixtures; measure errors, queue time, latency, memory, CPU.
4. **Multi-model placement** — 0.5B/1B/3B/7B/13B/70B profiles; competing
   placements and insufficient-capacity conditions; assert the scheduler never
   over-allocates.
5. **Failure injection** — Fusion crash after reservation, Ignite crash during
   deployment, Foundry unavailable, K8s API timeout, readiness timeout,
   duplicate deployment event, stale reservation, concurrent rollback, network
   interruption, node failure, invalid config, resource exhaustion.
6. **Security** — no command injection, safe URL handling, least-privilege RBAC,
   safe error messages, dependency scanning, static analysis, input validation.
7. **Stress/soak** — 30-minute sustained workload, repeated cold starts and
   scale up/down, multi-model contention, scheduler restart under load.

## Tooling (to be run on the Mac)

```bash
scripts/setup_mac.sh        # install kind/kubectl/helm/docker (ARM64)
scripts/create_cluster.sh   # create the local cluster
scripts/deploy_stack.sh     # deploy Foundry, Fusion, Ignite, Platform
scripts/run_tests.sh        # unit + contract + integration + k8s e2e
scripts/collect_reports.sh  # gather JSON + Markdown reports
scripts/cleanup.sh          # uninstall and delete the cluster
```

## Explicit non-claims

* Passing simulations are **not** hardware validation.
* No production-readiness claim is made from Mac results.
* GPU-specific tests are skipped here and covered by `docs/GPU_VALIDATION.md`.

## Exit criteria

* `scripts/run_tests.sh` green, with every skip explicitly listed.
* Failure-injection matrix produces a safe, observable recovery for each fault.
* Reservation consistency holds: no double reservations, no leaked capacity.
