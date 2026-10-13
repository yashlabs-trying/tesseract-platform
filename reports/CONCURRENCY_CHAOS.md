# Concurrency & Chaos

## Platform (CPU, in-process) — evidence `reports/raw/cpu_phases.json`
* Concurrency levels 1/10/25/100/250/500: all succeeded, 0 failures; p50 0.84→136 ms.
* Reservation invariant `committed + active_reserved ≤ total` holds under 100
  randomized concurrent requests; no duplicate reservations; every ACTIVE op has
  a reservation and deployment.
* Chaos matrix: Foundry unavailable → ACTIVE (conservative defaults); Adaptive
  unavailable → ACTIVE; Fusion/Ignite/readiness faults → FAILED with capacity
  released and audit trail. Repeated release idempotent; stale reservation
  expired by the reconciler.
* Short soak: 100 reservation/release cycles, ledger used=0, RSS delta 0.

## Kubernetes concurrency & chaos
**NOT TESTED** on this host (no cluster). The Ignite `k8s-e2e` suite includes a
100-concurrent-cold-request scenario (E2E 9) which currently **fails**, and an
operator-restart scenario (E2E 5) which **passes**.

## 30-minute soak
**NOT RUN** in this session (time-bounded). Command prepared in
`scripts/run_full_stack_validation.sh`.

## Verdict
CPU concurrency/chaos: **PASS**. Kubernetes concurrency/chaos: **NOT TESTED /
partial**. No production load capacity is claimed.
