# 07 — Chaos & Failure Recovery

Evidence: `reports/raw/cpu_phases.json` (phase G) + Phase D fault injection.

## Failure matrix

| Fault injected | Outcome | Capacity released | Invariant OK | Audit trail |
|---|---|---|---|---|
| Foundry unavailable | **ACTIVE** (conservative defaults) | n/a | YES | `foundry_unavailable`, `workload_active` |
| Adaptive unavailable | **ACTIVE** (declared precision) | n/a | YES | — |
| Fusion unavailable | **FAILED** | YES | YES | `operation_failed`, `compensation` |
| Ignite unavailable | **FAILED** | YES | YES | `operation_failed`, `compensation` |
| Readiness timeout | **FAILED** | YES | YES | `operation_failed`, `compensation` |
| Reservation expiration | **RELEASED** (reconciler) | YES | YES | `expire_reservation` |

## Properties verified

1. No silent state corruption — every terminal op has a recorded error.
2. No resource over-allocation — capacity invariant holds in all cases.
3. No unbounded retry loops — submission is bounded; failures terminate.
4. No permanent orphaned reservation — compensation/reconciler releases.
5. No routing to unready replicas — readiness gate before ACTIVE.
6. Clear error reporting — `op.error` populated; audit events recorded.
7. Recovery or safe terminal failure — FAILED is terminal and compensated.
8. Audit trail preserved — lifecycle + audit rows present.

## Recovery times

Compensation (release) completes in the same request; measured rollback
completion p50 ≈ 0.91 ms (Phase J). No asynchronous reconciliation wait was
required for the tested faults.

## Not run (require a cluster / longer sessions)

Kubernetes API timeout, node loss, network partition, failed image pull, replica
crash, database corruption simulation, out-of-order lifecycle events across
process restarts, incomplete rollback across process restarts. These are listed
in `reports/FINAL_BUG_REGISTER.md` as pending Phase E work.

## Verdict

CPU failure-injection matrix: **PASS**. Cluster-level chaos: **NOT TESTED**.
