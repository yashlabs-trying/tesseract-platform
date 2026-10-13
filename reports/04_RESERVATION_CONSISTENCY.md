# 04 — Reservation Consistency

Evidence: `reports/raw/cpu_phases.json` (phase D), plus
`tests/integration/test_reservation_invariants.py` (8 tests) and
`tesseract_platform/state/capacity.py`.

## New platform capability

A thread-safe `CapacityLedger` enforces, per resource pool::

    committed + active_reserved <= total_usable_capacity

It is wired into the coordinator's reserve/commit/compensate path, so capacity is
held from reservation to release and freed on failure. Enabled via
`TESSERACT_TOTAL_USABLE_CAPACITY` (>0). Units = `Capacity.gpu_count`.

## Randomized concurrency result

100 concurrent submissions (random 1–3 GPU units each) against capacity 20:

| Invariant | Result |
|---|---|
| `committed + active_reserved <= total` | **TRUE** (committed=20, available=0) |
| No duplicate reservations | TRUE |
| Every ACTIVE op has a reservation | TRUE |
| Every ACTIVE op has a deployment | TRUE |

12 requests succeeded (used=20 units), 88 failed cleanly with capacity exhausted.
No over-allocation.

## Scenario matrix

| Scenario | Result |
|---|---|
| Single reservation | PASS |
| Concurrent reservations | PASS (invariant holds) |
| Duplicate requests (idempotency key) | PASS — one reservation only |
| Reservation expiration (stale) | PASS — reconciler expires → RELEASED |
| Failed deployment | PASS — capacity released |
| Fusion unavailable | PASS — FAILED, capacity released |
| Ignite unavailable | PASS — FAILED, capacity released |
| Readiness timeout | PASS — FAILED, capacity released |
| Repeated release | PASS — idempotent, used=0 |
| Released owner cannot commit | PASS — raises |
| Concurrent rollback | PASS — compensation idempotent |
| Capacity exhaustion | PASS — CapacityExhausted, invariant holds |

## Additional invariants verified

* One operation ID cannot create duplicate reservations (unique key + ledger
  idempotency per owner).
* Every reservation has an accountable owner (operation ID).
* Every successful deployment has a traceable reservation (`reservation_id` on
  the deployment).
* A failed deployment does not retain capacity (compensation releases it).
* Recovery does not create duplicate deployments (idempotent operation reuse).
* Readiness is not inferred from resource creation alone (explicit `await_ready`).

## SQLite / multi-replica note

The state store uses a single SQLite connection with a process lock and WAL. The
`CapacityLedger` is process-local. **The control plane must run as a single
writer** with SQLite. For multiple coordinator replicas, migrate to Postgres and
back the ledger with a shared row (or advisory locks). This is documented in
`docs/ARCHITECTURE.md`; no multi-replica mode is claimed.

## Verdict

Reservation invariants hold under randomized concurrency and fault injection.
**PASS.**
