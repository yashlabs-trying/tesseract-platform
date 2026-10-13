# Architecture

## Goal

One functioning system that coordinates profiling (Foundry), GPU placement and
capacity reservation (Fusion), model deployment and routing (Ignite), and
optionally adaptive quantization (Adaptive).

```
                     CUSTOMER REQUEST
                            |
                            v
                  +-------------------+
                  |  TESSERACT        |
                  |  PLATFORM         |
                  |  (coordinator)    |
                  +-------------------+
                            |
             +--------------+--------------+
             v                             v
   Workload Requirements            Reservation ledger
             |                      (durable state machine)
             v
        +----------+   profile   +----------+   placement   +----------+
        | FOUNDRY  |------------>|  FUSION  |-------------->|  IGNITE  |
        | profiling|             | reserve  |               | deploy / |
        +----------+             +----------+               | wake /   |
             ^                                             | route    |
             |            runtime telemetry                +----------+
             +---------------------------------------------+    |
                                                              v
                                                        MODEL SERVER
                                                              |
                                                              v
                                                     +-----------------+
                                                     | OPTIMIZATION    |
                                                     | CONTROLLER      |
                                                     | observe->act    |
                                                     +-----------------+
```

## Components and ownership

| Component | Owns | Package |
|---|---|---|
| **Coordinator** | Workload identity, idempotency, the reservation/deployment lifecycle, compensation | `tesseract_platform.coordinator` |
| **State store** | Durable operations, reservations, deployments, lifecycle/audit log, telemetry | `tesseract_platform.state` |
| **Policies** | Input validation, evidence policy (conservative defaults vs reject) | `tesseract_platform.policies` |
| **Observability** | Telemetry aggregation, optimization controller | `tesseract_platform.observability` |
| **API** | Control-plane REST surface | `tesseract_platform.api` |
| **Foundry** | Profiling / measurement; emits `tesseract.performance-profile.v1` | external |
| **Fusion** | Inventory, placement decision, advisory reservation | external |
| **Ignite** | `ReplicaGroup` CRD, Deployment lifecycle, proxy routing, telemetry | external |
| **Adaptive** | Precision profiles (optional) | external |

The platform deliberately does **not** create another Kubernetes controller for
pod lifecycle. Ignite's operator already reconciles `ReplicaGroup` objects; the
platform's reconciler only repairs *platform-level* drift (expired reservations,
stuck deployments) by driving Fusion and Ignite through their APIs.

## Why a state machine, not a transaction

Fusion capacity lives in Fusion's database; the model server lives in
Kubernetes. No single transaction spans both. The platform therefore keeps an
authoritative, durable reservation ledger and models every step as a validated
state transition with an explicit **compensating action** on failure.

Reservation / operation lifecycle:

```
PENDING -> RESERVING -> RESERVED -> DEPLOYING -> READY -> ACTIVE
      \-> RELEASING -> RELEASED          \-> FAILED -> RELEASING -> RELEASED
```

* `PENDING` — operation accepted, idempotency recorded.
* `RESERVING` — asking Fusion for capacity.
* `RESERVED` — capacity committed to the ledger (revision bumped).
* `DEPLOYING` — Ignite creating the `ReplicaGroup` + `Deployment`.
* `READY` — Ignite reports ready replicas.
* `ACTIVE` — traffic routed; the endpoint is live.
* `RELEASING` / `RELEASED` — compensating release of Ignite then Fusion.
* `FAILED` — terminal until compensated; still releasable.

Every transition is appended to the `lifecycle` table (append-only audit).

## Request flow (`WorkloadCoordinator.submit`)

1. **Validate** the request (policy). Reject malformed input early.
2. **Idempotency** — `create_operation` keyed on `idempotency_key`. A replay
   returns the existing operation and never reserves twice.
3. **Profile** — call Foundry. If unavailable, apply explicit conservative
   defaults (or reject, per policy). Never fabricate a *measured* profile.
4. **Precision** (optional) — ask Adaptive for a profile; disabled by default.
5. **Reserve** — Fusion returns a placement; the platform commits a reservation.
6. **Deploy** — Ignite creates the `ReplicaGroup` + model `Deployment`.
7. **Await readiness** — poll until Ignite reports ready replicas. Traffic is
   routed only after readiness.
8. **Activate** — mark `ACTIVE`, record telemetry.
9. **Failure** at any step → `_compensate` releases Ignite and Fusion resources
   and records a `RollbackPlan`.

## Reconciler

`Reconciler.reconcile()` performs one pass:

* releases reservations whose `expires_at` has passed and were never committed;
* rolls back deployments stuck in `DEPLOYING` past the readiness timeout.

It is safe to run repeatedly and never leaves capacity reserved without a
corresponding deployment.

## Optimization controller

`OptimizationController` compares measured TTFT p95 against the SLO with
hysteresis, and depending on mode:

* `observe_only` (default) — records a decision, never acts;
* `recommendation` — emits `recommend_reconfigure`;
* `controlled_action` — may emit `reconfigure`, bounded by a **cooldown** and an
  **action budget** (default 0).

The platform always starts in `observe_only`.

## Configuration

Environment-driven (`tesseract_platform/config.py`); see `.env.example`. With no
product URLs configured the platform still runs (providers are `None`) and tests
exercise the real coordinator with in-process fakes.

## Known limitations

See `docs/GPU_VALIDATION.md` and the risk register in the platform README
roadmap. Kubernetes/GPU validation is pending.
