# Tesseract Fusion — Architecture Audit

**Audit date:** 2026-10-12
**Package:** `tesseract-fusion` v0.1.0 (optimizer provenance reports `0.3.0` for `GlobalAllocator`, `0.2.0` for `OptimizationEngine`)
**Source revision:** transferred from the Windows/WSL repository at git commit `4539cb1`, then audit fixes applied on the Mac (not a git checkout on the audit host)
**Audit host:** Apple Silicon Mac (M5), macOS 26.6.2 (build 25G83), arm64
**Python:** 3.12.15 (venv at `~/tesseract-fusion/.venv`)
**Docker:** 29.5.2 (Docker Engine Community) via Colima — 4 CPU, 6 GiB RAM, 60 GiB disk
**Kubernetes:** kind v1.37.0, cluster `fusion-audit`, single control-plane node
**Pre-existing cluster preserved untouched:** `tesseract-ignite-gate2`

---

## 1. Overview

Fusion is an **advisory-only GPU fleet economics and optimization service**. It reads
inventory, telemetry and cost data from external systems, computes an economically
optimal allocation of workloads to GPU resources, and emits a *recommendation* for a
human operator. It does **not** schedule pods, does **not** mutate cluster state, and
does **not** replace or extend `kube-scheduler`.

The service is delivered as:

- a FastAPI application (`fusion.api.app:create_app`),
- a Typer CLI (`fusion.cli.main:app`, console script `fusion`),
- a set of pure-Python engines (optimizer, economics, simulator, policy),
- a protocol-based adapter layer (Kubernetes, Prometheus, OpenCost, Dynamo, KAI, Ignite).

The optimization core is deterministic. There is **no reinforcement learning, no
learned policy, and no training loop** anywhere in the repository (verified by
`tests/audit/test_rl_readiness.py`).

---

## 2. Deployment Model — Advisory, Not a Scheduler

This is the single most important architectural fact and it is stated explicitly so it
cannot be mistaken:

| Question | Answer | Evidence |
|----------|--------|----------|
| Is Fusion a Kubernetes scheduler? | **No** | No `SchedulerPlugin`, extender, or webhook exists; `deploy/` contains only API deployment + read-only RBAC |
| Is Fusion a scheduler extender / admission webhook? | **No** | No HTTP extender endpoint, no `MutatingWebhookConfiguration`/`ValidatingWebhookConfiguration` |
| Does Fusion replace `kube-scheduler`? | **No** | It consumes node state read-only via `KubernetesInventoryAdapter` |
| Does Fusion mutate cluster state? | **No** | All recommendations set `approval_status=ADVISORY_ONLY`, `execution_eligibility=False` |
| What RBAC does it request? | **Read-only** | `deploy/rbac/cluster-role.yaml` (read nodes/pods/namespaces) |

Fusion's Kubernetes surface is a **read-only inventory adapter**
(`src/fusion/adapters/kubernetes.py`), which lists nodes and converts each
`nvidia.com/gpu` capacity into a `GPUInventory` record. Recommendations reference only
observed `gpu_id` values (enforced by
`tests/audit/test_k8s_integration.py::test_recommendation_uses_observed_state`).

**Placement execution is out of scope.** The pipeline terminates at a recommendation;
actually applying it to Kubernetes is a separate, not-yet-implemented concern.

---

## 3. Component Inventory

| Layer | Component | File | Responsibility |
|-------|-----------|------|----------------|
| API | `create_app` | `src/fusion/api/app.py` | FastAPI service, in-memory stores |
| CLI | Typer app | `src/fusion/cli/main.py` | `inventory`, `optimize`, `simulate`, `economics`, `serve` |
| Config | `Settings` | `src/fusion/config/settings.py` | pydantic-settings; `api_key` is `SecretStr` |
| Models | Pydantic types | `src/fusion/models/types.py` | Domain model + `ge=0` constraints |
| Models | SQLAlchemy ORM | `src/fusion/models/orm.py` | 9 ORM tables (defined; API uses in-memory dicts) |
| DB | session | `src/fusion/database/session.py` | `init_db`, `get_db` |
| Inventory | `InventoryService` | `src/fusion/inventory/provider.py` | Caching service over `InventoryProvider` protocol |
| Inventory | `FixtureInventoryAdapter` | `src/fusion/inventory/fixture_adapter.py` | YAML/dict inventory |
| Demand | `DemandForecaster` | `src/fusion/demand/forecaster.py` | Moving-average forecast |
| Demand | `WorkloadProfileBuilder` | `src/fusion/demand/profile.py` | Build profiles from demand history |
| Economics | `CostAccountingEngine`, `RevenueCalculator`, `MarginAnalyzer` | `src/fusion/economics/accounting.py` | GPU/energy/hosting cost + margin |
| Optimizer | `CandidateGenerator` | `src/fusion/optimizer/candidate.py` | Enumerate placement alternatives |
| Optimizer | `AlternativeEvaluator` | `src/fusion/optimizer/evaluator.py` | Contribution + SLO probability |
| Optimizer | `OptimizationEngine` | `src/fusion/optimizer/engine.py` | Per-workload recommendation (fleet-aware accounting) |
| Optimizer | `GlobalAllocator` | `src/fusion/optimizer/global_allocator.py` | Fleet-wide allocation (exhaustive / CP-SAT / greedy) |
| Optimizer | `DemandAwareOptimizer` | `src/fusion/optimizer/demand_aware.py` | Forecast-aware selection + scale-to-zero |
| Optimizer | `HistoricalReplayEngine` | `src/fusion/optimizer/historical_replay.py` | Deterministic replay comparison |
| Simulator | `FleetSimulator` | `src/fusion/simulator/engine.py` | Coarse event loop over scenarios |
| Simulator | `ScenarioLoader` | `src/fusion/simulator/scenario.py` | Scenario parsing + conversion |
| Policies | `ConstraintValidator` | `src/fusion/policies/validator.py` | 10 constraint checks |
| Policies | `IsolationValidator` | `src/fusion/policies/isolation.py` | Isolation mode validation |
| Recommendations | `RecommendationReport`, `ReportGenerator` | `src/fusion/recommendations/report.py` | Markdown report generation |
| Adapters | `KubernetesInventoryAdapter` | `src/fusion/adapters/kubernetes.py` | Read-only node → GPU inventory |
| Adapters | `PrometheusMetricsAdapter` | `src/fusion/adapters/prometheus.py` | Metrics interface |
| Adapters | `OpenCostAdapter` | `src/fusion/adapters/opencost.py` | Cost interface |
| Adapters | `DynamoMetricsAdapter` | `src/fusion/adapters/dynamo.py` | Dynamo planner metrics |
| Adapters | `KAIAdapter` | `src/fusion/adapters/kai.py` | KAI scheduler data |
| Adapters | `IgniteProvider` | `src/fusion/adapters/ignite.py` | Capacity-contract protocol |
| Plugins | — | `src/fusion/plugins/` | **Empty** |

---

## 4. Scheduling Decision Path

Fusion's "scheduling" is a recommendation pipeline, not a scheduler loop. The path is:

```
Workload arrival
      │  (API /v1/optimization/evaluate, CLI `fusion optimize`, or direct call)
      ▼
Observation            InventoryService / adapter → GPUInventory[]; profiles; demands
      ▼
Eligibility            TenantPolicy lookup, isolation mode, GPU compatibility
      ▼
Resource feasibility   ConstraintValidator.validate_all (10 checks: slots, memory,
                       isolation, compatibility, quota, resource_quota, health,
                       engine, sharing, latency)
      ▼
SLO estimation         AlternativeEvaluator: slo_violation_probability
                       (0.10 without profile, 0.01 with profile); latency SLO check
                       compares profile.ttft_p99_ms vs policy.latency_p99_ms
      ▼
Scoring                contribution = revenue - cost; GlobalAllocator adds
                       serve-first objective (weighted_served × 1,000,000)
      ▼
Recommendation         FusionRecommendation (recommended_plan, rejected_alternatives,
                       rejected_reasons, expected_economics, provenance)
      ▼
Human approval         approval_status = ADVISORY_ONLY; execution_eligibility = False
      ▼
Kubernetes placement   *** OUT OF SCOPE — NOT IMPLEMENTED ***
      ▼
Telemetry feedback     *** NOT IMPLEMENTED as a closed loop ***
                       (Prometheus adapter exists as an interface; no controller loop)
```

Key properties of this path:

- The two optimizers differ in scope. `OptimizationEngine` recommends **per workload**
  and now performs fleet-wide slot+memory accounting across deployments
  (`engine.py:52-56`, `engine.py:122-167`). `GlobalAllocator` optimizes the **whole
  fleet jointly** and is the correct tool for packing.
- `GlobalAllocator` selects an engine dynamically: exhaustive search when
  `len(deployments) <= 9` **and** the candidate product `<= max_exhaustive_combinations`
  (default 10,000); otherwise CP-SAT if `ortools` is importable; otherwise greedy
  (`global_allocator.py:87-105`). This guard was added to fix silent under-exploration
  (bug #4).
- The `__baseline__` alternative is a reference point and is **never** a candidate
  (`global_allocator.py:67`); an explicit `__unplaced__` option permits a partial plan
  when physical capacity is genuinely insufficient.

---

## 5. Externally Visible Guarantees

The following table separates guarantees that are **implemented and tested** from
**claims** that are documented but not fully backed by code. This distinction is the
core of the audit.

### 5.1 Implemented and tested guarantees

| Guarantee | How it is enforced | Evidence |
|-----------|--------------------|----------|
| Advisory-only output | `approval_status=ADVISORY_ONLY`, `execution_eligibility=False` on every `FusionRecommendation` | `global_allocator.py:208-209`, `engine.py:227-228`; `test_security.py::test_advisory_mode_default` |
| Never mutates cluster state | No write calls in any adapter; RBAC is read-only | `adapters/kubernetes.py` (read calls only); `deploy/rbac/cluster-role.yaml` |
| No GPU over-commit in global plans | Slot + memory accounting per GPU across deployments | `test_optimizer_oracle.py::test_no_gpu_overcommit`, `test_concurrency_races.py::assert_no_overcommit` |
| Unhealthy GPUs are never placed | Health check rejects non-HEALTHY | `validator.py:97-104`; `test_slo_validation.py::test_unavailable_hardware` |
| Revenue is never fabricated | `estimated_revenue_per_hour` stays `None` absent explicit billing | `candidate.py:129-134`; `test_optimizer_engine.py::test_revenue_is_never_fabricated` |
| Deterministic plans | Pure functions of input | `test_optimizer_oracle.py::test_determinism`, `test_stability.py`, `test_concurrency_races.py` |
| Malformed input → 422, not 500 | `ValidationError` mapped to HTTP 422 | `api/app.py:91-98`; `test_security.py::test_malformed_workload_content_rejected` |
| Secrets not leaked in `repr()` | `api_key: SecretStr` | `config/settings.py:41`; `test_security.py::test_secrets_not_logged` |
| Negative resource values rejected | Pydantic `Field(ge=0)` | `types.py:94-96,141-145`; `test_fault_injection.py::test_invalid_gpu_memory_negative` |
| Tenant `resource_quotas` enforced | Validator + allocator checks | `validator.py:87-95`, `global_allocator.py:512-517`; `test_security.py::test_resource_quota_respected` |
| Recommendations reference only observed GPUs | Provenance check against adapter output | `test_k8s_integration.py::test_recommendation_uses_observed_state` |

### 5.2 Claims vs implemented — explicit marking

| Claim in docs (`docs/architecture.md`, README, `docs/limits.md`) | Status | Note |
|------------------------------------------------------------------|--------|------|
| "ADVISORY mode by default — never mutates cluster state" | **Implemented** | Verified by security tests |
| "Deterministic optimization — reproducible candidate evaluation" | **Implemented** | Verified by determinism tests |
| "Kubernetes API (read-only inventory)" | **Implemented (interface + real adapter)** | Tested against real kind cluster; costs reported as `Decimal("0")` from K8s labels |
| "Prometheus (GPU metrics and telemetry)" | **Claim / interface-only** | Adapter class exists; no live DCGM/Prometheus validation performed |
| "OpenCost (cost allocation data)" | **Claim / interface-only** | Adapter class exists; costs remain user-supplied/fixture-based |
| "NVIDIA Dynamo Planner external plugin" | **Not implemented** (acknowledged in `docs/limits.md`) | `DynamoMetricsAdapter` exists as a data interface only |
| "KAI Scheduler placement integration" | **Not implemented** (acknowledged) | `KAIAdapter` exists as a data interface only |
| "Downstream: Tesseract Foundry / Ignite" | **Interface only** | `IgniteProvider` protocol defined; no live handshake |
| "Optimizer engine" (listed as *Prototype quality*) | **Implemented** | Exceeds "prototype": oracle optimality gap = 0 on small cases |
| "Fleet simulator" (listed as *Prototype quality*) | **Implemented but limited** | No latency/queue model (see §7 and `SLO_VALIDATION.md`) |

---

## 6. Architecture Diagram

```
                         ┌──────────────────────────────────────────────┐
                         │            External data sources             │
                         │  Kubernetes (read-only) · Prometheus ·       │
                         │  OpenCost · Dynamo · KAI · Fixtures          │
                         └───────────────────────┬──────────────────────┘
                                                 │  adapters (Protocol-based)
                                                 ▼
   ┌─────────────────────────────┐   ┌───────────────────────────────────┐
   │        API (FastAPI)        │   │          CLI (Typer)               │
   │  /v1/inventory  /v1/workloads│   │  fusion inventory|optimize|       │
   │  /v1/optimization/evaluate   │   │  simulate|economics|serve          │
   │  /v1/simulations /v1/forecast│   └───────────────────────────────────┘
   └──────────────┬──────────────┘
                  │  in-memory stores (no controller persistence)
                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                       Optimization Core (advisory)                  │
   │  InventoryService → CandidateGenerator → ConstraintValidator        │
   │                   → AlternativeEvaluator → CostAccountingEngine     │
   │                   → OptimizationEngine / GlobalAllocator            │
   │                   → DemandAwareOptimizer / HistoricalReplayEngine   │
   └───────────────────────────────┬────────────────────────────────────┘
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
   ┌────────────────────┐                     ┌──────────────────────────┐
   │  FleetSimulator    │                     │  ReportGenerator         │
   │  (coarse, no       │                     │  → Markdown recommendation│
   │   latency model)   │                     └──────────────────────────┘
   └────────────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────────┐
                  │   FusionRecommendation              │
                  │   approval_status = ADVISORY_ONLY   │
                  │   execution_eligibility = False     │
                  └───────────────┬────────────────────┘
                                  │
                     ┌────────────┴─────────────┐
                     ▼                          ▼
        ┌────────────────────────┐   ┌───────────────────────────────┐
        │ Human operator review  │   │ Kubernetes placement          │
        │ (out-of-band)          │   │ *** NOT IMPLEMENTED ***       │
        └────────────────────────┘   └───────────────────────────────┘
```

---

## 7. Subsystem Classification

Classification key: **Implemented+Tested** · **Implemented but unverified** ·
**Partial** · **Stub** · **Not implemented** · **N/A**

| # | Subsystem | Classification | Evidence |
|---|-----------|----------------|----------|
| 1 | Kubernetes scheduling integration | **Not implemented** | No plugin/extender/webhook; `deploy/` = API only |
| 2 | Cluster resource discovery | **Implemented+Tested** | `adapters/kubernetes.py`; `test_k8s_integration.py` (real kind) |
| 3 | GPU inventory | **Implemented+Tested** | `inventory/provider.py`, `inventory/fixture_adapter.py`; `test_inventory.py` |
| 4 | Model resource requirements | **Implemented+Tested** | `ModelDeployment.expected_memory_mb` (`types.py:145`) |
| 5 | Model/replica placement | **Implemented+Tested** | `optimizer/global_allocator.py`; `test_optimizer_oracle.py` |
| 6 | GPU packing logic | **Implemented+Tested** | Exhaustive + CP-SAT + greedy (`global_allocator.py:110-453`) |
| 7 | SLO-aware scheduling decisions | **Partial** | Latency SLO check `validator.py:125-132`; simulator has no latency model |
| 8 | Capacity prediction | **Partial** | `demand/forecaster.py` (moving average only) |
| 9 | Scaling recommendations | **Partial** | `optimizer/demand_aware.py::should_scale_to_zero` |
| 10 | Admission control | **Not implemented** | No admission path |
| 11 | Queue management | **Not implemented** | No queue subsystem |
| 12 | Resource reservation/release | **Partial** | In-request slot/memory accounting only; no persistence across calls |
| 13 | Metrics collection | **Implemented but unverified** | `adapters/prometheus.py` interface; no live metrics run |
| 14 | Cost modeling | **Implemented+Tested** | `economics/accounting.py`; `test_economics.py` |
| 15 | Controller state persistence | **Partial** | ORM defined (`models/orm.py`, 9 tables); API uses in-memory dicts (`api/app.py:51-59`) |
| 16 | Failure recovery | **Partial** | Faults surface cleanly (`test_fault_injection.py`); no bounded retries, no controller loop |
| 17 | Policy configuration | **Implemented+Tested** | `TenantPolicy` (`types.py:116-130`); `test_security.py` |
| 18 | RL / bandit / simulation | **Simulation implemented; RL not implemented** | `simulator/engine.py`; `test_rl_readiness.py` asserts RL absence |

### The six engines

| Engine | Classification | File |
|--------|----------------|------|
| `OptimizationEngine` | Implemented+Tested | `optimizer/engine.py` |
| `GlobalAllocator` | Implemented+Tested | `optimizer/global_allocator.py` |
| `DemandAwareOptimizer` | Partial | `optimizer/demand_aware.py` |
| `HistoricalReplayEngine` | Implemented+Tested | `optimizer/historical_replay.py` |
| `FleetSimulator` | Implemented (limited fidelity) | `simulator/engine.py` |
| `CostAccountingEngine` | Implemented+Tested | `economics/accounting.py` |

### Adapters

| Adapter | Classification | File |
|---------|----------------|------|
| Kubernetes (inventory) | Implemented+Tested (real kind cluster) | `adapters/kubernetes.py` |
| Prometheus (metrics) | Implemented but unverified (interface) | `adapters/prometheus.py` |
| OpenCost (cost) | Implemented but unverified (interface) | `adapters/opencost.py` |
| Dynamo (planner metrics) | Stub / interface | `adapters/dynamo.py` |
| KAI (scheduler data) | Stub / interface | `adapters/kai.py` |
| Ignite (capacity contract) | Stub / protocol | `adapters/ignite.py` |

---

## 8. Key Gaps

1. **No placement execution.** Recommendations are never applied. There is no
   controller, no CRD, no webhook. "Kubernetes placement" is explicitly out of scope.
2. **No closed telemetry feedback loop.** The Prometheus adapter is an interface; no
   running process consumes it to re-optimize.
3. **State is not persisted by the service.** SQLAlchemy ORM tables exist, but the API
   keeps everything in in-memory dicts; restart loses all state. Reservation/release is
   in-request only.
4. **No queueing or latency model.** The simulator cannot produce p50/p95/p99 latency or
   queue percentiles (`SLO_VALIDATION.md`). SLO numbers are synthetic counters.
5. **Capacity prediction is moving-average only.** No seasonality, no ML, no validated
   forecast accuracy on real traces.
6. **No admission control and no queue management.** Neither exists.
7. **No failure-recovery controller loop and no bounded retries.** Faults are handled by
   surfacing cleanly or falling back, not by retrying.
8. **Benchmark advantage is small and simulated.** On the 6-workload / 8-GPU simulated
   benchmark Fusion improves 6.67% over `k8s_default` and `round_robin`, and 0.00% over
   `bin_packing`, `least_loaded`, and `slo_greedy`. The aspirational 20–30% target is
   **not demonstrated**.
9. **Adapters other than Kubernetes are unexercised against live systems.** Prometheus,
   OpenCost, Dynamo and KAI have no live validation in this audit.
10. **`plugins/` is empty.** The pluggable-extension claim is not yet realized.
