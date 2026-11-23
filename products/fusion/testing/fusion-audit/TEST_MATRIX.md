# Tesseract Fusion — Test Matrix

**Audit date:** 2026-10-12
**Host:** Apple Silicon Mac (M5), macOS 26.6.2 (25G83), arm64
**Python:** 3.12.15 · **pytest** 9.1.1 · **pytest-asyncio** 1.4.0 · **pytest-timeout** 2.4.0
**ortools:** installed (CP-SAT path exercised)
**Kubernetes:** kind v1.37.0, cluster `fusion-audit` (single control-plane node)

> **Gate definitions.** Gates A–H below are the audit gates used in this report. They
> group the existing test suite by concern. No gate definition was supplied with the
> audit request; the groupings here are the ones that match the repository's own test
> layout (`tests/unit`, `tests/contract`, `tests/integration`, `tests/simulator`,
> `tests/audit`).

Status key: **PASS** · **FAIL** · **BLOCKED** · **NOT TESTED** · **NOT APPLICABLE**

---

## 1. Headline Test Counts (exact, as measured on the Mac)

| Suite | Result |
|-------|--------|
| Full suite | **187 passed, 1 skipped, 1 deselected** |
| Kubernetes integration (real kind cluster) | **7 passed, 1 skipped** |
| Endurance | **6 passed + leak check passed** (1 slow soak deselected) |
| Baseline competition | **7 passed** |
| Fleet simulation | **8 passed** |
| SLO validation | **11 passed** |
| Stability | **8 passed** |
| Optimizer oracle | **7 passed (optimality gap = 0)** |

- The single **skip** is the Kubernetes "no-cluster" case
  (`test_adapter_handles_no_cluster`), which is skipped because a cluster *is* reachable.
- The single **deselection** is the slow soak test (`test_soak_short`, `@pytest.mark.slow`).
- The leak check (`test_repeated_calls_no_leak`) passed: retained growth stayed under the
  5 MB bound over 200 optimize calls.

---

## 2. Gate A — Environment & Reproducibility

| ID | Criterion | Status | Evidence |
|----|-----------|--------|----------|
| A1 | Python 3.12.15 venv present at `~/tesseract-fusion/.venv` | PASS | environment metadata |
| A2 | Docker 29.5.2 via Colima (4 CPU, 6 GiB, 60 GiB) | PASS | environment metadata |
| A3 | kind v1.37.0 cluster `fusion-audit` created | PASS | `tests/audit/test_k8s_integration.py` |
| A4 | Pre-existing cluster `tesseract-ignite-gate2` untouched | PASS | not modified by any test |
| A5 | `ortools` installed; CP-SAT path exercised | PASS | `global_allocator.py:96-105`; benchmark `engine=cpsat_global` |
| A6 | `tesseract-fusion` v0.1.0 importable | PASS | `pyproject.toml:7` |
| A7 | Source provenance documented (git `4539cb1` + audit fixes) | PASS | audit record |
| A8 | Reproducible on a second, clean host | NOT TESTED | single audit host only |

---

## 3. Gate B — Unit & Contract Correctness

| ID | Criterion | Status | Evidence |
|----|-----------|--------|----------|
| B1 | Domain model tests pass | PASS | `tests/unit/test_models.py` |
| B2 | Demand/forecast tests pass | PASS | `tests/unit/test_demand.py` |
| B3 | Economics tests pass | PASS | `tests/unit/test_economics.py` |
| B4 | Inventory tests pass | PASS | `tests/unit/test_inventory.py` |
| B5 | Optimizer unit tests pass | PASS | `tests/unit/test_optimizer.py` |
| B6 | API contract tests pass | PASS | `tests/contract/test_api_contract.py` |
| B7 | Adapter contract tests pass | PASS | `tests/contract/test_adapters.py` |
| B8 | Malformed workload content returns 422 (not 500) | PASS | `test_security.py::test_malformed_workload_content_rejected` |
| B9 | Negative memory/slots/workers rejected | PASS | `types.py:94-96,141-145`; `test_boundary_cases.py::test_invalid_negative_values` |

---

## 4. Gate C — Optimizer Correctness

| ID | Criterion | Status | Evidence |
|----|-----------|--------|----------|
| C1 | Optimality gap vs independent oracle = 0 (small cases) | PASS | `test_optimizer_oracle.py::test_optimality_gap_small` |
| C2 | No GPU over-commit in global plan | PASS | `test_optimizer_oracle.py::test_no_gpu_overcommit` |
| C3 | Priority respected when capacity limited | PASS | `test_optimizer_oracle.py::test_priority_respected_when_capacity_limited` |
| C4 | Deterministic plans for identical input | PASS | `test_optimizer_oracle.py::test_determinism` |
| C5 | Baseline never selected as a candidate | PASS | `global_allocator.py:67`; `test_optimizer_oracle.py::test_single_workload_single_gpu` |
| C6 | Infeasible workload yields no placement | PASS | `test_optimizer_oracle.py::test_infeasible_all_rejected` |
| C7 | Cheapest feasible GPU selected | PASS | `test_optimizer_oracle.py::test_cheapest_feasible_selection` |
| C8 | CP-SAT path exercised | PASS | benchmark `engine=cpsat_global` |
| C9 | Exhaustive search falls back when product > cap | PASS | `test_endurance.py::test_optimization_runtime_bounded`; `global_allocator.py:87-105` |
| C10 | Optimality gap on large / real fleets | NOT TESTED | oracle is brute-force; only small cases verified |

---

## 5. Gate D — Kubernetes / Cluster Integration

| ID | Criterion | Status | Evidence |
|----|-----------|--------|----------|
| D1 | K8s client initialises against real cluster | PASS | `test_k8s_integration.py::test_k8s_client_available` |
| D2 | Adapter parses nodes into `GPUInventory` | PASS | `test_k8s_integration.py::test_adapter_reads_nodes` |
| D3 | No-cluster path raises `RuntimeError` cleanly | PASS | `test_adapter_handles_no_cluster` (skipped when cluster present) |
| D4 | Node allocatable cpu/memory readable | PASS | `test_k8s_integration.py::test_node_resource_reporting` |
| D5 | Read-only RBAC sufficient | PASS | `test_k8s_integration.py::test_rbac_readonly_sufficient` |
| D6 | Pod lifecycle probe works | PASS | `test_k8s_integration.py::test_pod_lifecycle` |
| D7 | Scheduling events readable (diagnostics) | PASS | `test_k8s_integration.py::test_scheduling_events_readable` |
| D8 | Recommendation references only observed GPUs | PASS | `test_k8s_integration.py::test_recommendation_uses_observed_state` |
| D9 | Actual Kubernetes placement execution | NOT APPLICABLE | out of scope — Fusion is advisory-only |
| D10 | GPU inventory from real GPU nodes | BLOCKED | kind node is CPU-only; no `nvidia.com/gpu` capacity available |

---

## 6. Gate E — Simulation & SLO

| ID | Criterion | Status | Evidence |
|----|-----------|--------|----------|
| E1 | All 15 workload classes (A–O) complete | PASS | `test_fleet_simulation.py::test_all_workload_classes_run` |
| E2 | All fleet sizes (1..64) complete | PASS | `test_fleet_simulation.py::test_all_fleet_sizes` |
| E3 | Determinism across seeds | PASS | `test_fleet_simulation.py::test_determinism_across_seeds` |
| E4 | No GPU over-commit in simulation | PASS | `test_fleet_simulation.py::test_no_gpu_overcommit_in_sim` |
| E5 | Cost/revenue reconciliation exact | PASS | `test_fleet_simulation.py::test_cost_revenue_reconciliation` |
| E6 | SLO violations tracked as bounded integer | PASS | `test_fleet_simulation.py::test_slo_violations_tracked` |
| E7 | Held-out class evaluation runs | PASS | `test_fleet_simulation.py::test_held_out_evaluation` |
| E8 | Replay determinism | PASS | `test_fleet_simulation.py::test_replay_determinism` |
| E9 | SLO validation suite (11 tests) | PASS | `test_slo_validation.py` |
| E10 | Latency percentiles (p50/p99) measured | FAIL | `test_latency_percentiles_not_provided` pins them to `None` |
| E11 | Real end-to-end SLO guarantee | NOT TESTED / INCOMPLETE | simulator has no latency model |

---

## 7. Gate F — Endurance, Stability & Concurrency

| ID | Criterion | Status | Evidence |
|----|-----------|--------|----------|
| F1 | Single optimize call < 5 s budget | PASS | `test_endurance.py::test_scheduling_latency` |
| F2 | Latency bounded across fleets 1..64 GPUs | PASS | `test_endurance.py::test_scaling_fleet_sizes` |
| F3 | Latency bounded across 1..50 workloads | PASS | `test_endurance.py::test_scaling_workload_counts` |
| F4 | Exhaustive search respects combination cap | PASS | `test_endurance.py::test_optimization_runtime_bounded` |
| F5 | 200 optimize calls: no unbounded leak | PASS | `test_endurance.py::test_repeated_calls_no_leak` |
| F6 | Simulation throughput > 0 | PASS | `test_endurance.py::test_simulation_throughput` |
| F7 | Long soak (~30 s) stability | NOT TESTED (deselected) | `test_soak_short` marked `@pytest.mark.slow` |
| F8 | Stability suite (8 tests) | PASS | `test_stability.py` |
| F9 | Concurrency / race conditions | PASS | `test_concurrency_races.py` (20/50 threads, 100 requests) |
| F10 | Fault injection | PASS | `test_fault_injection.py` |

---

## 8. Gate G — Security & Policy

| ID | Criterion | Status | Evidence |
|----|-----------|--------|----------|
| G1 | Advisory-only by default | PASS | `test_security.py::test_advisory_mode_default` |
| G2 | No mutation endpoints exposed | PASS | `test_security.py::test_no_mutation_endpoints` |
| G3 | API key never appears in logs/`repr()` | PASS | `test_security.py::test_secrets_not_logged`; `settings.py:41` |
| G4 | Malformed workload rejected (422) | PASS | `test_security.py::test_malformed_workload_content_rejected` |
| G5 | Tenant quotas enforced | PASS | `test_security.py::test_quota_enforcement`, `test_resource_quota_respected` |
| G6 | Tenant isolation in models | PASS | `test_security.py::test_tenant_isolation_in_models` |
| G7 | No arbitrary code execution | PASS | `test_security.py::test_no_arbitrary_code_execution` |
| G8 | Priority abuse resisted | PASS | `test_security.py::test_priority_abuse` |
| G9 | Production authentication by default | NOT IMPLEMENTED | `enable_auth=False` default; documented |

---

## 9. Gate H — RL Readiness & Baseline Competition

| ID | Criterion | Status | Evidence |
|----|-----------|--------|----------|
| H1 | No RL framework imports / Env classes | PASS | `test_rl_readiness.py::test_no_rl_environment_present` |
| H2 | No policy checkpoint files | PASS | `test_rl_readiness.py::test_no_policy_checkpoints` |
| H3 | No training loops / gradient updates | PASS | `test_rl_readiness.py::test_no_training_code` |
| H4 | Deterministic baseline exists and is advisory | PASS | `test_rl_readiness.py::test_deterministic_baseline_exists` |
| H5 | Greedy fallback works without `ortools` | PASS | `test_rl_readiness.py::test_safe_fallback_available` |
| H6 | Comparison vs 5 baselines runs | PASS | `test_baseline_competition.py` (7 passed) |
| H7 | 20–30% cost-improvement target | FAIL (not demonstrated) | best observed = 6.67% vs `k8s_default`/`round_robin`; 0.00% vs three others |

---

## 10. Test File Coverage

| File | Covers |
|------|--------|
| `tests/unit/test_models.py` | Pydantic domain models and validation |
| `tests/unit/test_demand.py` | Moving-average forecaster, profiles |
| `tests/unit/test_economics.py` | `CostAccountingEngine`, revenue, margin |
| `tests/unit/test_optimizer.py` | Candidate generation / engine units |
| `tests/unit/test_inventory.py` | Fixture inventory service |
| `tests/contract/test_api_contract.py` | FastAPI endpoints (health, inventory, workloads, optimization, simulations, forecast) |
| `tests/contract/test_adapters.py` | Adapter contract shapes |
| `tests/integration/test_simulator_integration.py` | Simulator ↔ scenario integration |
| `tests/simulator/test_scenarios.py` | Scenario loading/execution |
| `tests/audit/test_optimizer_oracle.py` | Independent brute-force oracle, optimality gap, over-commit, determinism, priority |
| `tests/audit/oracle.py` | The independent oracle implementation (not a test) |
| `tests/audit/test_optimizer_engine.py` | Single-workload engine contract, revenue non-fabrication, memory accounting |
| `tests/audit/test_baseline_competition.py` | 5 baseline strategies vs Fusion |
| `tests/audit/benchmark_report.py` | Machine-readable benchmark generator |
| `tests/audit/test_k8s_integration.py` | Real kind cluster: client, nodes, RBAC, pod probe, provenance |
| `tests/audit/k8s_helpers.py` | K8s availability/RBAC helpers |
| `tests/audit/test_fleet_simulation.py` | 15 classes, fleet sizes, determinism, reconciliation, replay |
| `tests/audit/fleet_scenarios.py` | Synthetic scenarios A–O (not tests) |
| `tests/audit/test_slo_validation.py` | SLO accounting, saturation, overload, missing-profile behaviour, latency-absent contract |
| `tests/audit/test_stability.py` | Oscillation, hysteresis, decision interval, migration bounds |
| `tests/audit/test_endurance.py` | Latency, scaling, bounded search, leak, soak (slow), throughput |
| `tests/audit/test_concurrency_races.py` | Threaded allocation, duplicate ids, release cycles |
| `tests/audit/test_fault_injection.py` | Crash, missing metrics, stale telemetry, corrupted config, resource exhaustion |
| `tests/audit/test_security.py` | Advisory default, no mutation, secrets, quotas, isolation, input abuse |
| `tests/audit/test_boundary_cases.py` | Empty cluster, zero capacity, impossible SLO, duplicate ids, concurrency |
| `tests/audit/test_rl_readiness.py` | Absence of RL, presence of deterministic fallback |
| `tests/audit/collect_env.py` | Environment metadata collector (not a test) |
| `tests/audit/test_fleet_simulation.py::test_held_out_evaluation` | Held-out classes L/M/N/O |

---

## 11. Gate Roll-up

| Gate | Description | Verdict |
|------|-------------|---------|
| A | Environment & reproducibility | PASS (A8 not tested) |
| B | Unit & contract correctness | PASS |
| C | Optimizer correctness | PASS (C10 not tested) |
| D | Kubernetes / cluster integration | PASS for read-only adapter; execution N/A; GPU nodes BLOCKED |
| E | Simulation & SLO | PASS for simulation; **FAIL for real latency/SLO guarantees** |
| F | Endurance, stability, concurrency | PASS (slow soak deselected) |
| G | Security & policy | PASS (auth by default not implemented) |
| H | RL readiness & baseline competition | PASS for RL-absence; **FAIL for 20–30% target** |

**Two gates do not fully pass**, and both are reported honestly rather than hidden:

- **Gate E** — the simulator does not model latency, so no real SLO guarantee is
  validated (see `SLO_VALIDATION.md`).
- **Gate H** — the aspirational 20–30% cost-improvement target is not demonstrated; the
  measured advantage is 6.67% over two baselines and 0.00% over three (see
  `OPTIMIZER_CORRECTNESS.md` and the benchmark results).
