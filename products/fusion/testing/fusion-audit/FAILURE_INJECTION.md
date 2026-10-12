# Fusion — Failure Injection Report

**Scope:** Tesseract-Fusion allocator / optimizer / API / Kubernetes integration
**Method:** Deterministic fault injection at the optimizer, allocator, simulator, and Kubernetes layers
**Result summary:** 12/12 optimizer fault-injection tests PASS. Kubernetes pod-lifecycle and node
scenarios PASS at the API-server/scheduler level. Controller crash / leader-election tests are
**NOT APPLICABLE** (Fusion has no controller loop — see below). Concurrency/race suite PASS at all
tested levels with no crashes and no GPU overcommit.

> Evidence basis: this report reflects observed test outcomes only. No real-GPU hardware behavior
> was exercised (CPU-only kind cluster, no NVIDIA device plugin). All numeric cost/reward behavior is
> simulated.

---

## 1. Fault-injection matrix (optimizer / allocator / simulator)

| # | Fault | Injected how | Observed behavior | Detection | Safe? | Recovery | Pass/Fail |
|---|-------|--------------|-------------------|-----------|-------|----------|-----------|
| 1 | Mid-optimization crash | Abort/raise during an optimization pass | Optimization aborts without emitting a partial/corrupt plan; state remains consistent | Exception surfaced to caller; no partial plan committed | Yes | Caller can re-run optimization; no cleanup required | PASS |
| 2 | Missing metrics | Remove metrics for one or more workloads | Optimizer proceeds without crashing; workloads without metrics handled via fallback path | Missing-metric branch exercised; no KeyError/None deref | Yes | Fallback profile / conservative placement | PASS |
| 3 | Stale telemetry | Feed telemetry with stale timestamps | Stale samples detected; not treated as fresh; no crash | Staleness check on timestamp age | Yes | Uses last-known/fallback; flags staleness | PASS |
| 4 | Corrupted config | Malformed/invalid config object | `ValidationError` raised; no unbounded plan produced | Pydantic `ValidationError`; plan remains bounded | Yes | Request rejected with bounded/validated plan | PASS |
| 5 | Zero-slot GPU | GPU advertised with 0 usable slots | Zero-slot GPU excluded from allocation; no division-by-zero, no phantom placement | Slot-count guard | Yes | GPU skipped; capacity remains consistent | PASS |
| 6 | Resource exhaustion | Exhaust fleet CPU/GPU/memory | Allocation fails cleanly when capacity is exhausted; no negative capacity | Capacity accounting check | Yes | Explicit failure; no overcommit | PASS |
| 7 | Unschedulable workload | Workload cannot fit any node/GPU | Workload reported unschedulable; not silently dropped or force-fit | Schedulability check | Yes | Explicit unschedulable result | PASS |
| 8 | Partial allocation failure | Some (not all) placements fail | Successful placements preserved; failed placements reported; no inconsistent fleet state | Per-placement status | Yes | Partial result reported; retryable | PASS |
| 9 | Disappearing node | Node removed from available set mid-flight | Node removed from consideration; no stale reference used | Node-set reconciliation | Yes | Re-plan / skip node | PASS |
| 10 | Duplicate leaders | Duplicate "leader" entries injected into state | Duplicates deduplicated/rejected; no double-counting of capacity | Uniqueness check on leader identity | Yes | Deduplicated state | PASS |
| 11 | Negative GPU memory | GPU with negative memory value | Rejected (validation) — negative values not accepted | `ge=0` validation constraint | Yes | Request rejected | PASS |
| 12 | Unknown model (no profile) | Model with no known profile | Handled via fallback; no crash, no KeyError | Profile-miss branch | Yes | Conservative/fallback profile | PASS |

All 12 optimizer/allocator fault-injection tests **PASS**.

---

## 2. Kubernetes environment & scenarios (Phase 2/7)

### 2.1 Cluster facts

| Item | Value |
|------|-------|
| Tool | kind v1.37.0 |
| Cluster name | `fusion-audit` |
| Topology | Single control-plane |
| CPU / memory | 4 CPU, 6 GiB |
| GPU resources | **None** (CPU-only kind) |
| Preserved cluster | Ignite's `tesseract-ignite-gate2` left intact |
| Device plugin | **No NVIDIA device plugin** |

### 2.2 Validated cluster behaviors

| Scenario | Injected how | Observed behavior | Detection | Safe? | Recovery | Pass/Fail |
|----------|--------------|-------------------|-----------|-------|----------|-----------|
| API server health | `kubectl` against control-plane | API server reachable, responds to queries | Direct API probe | Yes | n/a | PASS |
| Scheduler health | Inspect scheduler | Scheduler running | Pod/event inspection | Yes | n/a | PASS |
| Node Ready | Inspect node status | Node reports `Ready` | `Node.status.conditions` | Yes | n/a | PASS |
| Allocatable reporting | Inspect node capacity/allocatable | Allocatable CPU/memory reported | `Node.status.allocatable` | Yes | n/a | PASS |
| Namespaces | Create/inspect namespaces | Namespaces present and usable | Namespace listing | Yes | n/a | PASS |
| RBAC applied | Apply ClusterRole/bindings | RBAC resources applied successfully | RBAC resource inspection | Yes | n/a | PASS |
| Pod lifecycle | Schedule busybox pod | Pod scheduled and ran | Pod phase / events | Yes | Pod deleted cleanly | PASS |
| Scheduling events readable | Read events for scheduled pod | Scheduling events readable | `kubectl get events` | Yes | n/a | PASS |

### 2.3 K8s integration tests

- **7 passed, 1 skipped.**
- The skipped test is attributable to absent GPU resources (no NVIDIA device plugin / no GPU on the
  CPU-only node). No real GPU validation was performed.

---

## 3. NOT APPLICABLE: controller crash / leader election

Fusion is an **advisory-only** system. By default `Settings.advisory_mode=True` and
`execution_eligibility=False`. It exposes **no controller loop**, no reconciliation loop, and no
mutation of cluster state (see `SECURITY_AUDIT.md`). Therefore:

| Test class | Status | Reason |
|------------|--------|--------|
| Controller crash recovery | **NOT APPLICABLE** | Fusion has no controller/reconciler loop to crash |
| Leader-election failover | **NOT APPLICABLE** | No leader election; no active/standby control loop |
| Controller-driven reschedule on node loss | **NOT APPLICABLE** | Fusion only advises; it does not drive the scheduler/controllers |

The "duplicate leaders" fault (#10) is a **state-level** injection into the optimizer/allocator input,
not a Kubernetes leader-election test. It validates deduplication of duplicate identity entries and
does not imply Fusion participates in leader election.

The "disappearing node" (#9) and Kubernetes node scenarios validate Fusion's *handling of observed
cluster state*, not controller-driven remediation.

---

## 4. Concurrency & race results

| Concurrency test | Levels / shape | Observed behavior | Overcommit? | Crashes? | Pass/Fail |
|------------------|----------------|-------------------|-------------|----------|-----------|
| Optimizer-level concurrency | levels 1, 5, 10, 25, 50 | Stable at every level | None observed | None | PASS |
| Shared-allocator threads | 20 threads sharing one allocator | Stable | None observed | None | PASS |
| Independent engines | 50 independent engines | Stable | None observed | None | PASS |
| Allocate cycles | 30 allocate cycles | Stable | None observed | None | PASS |
| Sequential requests | 100 sequential requests | Stable | None observed | None | PASS |

**Documented finding (pre-fix):** `OptimizationEngine` optimized each workload **independently**;
fleet-wide no-overcommit was guaranteed **only** by `GlobalAllocator`. This was the state observed
during concurrency testing.

**Post-fix:** the engine now enforces **fleet accounting** as well, so the fleet no-overcommit
guarantee no longer depends solely on `GlobalAllocator`.

---

## 5. Limitations / caveats

- No real GPU validation was performed. The kind cluster is CPU-only with no NVIDIA device plugin.
- All optimizer cost/reward behavior is simulated (see `BENCHMARK_RESULTS.md`,
  `SIMULATOR_VALIDATION.md`).
- Controller-level failure modes are out of scope because Fusion is advisory-only.
- The single skipped K8s integration test corresponds to GPU-dependent behavior unavailable in this
  environment.
