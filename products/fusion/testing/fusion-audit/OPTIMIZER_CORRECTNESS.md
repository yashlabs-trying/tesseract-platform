# Tesseract Fusion — Optimizer Correctness

**Audit date:** 2026-10-12
**Component under test:** `fusion.optimizer.global_allocator.GlobalAllocator` (optimizer version `0.3.0`)
**Oracle:** `tests/audit/oracle.py` — an independent brute-force implementation
**Result:** optimality gap = **0** on all small cases tested; determinism and no-overcommit
invariants hold.

---

## 1. Purpose

The optimizer is the core of the product: it decides which workload runs on which GPU.
This section establishes, with evidence, that its output is:

1. **optimal** on small cases (against an independent oracle),
2. **deterministic** (reproducible for identical input),
3. **safe** (never over-commits a GPU),
4. **priority-correct** when capacity is scarce.

It also documents the **pre-fix vs post-fix** behaviour, because the audit found and fixed
two critical optimizer bugs. The audit does not hide the earlier broken behaviour.

---

## 2. The Independent Oracle

The oracle is a **deliberately separate implementation** of the optimization problem. It
does not import, wrap, or call Fusion's optimizer. It lives in
`tests/audit/oracle.py` and is a brute-force enumerator:

```python
def oracle_optimal_contribution(deployments, gpus, choices_per_deployment):
    for combo in product(*choices_per_deployment):
        # accumulate slot + memory usage per GPU
        # mark infeasible if usage > capacity
        # keep the minimum total cost over feasible combos
```

Key points about the oracle (`oracle.py:1-63`):

- It enumerates **every combination** in the Cartesian product of per-deployment choices
  (`itertools.product`), so on small instances it is the true optimum by construction.
- It tracks per-GPU **slot usage** and **memory usage** independently, mirroring the
  physical constraints.
- Because revenue is identical across plans in the no-billing case, maximizing
  contribution (`revenue − cost`) is equivalent to **minimizing total cost**. The oracle
  returns the minimum total cost.
- The test file (`test_optimizer_oracle.py:96-127`) builds the oracle's option set with
  `build_choices()`, which mirrors `CandidateGenerator`'s option set but is written
  separately and does **not** call Fusion's code.

> **Why this matters.** An optimizer validated only against itself proves nothing. The
> oracle is a second, naive implementation; agreement between the two is real evidence.

---

## 3. Optimality-Gap Methodology

1. Construct a small instance (few deployments, few GPUs) with explicit costs.
2. Run `GlobalAllocator().allocate(...)` and read Fusion's reported total operating cost
   from `rec.expected_economics.expected_operating_cost`.
3. Run the oracle on the same instance to get the true minimum cost.
4. Compute `gap = fusion_cost − oracle_cost`.

A gap of `Decimal("0")` means Fusion found the optimum.

### Result — `test_optimality_gap_small`

- Instance: **3 workloads / 3 GPUs**, each GPU with 2 slots, costs `$2`, `$3`, `$4`.
- `n > 0` combinations enumerated by the oracle; `oracle_cost is not None`.
- **`gap == Decimal("0")`** (assertion in `test_optimizer_oracle.py:162`).
- All 3 workloads served (`len(rec.recommended_plan) == 3`).

Suite-level result: **7 passed** in `tests/audit/test_optimizer_oracle.py`, with
**optimality gap = 0**.

---

## 4. Invariants Verified

| Invariant | Test | What it asserts |
|-----------|------|-----------------|
| Single workload, single GPU → placed | `test_single_workload_single_gpu` | `recommended_plan[0].gpu_pool_id == "gpu-0"` |
| Optimality gap = 0 | `test_optimality_gap_small` | `fusion_cost − oracle_cost == 0` |
| Infeasible workload → no placement | `test_infeasible_all_rejected` | workload needing 1,000,000 MB on an 81,920 MB GPU is not placed |
| No GPU over-commit | `test_no_gpu_overcommit` | summed replicas per GPU ≤ slots |
| Determinism | `test_determinism` | 10 identical runs → 1 unique plan signature |
| Cheapest feasible selection | `test_cheapest_feasible_selection` | cheap-but-too-small GPU rejected; expensive feasible GPU chosen |
| Priority correctness | `test_priority_respected_when_capacity_limited` | higher-priority tenant wins the only slot |

### 4.1 No-overcommit invariant

`test_no_gpu_overcommit` (`test_optimizer_oracle.py:177-193`) runs 3 workloads against
`gpu-0` (2 slots) and `gpu-1` (1 slot) and asserts, for every GPU, that the summed
`replica_count` of the recommendation never exceeds `available_resource_slots`. The same
invariant is enforced independently in `test_concurrency_races.py` and
`test_fault_injection.py` via `assert_no_overcommit`.

### 4.2 Determinism

`test_determinism` (`test_optimizer_oracle.py:196-207`) runs the allocator 10 times on
identical input and asserts the set of plan signatures has exactly one element. The plan
signature is `(deployment_id, gpu_pool_id, replica_count, cost)` — the `recommendation_id`
UUID is deliberately excluded. Determinism is also asserted in
`test_concurrency_races.py` (30 fresh-allocator cycles, 100 sequential calls) and
`test_rl_readiness.py::test_deterministic_baseline_exists`.

### 4.3 Priority correctness

`test_priority_respected_when_capacity_limited` (`test_optimizer_oracle.py:237-251`)
gives two tenants one GPU slot: `t-high` (priority 10) and `t-low` (priority 0). It asserts
`d-high` gets `gpu-0` and `d-low` is not in the plan. This is enforced by the serve-first
objective, which weights each satisfied workload by `1 + priority`
(`global_allocator.py:160-165`).

---

## 5. Pre-fix vs Post-fix Behaviour

The audit found and fixed **11 bugs**; two were critical and directly affected optimizer
correctness. The pre-fix behaviour is recorded here for honesty.

### Bug #1 (CRITICAL) — baseline always won

- **Pre-fix:** `GlobalAllocator` always recommended the zero-cost `__baseline__`
  allocation, because revenue was never populated, so every real GPU placement had a
  *negative* contribution and lost to the free baseline. The optimizer appeared to
  "work" but recommended doing nothing.
- **Fix:** The baseline is excluded from the candidate space
  (`global_allocator.py:67`); the objective became serve-first. Every workload that *can*
  be placed must be placed on a real GPU.
- **Post-fix evidence:** `test_single_workload_single_gpu` now asserts the workload lands
  on `gpu-0`, not `__baseline__`.

### Bug #2 (CRITICAL) — natural replica count skipped

- **Pre-fix:** `CandidateGenerator` only generated `worker_count + 1`, so no natural
  **1:1** placement existed. The optimizer could not even represent "put this workload on
  one GPU".
- **Fix:** All replica counts are generated — `base`, `base-1`, `base+1`, `base-2`
  (`candidate.py:36-42`), and the loop iterates replica counts in the outer loop so the
  candidate cap truncates scale-ups rather than the natural placement.
- **Post-fix evidence:** `test_optimizer_engine.py::test_memory_check_uses_per_replica_accounting`
  asserts `1 in by_replicas` and `2 in by_replicas`.

### Other optimizer-relevant fixes

| # | Bug | Fix |
|---|-----|-----|
| 3 | Rewarded more replicas → over-provisioning | Reward per **satisfied workload**, not per replica (`global_allocator.py:160-165`) |
| 4 | Exhaustive search silently under-explored large spaces (capped at 10,000, explored only the last dimension) | Fall back to CP-SAT/greedy when the product > cap (`global_allocator.py:87-105`) |
| 5 | `OptimizationEngine` could place the same GPU on multiple deployments | Fleet-wide slot + memory accounting (`engine.py:52-56,122-167`) |
| 6 | Negative memory/slots/workers accepted | Pydantic `ge=0` (`types.py:94-96,141-145`) |
| 7 | Unhealthy GPUs silently filtered, never in `rejected_alternatives` | Offered as candidates and rejected explicitly with `gpu_health` reason (`candidate.py:29-31`, `validator.py:97-104`) |
| 11 | Memory double-counting (total multiplied by replica count again) | Derive total from per-replica value (`global_allocator.py:497-504`) |

---

## 6. Benchmark Result (Simulated)

The `GlobalAllocator` was compared against five baseline strategies on an identical
simulated instance: **6 workloads / 8 GPUs**, no revenue model. All costs are simulated
hourly GPU costs; there is no real billing data.

| Metric | Value |
|--------|-------|
| Fusion cost | **$21.00** |
| Workloads served | **6 / 6** |
| Engine | `cpsat_global` |

| Baseline | Fusion improvement |
|----------|--------------------|
| `k8s_default` | **6.67%** |
| `round_robin` | **6.67%** |
| `bin_packing` | **0.00%** |
| `least_loaded` | **0.00%** |
| `slo_greedy` | **0.00%** |

- `test_baseline_competition.py` — **7 passed**.
- All baselines serve the same set of deployments as Fusion
  (`test_equal_workload_served`).
- **Aspirational 20–30% target: NOT DEMONSTRATED.** This is a small scenario and the
  honest result is a modest 6.67% advantage over two baselines and parity with three.
  No larger benchmark was run in this audit, so no stronger claim is made.

---

## 7. Limitations of This Correctness Evidence

1. **Oracle scope.** The oracle is brute-force, so it is only tractable on small
   instances. Optimality is proven for small cases, **not** for large fleets. The
   `GlobalAllocator` falls back to CP-SAT or greedy above the cap; those paths are not
   compared against an independent optimum here.
2. **Simulated economics.** All costs are simulated or fixture-based. No real billing
   data was used.
3. **No revenue model.** Because revenue is `None`/zero in all audit scenarios, the
   oracle's objective reduces to minimum cost. Revenue-maximizing behaviour is therefore
   untested end-to-end.
4. **Candidate-space dependence.** The oracle mirrors `CandidateGenerator`'s option set.
   If that option set is incomplete, both implementations share the same blind spot. The
   oracle validates *selection*, not *candidate generation completeness*.
