# Fusion — Benchmark Results

> **CRITICAL DISCLAIMER: ALL COSTS REPORTED HERE ARE SIMULATED.**
> No real GPU hardware was exercised. No wall-clock speedup or real cost saving is claimed.
> The aspirational **20–30% cost-improvement target is NOT demonstrated** by these results.

---

## 1. Baseline comparison (simulated)

| Strategy | Simulated cost | Delta vs Fusion | Delta % |
|----------|----------------|-----------------|---------|
| **Fusion** | **$21.00** | — | — |
| `k8s_default` | $22.50 | +$1.50 | 6.67% |
| `round_robin` | $22.50 | +$1.50 | 6.67% |
| `bin_packing` | $21.00 | $0.00 | 0.00% |
| `least_loaded` | $21.00 | $0.00 | 0.00% |
| `slo_greedy` | $21.00 | $0.00 | 0.00% |

**Interpretation:**

- Fusion achieves a **6.67% simulated cost improvement** over `k8s_default` and `round_robin`.
- Fusion is **equal to** `bin_packing`, `least_loaded`, and `slo_greedy` (0.00% delta).
- This is **well short of the aspirational 20–30% target**, which is **not demonstrated**.
- The scenario is small (see §4), so these deltas should not be extrapolated.

---

## 2. Cost-improvement formula

Cost improvement relative to a baseline is computed as:

```
improvement_pct = (baseline_cost - fusion_cost) / baseline_cost * 100
```

Applied to the strongest baselines:

```
k8s_default:  (22.50 - 21.00) / 22.50 * 100 = 1.50 / 22.50 * 100 = 6.67%
round_robin:  (22.50 - 21.00) / 22.50 * 100 = 1.50 / 22.50 * 100 = 6.67%
bin_packing:  (21.00 - 21.00) / 21.00 * 100 = 0.00%
least_loaded: (21.00 - 21.00) / 21.00 * 100 = 0.00%
slo_greedy:   (21.00 - 21.00) / 21.00 * 100 = 0.00%
```

---

## 3. Pareto trade-offs

- The comparison is a **single-objective (cost)** view over a small scenario. Where Fusion ties the
  strongest baselines (0.00% delta), no cost win exists; any benefit would have to come from
  secondary objectives (e.g., SLO adherence, fragmentation, fairness) that are **not** quantified
  here.
- `k8s_default` and `round_robin` are the only baselines Fusion beats, and only by **6.67%**.
- Because all values are simulated and the scenario is small, **no Pareto frontier** can be claimed
  with confidence. Treat this as a directional sanity check, not a decision-grade result.

---

## 4. Scenario-size caveat

The benchmark scenario is **small**. A 6.67% delta on a small scenario is fragile: it can flip with
a different seed, workload mix, or fleet size. The result should not be generalized to production
fleets without a larger, calibrated evaluation.

---

## 5. Scheduling latency / endurance

| Test | Shape | Observed result |
|------|-------|-----------------|
| Soak test | ~30 s sustained load | Stable |
| Leak check | 200 calls | Passed — no memory growth |

- The soak run remained **stable** for ~30 seconds.
- A **200-call leak check passed** with **no observed memory growth**.
- These are stability/latency observations on the optimizer, **not** real-hardware scheduling
  latency measurements.

---

## 6. Summary of claims vs. reality

| Claim | Status |
|-------|--------|
| Fusion beats `k8s_default` / `round_robin` by 6.67% (simulated) | **Demonstrated (simulated)** |
| Fusion matches `bin_packing` / `least_loaded` / `slo_greedy` (simulated) | **Demonstrated (simulated)** |
| 20–30% cost improvement | **NOT demonstrated** |
| Real-hardware cost savings | **NOT demonstrated** |
| Scheduler endurance / no leak | **Demonstrated (optimizer soak)** |

---

## 7. Reproducibility notes

- All costs are produced by `FleetSimulator` (see `SIMULATOR_VALIDATION.md`), which is deterministic
  from a seed.
- The simulator does **not** model latency percentiles (always `None`); SLO violations are synthetic
  counters. Cost comparisons should be read with this in mind.
