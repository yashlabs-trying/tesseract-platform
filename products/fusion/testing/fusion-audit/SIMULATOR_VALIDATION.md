# Fusion — Simulator Validation

**Component:** `FleetSimulator`
**Purpose:** Deterministic, time-stepped evaluation of allocation/optimization strategies for Fusion.
**Claim boundary:** This document validates the simulator as a *deterministic software artifact*. It
does **not** claim the simulator is calibrated against real GPUs, and it does **not** claim any
hardware speedup or real cost saving.

---

## 1. Design

| Property | Value |
|----------|-------|
| Execution model | Time-stepped |
| Determinism | Deterministic from seed (same seed ⇒ same trace/outcome) |
| Workload classes | A–O (15 classes) |
| Fleet sizes | 1–64 nodes/GPUs |
| Seeds | 3 seeds |
| Evaluation method | Held-out traces |
| Latency percentiles | **Not modeled — always `None`** |
| SLO violations | **Synthetic counters** (not derived from real latency) |

### 1.1 Time-stepped model

The simulator advances in discrete time steps. At each step, workload arrivals, allocations, and
releases are processed deterministically given the seed and configuration. This makes runs
reproducible and diff-able.

### 1.2 Determinism

Given the same seed, configuration, and workload trace, the simulator produces identical results.
This supports regression testing and apples-to-apples comparison of strategies.

### 1.3 Workload classes A–O

Fifteen workload classes (A–O) span a range of resource profiles and behaviors so strategies can be
compared across a spectrum rather than a single workload shape.

### 1.4 Fleet sizes 1–64

The fleet can be swept from a single unit to 64 units, allowing scaling behavior to be observed.

### 1.5 Seeds and held-out traces

Three seeds are used. **Held-out traces** (not used during tuning) are used for evaluation to reduce
overfitting to a specific trace.

---

## 2. Validation performed

| Validation | Method | Result |
|------------|--------|--------|
| Determinism | Re-run same seed/config | Identical outcomes |
| Workload coverage | Classes A–O exercised | Covered |
| Fleet scaling | Sweep sizes 1–64 | Executed |
| Seed robustness | 3 seeds | Executed |
| Overfit control | Held-out traces | Used |
| Strategy comparison | Baseline vs Fusion | Produced (see `BENCHMARK_RESULTS.md`) |

---

## 3. Simulator assumptions

1. Resources are discretized into slots/units; allocation is integral where the model requires it.
2. Workload demand is specified by profile; no live measurement feeds the simulator in this mode.
3. Time advances in fixed steps; sub-step timing is not modeled.
4. SLO violations are counted synthetically rather than derived from measured latency.
5. Latency percentiles are **not** modeled (always `None`).
6. Determinism is guaranteed by the seed; stochasticity is only via the seeded RNG.
7. No real hardware contention, thermal, or power effects are represented.

---

## 4. Explicitly NOT calibrated against real GPUs

The following are **not** modeled or calibrated against real hardware. No hardware speedup is
claimed on the basis of these results.

| Aspect | Status | Note |
|--------|--------|------|
| Latency percentiles (p50/p95/p99) | **NOT modeled** | Always `None` |
| KV-cache dynamics | **NOT modeled** | No cache growth/eviction modeling |
| Cold-start timing | **NOT modeled** | No model-load/init timing |
| Power / energy | **NOT modeled** | No power draw or efficiency |
| NUMA / topology locality | **NOT modeled** | No interconnect/affinity effects |
| Real GPU execution | **NOT exercised** | CPU-only kind, no NVIDIA device plugin |
| Real scheduling latency | **NOT measured** | Soak/leak tests are optimizer-level only |

---

## 5. Evidence

- Simulator produced the baseline comparison in `BENCHMARK_RESULTS.md` (Fusion $21.00; `k8s_default`
  and `round_robin` $22.50 = 6.67%; `bin_packing`, `least_loaded`, `slo_greedy` $21.00 = 0%).
- Soak test (~30 s) stable; 200-call leak check passed with no memory growth.
- All costs are simulated.

---

## 6. Conclusion

`FleetSimulator` is a **deterministic, time-stepped** evaluation harness with workload classes A–O,
fleet sizes 1–64, 3 seeds, and held-out traces. It is suitable for **relative, reproducible strategy
comparison** under its stated assumptions.

It is **not** a hardware-calibrated digital twin. Latency percentiles, KV-cache dynamics, cold-start
timing, power, and NUMA are not modeled, and no real GPU was exercised. Accordingly, **no
hardware-level performance or cost claim is made** from simulator results.
