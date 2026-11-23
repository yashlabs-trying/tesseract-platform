# Tesseract Fusion — SLO Validation

**Audit date:** 2026-10-12
**Suite:** `tests/audit/test_slo_validation.py` — **11 passed**
**Verdict:** **INCOMPLETE.** Fusion has no validated, end-to-end SLO guarantee.

---

## 1. Headline Finding

Fusion's `FleetSimulator` **does not model latency**. It does not produce p50/p95/p99
latency or queue-duration percentiles, and it does not derive SLO violations from a
queueing model. The `slo_violations` field is a **synthetic random per-step counter**.

Therefore:

- The SLO numbers produced by the simulator are **not measured latency**.
- The 11 passing SLO tests validate the *accounting and control-flow* that exists — not
  real latency compliance.
- Real SLO guarantees remain **unvalidated** and are marked **INCOMPLETE**.

This is stated plainly rather than papered over. The audit deliberately pins the current
contract instead of inventing numbers.

---

## 2. Why the SLO Numbers Are Synthetic

`FleetSimulator.run()` (`src/fusion/simulator/engine.py:78-80`) does this on every step:

```python
# Track SLO violations (simulated)
if self._rng.random() < 0.01:
    result.slo_violations += 1
```

That is the entire SLO logic: with 1% probability per time step, increment a counter. It
is bounded above by the number of steps (one increment per step maximum), and it is
independent of load, concurrency, queue depth, or latency. It is **not** an SLO
measurement.

The `SimulationResult` model defines latency fields
(`latency_p50_ms`, `latency_p99_ms`, `queue_duration_p50_ms`, `queue_duration_p99_ms`) in
`src/fusion/models/types.py:344-347`, but `FleetSimulator` **never assigns them**. They
remain `None`.

The test `test_latency_percentiles_not_provided`
(`test_slo_validation.py:251-262`) exists specifically to pin this contract:

```python
assert result.latency_p50_ms is None
assert result.latency_p99_ms is None
assert result.queue_duration_p50_ms is None
assert result.queue_duration_p99_ms is None
```

---

## 3. What Was Tested

`tests/audit/test_slo_validation.py` — **11 tests, all passing**. The tests assert the SLO
*accounting* that actually exists and document what is missing.

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_normal_load_below_capacity` | Steady load below capacity completes; violation rate in `[0.0, 0.5)` |
| 2 | `test_load_approaching_saturation` | High utilisation still accounts for violations; `slo_violations` is a non-negative int bounded by step count |
| 3 | `test_sustained_overload` | Overload is deterministic for a fixed seed (`r1.slo_violations == r2.slo_violations`) |
| 4 | `test_sudden_burst` | Bursty workload completes and generates requests |
| 5 | `test_multi_tenant_competing` | Three tenants all accounted for; policies loaded |
| 6 | `test_priority_sensitive` | Tenant priority ordering preserved through scenario loading |
| 7 | `test_unavailable_hardware` | Unhealthy GPU never selected by the optimizer |
| 8 | `test_missing_performance_predictions` | No profile → conservative SLO probability `0.10`; with `MEASURED` profile → `0.01` |
| 9 | `test_queue_growth_bounded` | Saturated scenario terminates (no unbounded queue growth) |
| 10 | `test_cold_vs_hot_models` | `should_scale_to_zero` scales idle models and keeps warm active/bursty models |
| 11 | `test_latency_percentiles_not_provided` | Pins latency/queue percentiles to `None` |

### 3.1 The one real SLO signal: `AlternativeEvaluator`

The only SLO-aware logic that is *derived from data* rather than random is in
`src/fusion/optimizer/evaluator.py:32-38`:

- No performance profile → `slo_violation_probability = 0.10`, plus a risk adjustment of
  `cost × 0.2`.
- With a profile → `slo_violation_probability = 0.01`.

These are **fixed constants**, not a model. They are asserted in
`test_missing_performance_predictions`. They represent a conservative prior, not a
measured violation rate.

### 3.2 Latency SLO check (constraint path)

`ConstraintValidator._check_latency` (`src/fusion/policies/validator.py:125-132`) compares
a profile's `ttft_p99_ms` against the tenant policy's `latency_p99_ms`:

- PASS if `profile.ttft_p99_ms <= policy.latency_p99_ms`
- FAIL otherwise
- UNKNOWN if either side is missing

This is a real, deterministic constraint — but it depends entirely on a **supplied**
performance profile. No measured profile exists in the audit environment, so in practice
this check returns UNKNOWN and does not block.

---

## 4. What Remains Unvalidated

| Capability | Status |
|------------|--------|
| p50 latency measurement | NOT IMPLEMENTED (field always `None`) |
| p95/p99 latency measurement | NOT IMPLEMENTED |
| Queue-duration percentiles | NOT IMPLEMENTED |
| Queueing model (M/M/c, etc.) | NOT IMPLEMENTED |
| SLO violations derived from load/latency | NOT IMPLEMENTED (random counter) |
| NUMA / PCIe / power modeling | NOT IMPLEMENTED |
| Real measured performance profiles | NOT TESTED (no live Foundry data) |
| End-to-end SLO guarantee under production traffic | NOT TESTED |

### 4.1 Simulator limitations (as measured)

- Does **not** populate `latency_p50_ms`, `latency_p99_ms`, `queue_duration_p50_ms`, or
  `queue_duration_p99_ms` (always `None`).
- SLO violations are a random per-step counter bounded by step count — not derived from a
  queueing model.
- No NUMA, PCIe, or power modeling.
- All costs are user-supplied or fixture-based.

---

## 5. Honest Verdict

| Question | Answer |
|----------|--------|
| Do the SLO tests pass? | **Yes — 11/11** |
| Do they validate real latency SLOs? | **No** |
| Is the `slo_violations` number a measured latency violation count? | **No — it is a synthetic counter** |
| Is there an end-to-end SLO guarantee? | **No — INCOMPLETE** |
| What *is* validated? | The accounting/control-flow around SLO fields, determinism, conservative priors, and the latency-constraint check when a profile is supplied |

**The SLO subsystem is marked INCOMPLETE for real SLO guarantees.** The passing tests
should be read as "the existing (limited) SLO plumbing behaves consistently", not as
"Fusion meets its SLOs". Producing a real SLO guarantee requires a latency/queueing model
in the simulator and measured performance profiles — neither of which exists today.
