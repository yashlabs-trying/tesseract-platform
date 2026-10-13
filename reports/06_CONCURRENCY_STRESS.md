# 06 — Concurrency & Stress

Evidence: `reports/raw/cpu_phases.json` (phase F). In-process real coordinator
with fixture providers; lightweight fixtures only.

## Concurrency levels

| Level | Success | Failed | p50 (ms) | p95 (ms) | p99 (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 0.84 | 0.84 | 0.84 |
| 10 | 10 | 0 | 21.1 | 22.6 | 22.7 |
| 25 | 25 | 0 | 46.7 | 53.0 | 54.6 |
| 100 | 100 | 0 | 124.7 | 188.4 | 235.5 |
| 250 | 250 | 0 | 128.2 | 170.8 | 240.7 |
| 500 | 500 | 0 | 136.3 | 172.3 | 189.6 |

All requests completed; zero failures. Latency grows sub-linearly and plateaus
(the SQLite write lock serializes state writes; throughput is bounded by the
single writer). This is control-plane scheduling under fixtures — **not** model
inference throughput.

## Soak (short)

* 100 reservation/release cycles in **0.106 s**.
* `CapacityLedger` used=0 afterwards (no leaked capacity).
* RSS delta **0 KB** (`ru_maxrss` 61188 → 61188) — no memory growth detected at
  this scale.

## Not run

* **30-minute sustained soak**: not run in this session (time-bounded). Command
  to run it: `scripts/run_full_stack_validation.sh` plus a soak extension, or
  `pytest tests/integration/test_concurrency.py` under an external driver.
* Mixed short/long requests, simultaneous cold wakes, rapid scale up/down and
  multi-model contention at the Kubernetes layer — pending Phase E.

## Observations

* Database contention is the throughput ceiling for the single-writer SQLite
  control plane; documented as a scaling limitation.
* No resource leaks, deadlocks, or queue starvation observed in the tested range.

## Verdict

CPU concurrency 1→500 and a short soak: **PASS**. Production load capacity is
**not claimed** (fixtures + single-writer SQLite).
