# 10 — Platform Performance Baseline

Evidence: `reports/raw/performance_baseline.json`. In-process CPU measurements
after warmup; medians and tails. **Control-plane overhead only — not model
inference.**

| Stage | n | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---:|---:|---:|---:|---:|
| Foundry profile mapping | 2000 | 0.004 | 0.007 | 0.008 | ~0.004 |
| Reservation transaction (SQLite) | 1000 | 0.020 | 0.028 | 0.038 | ~0.021 |
| Ignite manifest build | 2000 | 0.001 | 0.001 | 0.002 | ~0.001 |
| Coordinator full lifecycle | 300 | 0.573 | 0.914 | 1.411 | ~0.61 |
| Rollback completion | 200 | 0.914 | 1.428 | 1.845 | ~0.98 |
| Telemetry aggregate (500 events) | 300 | 2.502 | 3.997 | 6.027 | ~2.7 |
| Controller evaluate | 1000 | 2.370 | 3.888 | 6.844 | ~2.6 |
| Reconciler pass | 300 | 3.691 | 20.966 | 24.827 | ~5.0 |
| API request (full lifecycle) | 200 | 3.003 | 4.240 | 4.629 | ~3.1 |
| API `/healthz` | 500 | 1.631 | 2.327 | 2.741 | ~1.7 |

## Interpretation

* Pure mapping/transaction/manifest operations are microseconds — negligible.
* A full in-process workload lifecycle is **sub-millisecond** (p50 0.57 ms).
* The HTTP API adds ~3 ms per lifecycle request (framework overhead).
* Telemetry aggregation and controller evaluation are ~2.5 ms for 500 events and
  scale with event count; the reconciler pass grows with the number of open
  reservations (tail up to ~25 ms).
* The single-writer SQLite store serializes state writes; that is the ceiling for
  concurrent throughput (see Phase F).

## Reproduce

```bash
python validation/performance_baseline.py
```

## Caveats

* These numbers describe orchestration overhead. They say nothing about LLM
  serving throughput, TTFT, or GPU behaviour. GPU comparison is Phase L.
* Reconciler tail latency should be re-measured at production reservation counts.
