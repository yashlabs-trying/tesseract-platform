# Final Performance Baseline (CPU)

Source: `reports/raw/performance_baseline.json` (Phase J) and
`reports/raw/cpu_phases.json` (Phase F). Host: WSL2 x86_64, Python 3.12.

## Control-plane overhead

| Operation | p50 | p95 | p99 |
|---|---:|---:|---:|
| Foundry profile mapping | 0.004 ms | 0.007 ms | 0.008 ms |
| Reservation transaction | 0.020 ms | 0.028 ms | 0.038 ms |
| Ignite manifest build | 0.001 ms | 0.001 ms | 0.002 ms |
| Coordinator full lifecycle | 0.573 ms | 0.914 ms | 1.411 ms |
| Rollback completion | 0.914 ms | 1.428 ms | 1.845 ms |
| Telemetry aggregate (500 events) | 2.502 ms | 3.997 ms | 6.027 ms |
| Controller evaluate | 2.370 ms | 3.888 ms | 6.844 ms |
| Reconciler pass | 3.691 ms | 20.966 ms | 24.827 ms |
| API request (full lifecycle) | 3.003 ms | 4.240 ms | 4.629 ms |

## Concurrency scaling (in-process, fixtures)

| Level | Success | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.84 ms | 0.84 ms | 0.84 ms |
| 25 | 25 | 46.7 ms | 53.0 ms | 54.6 ms |
| 100 | 100 | 124.7 ms | 188.4 ms | 235.5 ms |
| 500 | 500 | 136.3 ms | 172.3 ms | 189.6 ms |

## Soak

* 100 reservation/release cycles: 0.106 s; ledger used=0; RSS delta 0 KB.

## Meaning

This is **platform orchestration overhead**, not LLM serving. It is the
reproducible baseline the GPU campaign will compare against (per-experiment
records add hardware, engine, precision and REAL/EMULATED/SIMULATED/ESTIMATED
labels).

## Bottleneck

The single-writer SQLite store serializes state writes and is the throughput
ceiling; latencies plateau near ~136 ms p50 at 500 concurrent requests. Migrate
to Postgres for multi-replica scale-out.
