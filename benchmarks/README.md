# Benchmarks

The harness (`runner.py`) exercises the real coordinator in-process against
fixture providers and writes JSON + Markdown reports to `reports/` with full
provenance (git commit, pinned product commits, environment, scenario).

```
python benchmarks/runner.py --scenario benchmarks/scenarios/local-lifecycle.json
```

## Important

* These are **fixture** baselines. They measure orchestration overhead, not
  model-server or GPU performance.
* They must never be reported as production throughput or GPU validation.
* The regression detector compares against `baselines/<scenario>.json` and flags
  a >20% p95 increase. Small samples are **not** statistically significant and
  are reported as "suspected" only.

## GPU scenarios

GPU scenarios live alongside these and are run only on real hardware; see
`docs/GPU_VALIDATION.md`.
