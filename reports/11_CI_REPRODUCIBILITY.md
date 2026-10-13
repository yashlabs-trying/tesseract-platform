# 11 — CI & Reproducibility

## Single entry point

`scripts/run_full_stack_validation.sh [--keep-cluster] [--skip-products] [--gpu]`

It performs: dependency checks → pinned-commit verification → isolated
environments (uv) → product regression → platform tests → kind cluster (if
available) → deploy → Kubernetes E2E → integration/concurrency/chaos/telemetry
harnesses → security tools → benchmark collection → scorecard → cleanup
(unless `--keep-cluster`).

Verified run on WSL2 (this session):

```
RESULT: PASS=13 FAIL=0 NOT_TESTED=2
Scorecard: reports/SCORECARD.md
```

`NOT_TESTED`: kubernetes lifecycle (no cluster), GPU validation (no NVIDIA HW).

## CI jobs (`.github/workflows/`)

| Workflow | Purpose |
|---|---|
| `ci.yml` | contracts, ruff, docs check, unit+contract+integration tests (py3.11/3.12) |
| `integration.yml` | in-process integration tests; optional Kubernetes job (manual dispatch) |
| `security.yml` | pip-audit, bandit, gitleaks |
| `release.yml` | build + GitHub release on tags |

Actions are version-pinned; workflow permissions are `contents: read` (write only
for release). GPU tests are never required in ordinary CI; the Kubernetes job is
`workflow_dispatch`-only.

## Reproducibility controls

* `platform.manifest.yaml` pins product commits; the entry point verifies them.
* Isolated `uv` environments (Python 3.12) avoid host drift.
* Raw evidence written to `reports/raw/*.json` and `*.log`.
* `reports/SCORECARD.md` generated per run.

## Resource hygiene

The script uses bounded retries/timeouts and deletes the kind cluster unless
`--keep-cluster` is passed. No credentials are written to disk. Temporary venvs
live under `~/.venvs/` and product working trees are left clean.

## Verdict

**PASS** on this host; Kubernetes/GPU jobs remain gated.
