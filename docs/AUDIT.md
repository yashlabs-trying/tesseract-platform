# Milestone 1 Audit — Verified Source State

Audit performed 2026-10-13 against the local working copies under
`/home/yashs/` and the GitHub remotes of `yashlabs-trying`. Nothing was modified
or pushed as part of the audit.

## Verified commits

| Product | Repository | Visibility | Branch | Local HEAD | Remote `main` | Sync |
|---|---|---|---|---|---|---|
| Foundry | `tesseract-foundry` | public | main | `b8a70306d2c5d40b73156593987b3c8fab10cd29` | same | in sync |
| Fusion | `tesseract-fusion` | private | main | `c0f4eb9838c6203e185449989a932475a37ee827` | same | in sync |
| Ignite | `kubbernetd-saver` | public | main | `df227307a916a1ec3d3fadab9a28cb2cdb14e911` | `cae22248f0fd3a9f9078af13aa4208950743e602` | **3 ahead** |
| Adaptive | `tesseract-adaptive` | private | main | `06e2bf37861323f6a52fe20671992c45599590ff` | same | in sync |
| (legacy) | `tesseract-foundary` | public | main | `2fe7a4407d51585c8c8913caaf51f9972c3baab3` | same | in sync |

## Finding 1 — `foundry` vs `foundary`

**Both repositories exist.** This is not a typo in the assignment.

* `tesseract-foundry` (v1.0.0, Python package `foundry`) is the
  **profiling / performance** product. It ships
  `schemas/tesseract/performance-profile-v1.json`, whose description already
  states *"Foundry generates; Fusion/Ignite consume via REST."* **This is the
  canonical Foundry** and the one the platform integrates against.
* `tesseract-foundary` (v5.0.0, package `tesseract_foundary`) is a
  **model-serving API** (engine, registry, ONNX, A/B testing). It has **no
  collected tests** (`tests/` contains only `__init__.py`). It is a legacy /
  misnamed repository, retained as a possible reference model server but
  **not integrated**.

## Finding 2 — the "uncommitted changes" are line-ending noise

All four product working copies reported many modified files:

| Repo | Modified files | `git diff -w` | `--ignore-cr-at-eol` |
|---|---|---|---|
| `tesseract-foundry` | 46 | empty | empty |
| `tesseract-foundary` | 34 | empty | empty |
| `tesseract-fusion` | 2 | empty | empty |
| `kubbernetd-saver` | 89 | empty | empty |
| `tesseract-adaptive` | 5 + 10 untracked | **non-empty** | **non-empty** |

The first four are **pure CRLF/whitespace differences** (identical insertion and
deletion counts, e.g. 6628/6628). They are **not** semantic fixes, so no local
work was at risk. Only `tesseract-adaptive` has genuine uncommitted work (its M3
controller); it was left untouched.

## Finding 3 — Ignite's validated commits are safe but `main` is protected

`kubbernetd-saver` local `main` is 3 commits ahead of `origin/main`:

* `ddb01e8` Fix code issues: proxy streaming, request cancellation, controller
  state transitions, readiness detection, concurrency and lint
* `f6d131c` Complete single-cluster audit — 9 E2E tests pass, 12 bugs fixed
* `df22730` Gate 2 multi-node testing: node drain, CP restart, cross-node cold-wake

A direct `git push origin main` was **rejected** by branch protection:

```
remote: error: GH006: Protected branch update failed for refs/heads/main.
remote: - Required status check "k8s-e2e" is failing.
```

The commits are already on `origin/fix/code-issues` and proposed via **PR #1**
("Fix code issues across proxy, controller, operator and CLI"). Nothing is lost;
the `k8s-e2e` check must pass on a real cluster (Mac/kind) before merge. A remote
branch `feature/ignite-fusion-v0` also exists at `cae2224` (= `main`).

## Finding 4 — historical test counts are not yet reproduced

Collection under system Python 3.14 (no per-repo venv):

| Repo | Collected | Historical claim |
|---|---|---|
| `tesseract-foundry` | 42 | 101/101 |
| `tesseract-fusion` | collection errors (deps missing) | 194 + 1 skip |
| `kubbernetd-saver` | collection errors (deps missing) | 91/91, 15/15 k8s, 100/100 cold |
| `tesseract-adaptive` | 142 (its `.venv`, Py3.12) | — |

Reproducing these suites requires per-repo virtualenvs; that is tracked as
follow-up work. The counts are treated as **historical evidence, not verified**.

## Finding 5 — tooling

The Windows host has `git` and authenticated `gh` (account `yashlabs-trying`)
but **no Docker, kubectl, kind or helm**. All Linux tooling runs in WSL2 Ubuntu.
The intended Kubernetes target is the Mac at `192.168.1.4` (macOS 26.6.2).

## Integration gaps (real, verified)

1. **Foundry → Fusion evidence mismatch.** Foundry emits
   `output_tokens_per_second` / `completed_requests_per_second` and
   `evidence_type ∈ {synthetic, estimated, measured, validated}`. Fusion's
   `PerformanceProfile` uses `throughput_rps` and a different `EvidenceType`
   (`measured, estimated, simulated, extrapolated, unknown`). Adaptive's
   `integration/foundry.py` expects `throughput_tps`. The platform defines a
   canonical mapping (`integrations/foundry/adapter.py`) and a shared `Evidence`
   vocabulary.
2. **Fusion → Ignite has no reservation protocol.** Fusion only exposes
   `adapters/ignite.py` (`IgniteCapacityContract` = a *recommendation*).
   Ignite exposes a `ReplicaGroup` CRD (`kubbernetd.io/v1`) + proxy + operator,
   with no reservation lifecycle. The platform defines the 9-state lifecycle.
3. **Ignite → Foundry has no telemetry ingest.** Ignite emits metrics but
   Foundry has no lifecycle-telemetry endpoint. The platform defines the
   `tesseract.platform.telemetry.v1` contract and aggregator.
4. **No optimization feedback loop.** Fusion has batch optimization only; the
   platform adds an observe/recommend/act controller with cooldowns, hysteresis
   and an action budget.
5. **Adaptive is mid-development** with uncommitted work. The platform reserves
   the `AdaptivePrecisionProvider` contract and ships a disabled adapter.

## Decisions

* Canonical Foundry = `tesseract-foundry`.
* Platform control-plane package = `tesseract_platform` (not `platform`, which
  would shadow the stdlib `platform` module).
* No product code vendored; integration is via contracts + pinned revisions.
* Product repositories are left unmodified. Ignite's push is blocked by branch
  protection and documented above.
