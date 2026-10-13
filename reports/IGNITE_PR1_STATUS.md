# Ignite PR #1 Status

PR: https://github.com/yashlabs-trying/kubbernetd-saver/pull/1 (`fix/code-issues` → `main`)
Required check: **`k8s-e2e`** (GitHub-hosted, ubuntu-latest + kind) — **still FAILING**.
`main` is protected; the check must not be bypassed. **No merge performed.**

## Defects found and fixed on the PR branch (real GitHub CI)

The `k8s-e2e` job was failing at the first step and progressively revealed ten
genuine defects. All fixes are committed on `fix/code-issues` with regression tests
where practical (unit suite: **85 passed**).

| # | Commit | Defect | Fix |
|---|---|---|---|
| 1 | `a8006f7` | `chmod /usr/local/bin/kind` without sudo → `Operation not permitted` | download to workspace, `chmod`, `sudo mv` |
| 2 | `44967fe` | `config/crd.yaml` `bootstrap.cpuRAM` `type:string default:0` → invalid CRD | `default: "0"` + CRD schema test |
| 3 | `dd92861` | `kubbernetd install` piped `str` into `subprocess.run` → `TypeError` | add `text=True` + test |
| 4 | `a632856` | no diagnostics on failure | dump pods/logs/events (`if: always()`) |
| 5 | `c1e45cc` | `RequestForwarder` built aiohttp session before the loop → proxy crash-loop | lazy session creation + tests |
| 6 | `cb4df04` | operator RBAC lacked `watch` on CRDs → kopf disabled watching | add `watch` (+events) + test |
| 7 | `b06bd47` | `_trigger_wake` lost held futures when no endpoint → 60s timeout | poll discovery; fail released futures |
| 8 | `8d34274` | RUNNING→SLEEPING invalid transition left zero-worker groups stuck | route via DRAINING; proxy returns 503 not 500 |
| 9 | `3dbc209` | CI patched container by wrong name → invalid Deployment | index-based JSON patches |
| 10 | `b27aa79` | zero-replica target group initialised RUNNING | read replicas; start SLEEPING |
| 11 | `c9044bb` | clean-install reverted operator to unpullable ghcr image | re-pin CI image + rollout status |

## Current E2E result

With these fixes the job now runs the full E2E suite. Passing scenarios:
**1 (autonomous idle→zero), 2 (cold request through proxy), 3 (three sleep/wake
cycles), 4 (concurrent multi-model cold wake), 5 (operator restart + lifecycle),
6 (active request prevents scale-down), 7 (invalid config handling).**

Failing: **8 (actual readiness delay respected)** and **9 (100 concurrent cold
requests)**. In E2E 8 the `slow-model` backend never scales up within the proxy's
60s wake window (`pod=Pending`, `wake timeout`), despite the operator fixes. This
is an operator wake-timing issue under the CI environment that requires further
investigation; it is **not** bypassed or weakened.

## Consequence

The required `k8s-e2e` check remains red, so **PR #1 is not mergeable** and Ignite
stays on `fix/code-issues` (`c9044bb`). The three originally-reported commits plus
ten remediation commits are preserved. `main` is untouched.

## Reproduce
```bash
gh pr checks 1 -R yashlabs-trying/kubbernetd-saver
gh run list -R yashlabs-trying/kubbernetd-saver --limit 1
```
