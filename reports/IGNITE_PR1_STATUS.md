# Ignite PR #1 Status — RESOLVED ✅

PR: https://github.com/yashlabs-trying/kubbernetd-saver/pull/1 (`fix/code-issues` → `main`)
Head commit: `83742826aa652143aca0fe73cec55156851d212a`

**Required check `k8s-e2e`: PASS.** Unit tests (3.10/3.11/3.12): PASS.
PR state: **OPEN, CLEAN, MERGEABLE** (awaiting authorization to merge — not merged).

## Outcome

All nine `k8s-e2e` E2E scenarios now pass:
1. autonomous idle → zero
2. cold request preserved through proxy
3. three autonomous sleep/wake cycles
4. concurrent multi-model cold wake
5. operator restart + subsequent autonomous lifecycle
6. active request prevents scale-down
7. invalid configuration handling
8. actual readiness delay respected (`response_status=200 elapsed_ms=33202`)
9. 100 concurrent cold requests (`total=100 success=100 failures=0 wakeCount_delta=1`)

## Defects found and fixed (13 commits on the PR branch)

| # | Commit | Defect | Fix |
|---|---|---|---|
| 1 | `a8006f7` | `chmod /usr/local/bin/kind` without sudo → `Operation not permitted` | download to workspace + `sudo mv` |
| 2 | `44967fe` | `config/crd.yaml` `bootstrap.cpuRAM` `type:string default:0` → invalid CRD | `default: "0"` + CRD schema test |
| 3 | `dd92861` | `kubbernetd install` piped `str` into `subprocess.run` → `TypeError` | `text=True` + test |
| 4 | `c1e45cc` | `RequestForwarder` built aiohttp session before the loop → proxy crash-loop | lazy session creation + tests |
| 5 | `cb4df04` | operator RBAC lacked `watch` on CRDs → kopf disabled watching | add `watch` (+events) + test |
| 6 | `b06bd47` | `_trigger_wake` lost held futures when no endpoint → 60s timeout | poll discovery; fail released futures |
| 7 | `8d34274` | RUNNING→SLEEPING invalid transition left zero-worker groups stuck | route via DRAINING; proxy 503 not 500 |
| 8 | `b27aa79` | zero-replica target group initialised RUNNING | read replicas; start SLEEPING |
| 9 | `3dbc209` | CI patched container by wrong name → invalid Deployment | index-based JSON patches |
| 10 | `c9044bb` | clean-install reverted operator to unpullable ghcr image | re-pin CI image + rollout status |
| 11 | **`cdea926`** | **`KUBBERNETD_PROXY_PORT` collides with the Kubernetes Service env var for `kubbernetd-proxy` → operator crashes at import after any restart** | rename to `KUBBERNETD_PROXY_LISTEN_PORT`; defensive `_env_int` + tests |
| 12 | `1f9aaaf` | no E2E 8 diagnostics | dump rg phase/replicas/pods on failure |
| 13 | `8374282` | `grep -c -v ... \|\| echo 0` produced `"0\n0"` → false E2E 9 failure | count with `awk`; failures = TOTAL − SUCCESS |

Defect #11 is the key production bug: it also affects any real cluster where the
`kubbernetd-proxy` Service exists before an operator (re)start. It was already
diagnosed in the repo's `testing/audit/fix_env_collision.py` but never applied to
source until now.

Unit suite: **88 passed**.

## Reproduce
```bash
gh pr checks 1 -R yashlabs-trying/kubbernetd-saver   # all pass
gh run view 38403257029 -R yashlabs-trying/kubbernetd-saver --job 115316342670
```
