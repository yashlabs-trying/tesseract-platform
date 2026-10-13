# 09 — Security Audit

Host: WSL2 x86_64. Tools: `ruff`, `mypy`, `bandit`, `pip-audit`. Raw logs in
`reports/raw/security_*.log`.

## Tool results

| Tool | Scope | Result |
|---|---|---|
| ruff | platform + integrations + validation | **All checks passed** |
| mypy (`--ignore-missing-imports`) | 32 source files | **Success: no issues found** |
| bandit `-ll` | platform + integrations | **High: 0, Medium: 0, Low: 1** |
| pip-audit | validation environment | **No known vulnerabilities** |

Bandit: the single Low finding is `B104` (bind-all-interfaces). The default
`api_host` was changed `0.0.0.0 → 127.0.0.1`; deployments set `0.0.0.0`
explicitly. The two `B608` SQL findings are false positives (parameterised `?`
markers) and carry justified `# nosec B608` annotations.

## Defect found and fixed during Phase I

**P-02 (platform, fixed): the control-plane API had no authentication.** Added
optional `X-API-Key` auth (`TESSERACT_API_KEY_ENABLED` / `TESSERACT_API_KEY`)
applied to all mutating `/v1` endpoints; `401` on missing/incorrect key. Tests:
`test_api_key_required_when_enabled`, `test_open_when_auth_disabled`.

## Adversarial input tests (12, all pass)

`tests/security/test_adversarial.py`:

* API key required when enabled; open when disabled.
* Resource-name sanitisation (`dns_name`) for `../../etc/passwd`, `a/../b`,
  `x; rm -rf /`, uppercase, 300-char, `$$$(whoami)` — no `/`, `..`, spaces; DNS
  length bound.
* Ignite `ReplicaGroup`/`Deployment` manifests sanitise malicious model/workload
  IDs (no path traversal in Kubernetes object names).
* Invalid telemetry schema rejected (`422`).
* Unsafe URL scheme (`file://`) rejected with `IntegrationUnavailable`.
* Malformed workload rejected (`422`, not `500`).

## RBAC

`deployments/kubernetes/rbac.yaml` grants only in-namespace `get/list/watch/
create/update/patch/delete` on `kubbernetd.io/replicagroups` and
`apps/deployments`, and read on pods/services/endpoints/configmaps. No cluster-
wide permissions. `runAsNonRoot`, dropped capabilities, `allowPrivilegeEscalation:
false` in the pod spec.

## Product static findings (informational)

`ruff` on product repos: Foundry 219, Fusion 726, Ignite 99, Adaptive 6. These
are product-repo findings; products were not modified in this campaign.

## Secret exposure

No credentials or tokens are present in the repository. CI includes a `gitleaks`
job. `.env` is git-ignored; only `.env.example` is committed.

## Verdict

**No unresolved critical or high-severity findings.** Medium/High bandit findings
are zero. Security gate: **PASS**.
