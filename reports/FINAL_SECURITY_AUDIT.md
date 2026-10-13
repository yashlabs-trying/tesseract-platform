# Final Security Audit

Re-run via `scripts/run_full_stack_validation.sh` (2026-10-13). Raw logs in
`reports/raw/security_*.log`.

| Tool | Scope | Result |
|---|---|---|
| ruff | platform + integrations + validation | **All checks passed** |
| mypy (`--ignore-missing-imports`) | 32 source files | **Success: no issues found** |
| bandit `-ll` | platform + integrations | **High: 0, Medium: 0, Low: 1** |
| pip-audit | validation environment | **No known vulnerabilities** |

* The single bandit Low is `B104` (bind-all-interfaces); the default `api_host`
  is `127.0.0.1` and deployments opt into `0.0.0.0` explicitly.
* Platform API auth (P-02) and the 12 adversarial tests
  (`tests/security/test_adversarial.py`) remain in place.
* The new Foundry/Ignite fixes introduce no new platform findings; Foundry's
  `SafeJSON` preserves Decimal precision as strings and does not execute input.

## Product static analysis (informational)
Product-repo ruff findings are unchanged (Foundry 219, Fusion 726, Ignite 99,
Adaptive 6) — pre-existing; the products' CI does not gate on ruff/mypy.

## Secrets
No credentials or tokens in the repository; `.env` git-ignored; CI includes
gitleaks.

## Verdict
**No unresolved critical or high-severity findings.** Security gate: **PASS**.
