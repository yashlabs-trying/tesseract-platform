# Validation Scorecard

Generated: 2026-10-13T10:09:20Z
Host: Linux 6.18.33.2-microsoft-standard-WSL2 x86_64

| Check | Result | Detail |
|---|---|---|
| product:foundry | PASS | 38 passed, 4 skipped, 16 warnings in 0.49s |
| product:fusion | PASS | 183 passed, 6 skipped, 37344 warnings in 46.24s |
| product:kubbernetd | PASS | 77 passed in 6.68s |
| product:adaptive | PASS | 254 passed, 2 skipped in 3.37s |
| platform tests | PASS | 81 passed, 1 warning in 0.52s |
| kubernetes lifecycle | NOT TESTED | no kind/kubectl on this host |
| live integration (Foundry/Fusion) | PASS | see reports/raw/live_integration.json |
| concurrency/chaos/telemetry | PASS | see reports/raw/cpu_phases.json |
| adaptive quality safety | PASS | see reports/raw/adaptive_quality_safety.json |
| performance baseline | PASS | see reports/raw/performance_baseline.json |
| ruff | PASS | clean |
| mypy | PASS | clean |
| bandit | PASS | no High findings |
| pip-audit | PASS | no known vulnerabilities |
| GPU validation | NOT TESTED | --gpu not set (requires NVIDIA hardware) |

Totals: PASS=13 FAIL=0 NOT_TESTED=2
