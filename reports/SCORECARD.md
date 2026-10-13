# Validation Scorecard

Generated: 2026-10-13T12:28:46Z
Host: Linux 6.18.33.2-microsoft-standard-WSL2 x86_64

| Check | Result | Detail |
|---|---|---|
| product:foundry | PASS | 47 passed, 4 skipped, 21 warnings in 1.11s |
| product:fusion | PASS | 183 passed, 6 skipped, 35661 warnings in 48.37s |
| product:kubbernetd | PASS | 85 passed, 1 warning in 7.26s |
| product:adaptive | PASS | 254 passed, 2 skipped in 4.99s |
| platform tests | PASS | 81 passed, 1 warning in 1.06s |
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
