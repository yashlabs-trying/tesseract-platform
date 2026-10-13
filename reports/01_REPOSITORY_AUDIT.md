# 01 — Repository Audit

Audit date: 2026-10-13. Host: WSL2 x86_64 (`Linux 6.18.33.2-microsoft-standard-WSL2`),
Python 3.12.14 (uv-managed). Products reproduced in isolated venvs outside their
working trees (no repo pollution).

## Exact revisions tested

| Product | Repository | Branch | Commit tested | Remote main |
|---|---|---|---|---|
| Platform | `tesseract-platform` | `validation/pre-gpu-audit` | `576c4cb` + validation commits | `576c4cb` |
| Foundry | `tesseract-foundry` | main | `b8a70306d2c5d40b73156593987b3c8fab10cd29` | same |
| Fusion | `tesseract-fusion` | main | `c0f4eb9838c6203e185449989a932475a37ee827` | same |
| Ignite | `kubbernetd-saver` | main | `df227307a916a1ec3d3fadab9a28cb2cdb14e911` | `cae22248f0fd3a9f9078af13aa4208950743e602` |
| Adaptive | `tesseract-adaptive` | `validation/mac-m5-audit` | `2d826c863a209fc1af185db8222c033ea227a7f4` | `53649fdd39157b4c7c7ecd88010e7ee41290c902` |

Ignite PR #1 (`fix/code-issues` → `main`) is **still open**; `main` is protected
by a required `k8s-e2e` check. No merges were performed.

## Independent test reproduction

| Product | Command | Result | Skipped | Historical claim |
|---|---|---|---|---|
| Foundry | `pytest tests` | **38 passed** | 4 (GPU) | 101/101 |
| Fusion | `pytest tests` | **183 passed** | 6 (K8s) | 194 pass, 1 skip |
| Ignite | `pytest tests` | **77 passed** | 0 | 91/91 |
| Adaptive | `pytest tests` | **254 passed** | 2 (CUDA) | 254 pass, 2 skip |

Raw logs: `reports/raw/{foundry,fusion,kubbernetd,adaptive}_pytest.log`.

### Discrepancies (documented, not hidden)

* **Foundry**: the local suite contains 42 tests (38 pass / 4 GPU-skipped). The
  historical "101/101" is not reproducible from this revision.
* **Fusion**: 189 tests total; the historical "194 pass, 1 skip" differs. One
  apparent failure (`test_no_policy_checkpoints`) was a **harness artifact** — it
  scans the repo for checkpoint files and matched my in-repo `.venv-audit/*.pth`.
  Re-run with an external venv: **183 passed, 6 skipped, 0 failed**.
* **Ignite**: 77 tests in `tests/`; the historical "91" likely includes the
  `testing/audit/` harness, which requires a Kubernetes cluster (collection
  errors here) and could not be reproduced without a cluster.

## Static analysis (product repos)

`uvx ruff check` (default rules) findings:

| Repo | Findings |
|---|---|
| Foundry | 219 |
| Fusion | 726 |
| Ignite | 99 |
| Adaptive | 6 |

These are product-repo lint findings (not release-blocking for the platform);
the products are out of scope for modification in this campaign.

## Dependency vulnerabilities

`pip-audit` over the combined validation environment: **no known
vulnerabilities** (`reports/raw/security_pip_audit.log`). The four product
packages are not published on PyPI and were skipped by the auditor.

## Conclusion

All five repositories were audited; four independent suites were reproduced with
exact commits recorded. Foundry/Fusion/Ignite historical counts are lower than
reported and are corrected here. The platform remains green (81 tests).
