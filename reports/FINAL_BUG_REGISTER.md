# Final Bug Register

## Fixed during this campaign

### P-01 — Platform: Fusion advisory recommendations rejected
* **Symptom:** workload lifecycle FAILED when Fusion returned a valid plan with
  `execution_eligibility=false` / `approval_status=advisory_only`.
* **Root cause:** `FusionAdapter.map_recommendation` defaulted to
  `require_execution_eligible=True`; the platform (not Fusion) is the executor.
* **Repro:** Phase B live integration against the real Fusion service.
* **Fix:** default changed to `False`; capability retained behind an explicit flag.
* **Regression test:** `tests/unit/test_adapters.py::test_map_recommendation_accepts_advisory_plan`.
* **Status:** FIXED (platform).

### P-02 — Platform: control-plane API had no authentication
* **Symptom:** all `/v1` endpoints were open.
* **Fix:** optional `X-API-Key` auth on mutating endpoints
  (`TESSERACT_API_KEY_ENABLED`/`TESSERACT_API_KEY`), `401` otherwise.
* **Regression tests:** `tests/security/test_adversarial.py::test_api_key_required_when_enabled`,
  `test_open_when_auth_disabled`.
* **Status:** FIXED (platform).

### P-03 — Platform: mypy type errors (9)
* `policy.conservative_default` `**dict` spread; `coordinator` optional
  `reservation_id`; `adaptive` `_fail` missing `NoReturn`.
* **Fix:** explicit kwargs, guard, `NoReturn`. mypy now: **0 errors**.
* **Status:** FIXED (platform).

## Open — product defects (products not modified per campaign decision)

### F-01 — Foundry `/v1/services/recommend` returns 500 (HIGH impact)
* **Symptom:** `AttributeError: 'NoneType' object has no attribute 'cost_per_gpu_hour_usd'`.
* **Root cause:** `foundry/api/__init__.py:191` calls
  `evaluate(c, workload, inp, outp)` but the signature is
  `evaluate(candidate, workload, hardware, input_price, output_price, ...)` — the
  `hardware` argument is omitted, so `hardware` is `None`.
* **Fix (proposed, not applied):**
  `evaluate(c, workload, hardware_map[c.hardware_id], inp, outp)` (build the
  hardware map before the economics loop).
* **Impact:** Foundry HTTP recommendation endpoint unusable. Library path works
  and is what the platform uses.

### F-02 — Foundry `/v1/profiling/jobs` returns 500 (HIGH impact)
* **Symptom:** `sqlalchemy.exc.StatementError: Object of type Decimal is not JSON serializable`.
* **Root cause:** job `params` include `Decimal` (`cost_per_gpu_hour_usd`) and are
  persisted into a JSON column without a serializer.
* **Fix (proposed):** serialise job params with `json.dumps(..., default=str)` or a
  `Decimal`-aware encoder before insert.
* **Impact:** Foundry HTTP profiling endpoint unusable.

### F-03 — Foundry fixture/schema price field mismatch (LOW)
* Fixture uses `input_price_per_million` / `output_price_per_million`; `Scenario`
  defines `output_price_per_million_input_tokens` /
  `output_price_per_million_output_tokens`. Fixture prices are silently ignored.

## Known limitations (documented, not defects)

* **Adaptive C8:** the protect-MLP heuristic beats the sensitivity-guided
  optimizer at matched storage (0.0278 vs 0.0451 degradation @ ~448 MB).
* **Platform:** single-writer SQLite control plane; multi-replica requires
  Postgres + a shared capacity ledger.
* **Foundry/Adaptive schema drift:** Adaptive `integration/foundry.py` reads
  `throughput_tps`; Foundry emits `output_tokens_per_second`.

## Pending validation (not defects; not yet exercised)

* Kubernetes lifecycle, node loss, network partition, image-pull failure.
* 30-minute soak; multi-model contention at the cluster layer.
* GPU validation (NVIDIA).

## Test-harness artifact (not a product bug)

* Fusion `test_no_policy_checkpoints` fails only when a `.venv*` directory exists
  inside the repo (it scans for checkpoint files). Re-run with an external venv:
  183 passed, 6 skipped, 0 failed.
