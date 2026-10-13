# Foundry API Fixes (F-01, F-02, F-03)

Branch: `fix/pre-gpu-api-blockers` · PR: https://github.com/yashlabs-trying/tesseract-foundry/pull/1
Commit: `ca2bd6a` · Baseline: `b8a7030`

## F-01 — HTTP 500 on `/v1/services/recommend` and `/v1/economics/evaluate`
* **Root cause:** `foundry/api/__init__.py` called
  `evaluate(candidate, workload, input_price, output_price)` but the signature is
  `evaluate(candidate, workload, hardware, input_price, output_price, ...)` — the
  `hardware` argument was omitted, so `hardware` was `None` (or a price), and
  `evaluate` raised `AttributeError: 'NoneType' object has no attribute 'cost_per_gpu_hour_usd'`.
* **Fix:** build the hardware map from the scenario (recommend) or look up the
  registered hardware store (economics) and pass it; return **422** with a clear
  message when hardware is unknown. No hardware is hardcoded.
* **Verified:** real HTTP `recommend` → 200 (`approval_status=unverified_only`
  for synthetic fixtures, which is the correct honest result); `economics` → 200;
  unknown hardware → 422.

## F-02 — HTTP 500 on `/v1/profiling/jobs`
* **Root cause:** job `params` (containing `Decimal` cost values) were persisted
  into a SQLAlchemy `JSON` column; `json.dumps` raised
  `TypeError: Object of type Decimal is not JSON serializable`.
* **Fix:** added a `SafeJSON` `TypeDecorator` that encodes `Decimal` as a
  **string** (precision preserved — never silently floated) and handles
  datetime/enum values. Applied to `jobs.params` and the recommendation JSON columns.
* **Verified:** `POST /v1/profiling/jobs?mode=fixture` → 200 and the job is
  retrievable (proves the Decimal params persisted).

## F-03 — Price-field mismatch
* **Root cause:** fixtures used `input_price_per_million` /
  `output_price_per_million`; the canonical `Scenario` fields are
  `output_price_per_million_input_tokens` / `output_price_per_million_output_tokens`,
  so fixture prices were silently ignored and default prices were used.
* **Fix:** corrected all three scenario fixtures; added a `Scenario` validator
  that rejects negative or one-sided (incomplete) price pairs instead of silently
  defaulting.
* **Verified:** all fixtures parse with non-null prices; negative/incomplete
  pricing rejected.

## Tests
New `tests/integration/test_api_blockers.py` (9 tests). Full suite:
**47 passed, 4 skipped** (GPU). Foundry CI on PR #1: **pass**.

## Reproduce
```bash
uv venv --python 3.12 /tmp/fv && uv pip install --python /tmp/fv/bin/python -e ".[dev]"
/tmp/fv/bin/python -m pytest tests -q          # 47 passed, 4 skipped
uvicorn foundry.api:app --port 8001 &          # then POST the endpoints
```
