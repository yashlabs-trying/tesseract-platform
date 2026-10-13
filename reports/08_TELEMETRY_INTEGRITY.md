# 08 — Telemetry Integrity

Evidence: `reports/raw/cpu_phases.json` (phase H) + `tests/unit/test_telemetry.py`
+ `tests/contracts/test_contracts.py`.

## Verified properties

| Property | Result |
|---|---|
| Shared correlation IDs preserved across lifecycle | PASS (`test_correlation_id_preserved_across_lifecycle`) |
| Failed requests do **not** inflate successful throughput | PASS (3 success / 2 fail → throughput uses successes only) |
| Error rate computed over all requests | PASS (0.4 for 2/5) |
| TTFT percentile uses successful requests only | PASS (p50 = 100.0, not the 1.0 from failed) |
| Duplicate telemetry event de-duplicated (same `event_id`) | PASS |
| Out-of-order timestamps aggregate correctly | PASS (first < last) |
| Missing telemetry handled (empty summary) | PASS |
| Contract validation on ingest | PASS (unknown `event_type` → rejected) |
| Token counting | PASS (`tokens_output_total` sums successes only) |

## Foundry field verification

* `output_tokens_per_second` is the **canonical** Foundry throughput field and is
  mapped to `WorkloadProfile.output_tokens_per_second`.
* `completed_requests_per_second` is mapped separately.
* The earlier incorrect `throughput_tps` mapping is **not** reintroduced; the
  Adaptive `integration/foundry.py` adapter's `throughput_tps` expectation is
  flagged in `docs/INTEGRATION_CONTRACTS.md` as a discrepancy to reconcile.

## Provenance labelling

Every telemetry event carries `evidence ∈ {synthetic, estimated, measured,
validated}` and a `source`. Fixture-derived profiles are labelled **synthetic**
(verified in Phase B: `evidence_type="synthetic"`). Benchmark records identify
REAL/EMULATED/SIMULATED/ESTIMATED in `benchmarks/runner.py` output.

## Consistency

* Timestamps are timezone-aware UTC (`datetime.now(timezone.utc)`).
* Units: milliseconds for latency, requests/second for throughput, tokens for
  counts. Enforced by the `tesseract.platform.telemetry.v1` schema.

## Verdict

Telemetry integrity: **PASS** for the tested cases. Cross-process clock skew and
Kubernetes lifecycle telemetry remain to be validated in Phase E.
