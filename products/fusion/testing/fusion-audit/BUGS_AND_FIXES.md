# BUGS AND FIXES

Audit of Tesseract Fusion on Apple Silicon Mac M5.
Every defect below was **reproduced with a failing test first**, then fixed,
then re-verified. No test was weakened to make a result disappear.

Source baseline: commit `4539cb1` (transferred to the Mac), audit fixes applied
in place.

| # | Severity | Component | Defect | Reproduction | Fix | Regression test |
|---|----------|-----------|--------|--------------|-----|-----------------|
| 1 | **Critical** | `optimizer/global_allocator.py` | The fleet allocator always recommended the zero-cost `__baseline__` plan. Because candidate alternatives never populated `estimated_revenue_per_hour`, every GPU placement had contribution `-cost`, while the baseline had contribution `0`. Maximising contribution therefore always selected "do nothing". | `tests/audit/test_optimizer_oracle.py::test_single_workload_single_gpu` (xfail) | Exclude `__baseline__` from the candidate search space; require every feasible workload to be placed; objective = maximise satisfied workloads (priority-weighted) then minimise cost. | Same test now passes; `test_optimality_gap_small` gap = 0 |
| 2 | **Critical** | `optimizer/candidate.py` | `generate_alternatives` only emitted replicas where `replicas != worker_count`, so the natural 1:1 placement was never a candidate. Only scale-up options existed. | `test_optimizer_engine.py::test_memory_check_uses_per_replica_accounting` | Generate all replica counts (deduplicated, `>= 1`), including `worker_count`. | Updated test asserts replica 1 and 2 both present |
| 3 | **High** | `optimizer/global_allocator.py` | Objective rewarded total replica count, so the optimizer over-provisioned (chose 2 replicas when 1 sufficed) to inflate the score. | `test_optimizer_oracle.py::test_optimality_gap_small` (gap 11) | Reward is per **satisfied workload** (replica_count ≥ required), not per replica. | Gap = 0 |
| 4 | **High** | `optimizer/global_allocator.py` | Exhaustive search capped at 10 000 combinations and iterated the *last* workload fastest, so with ≥ 5 workloads and many candidates it explored almost none of the space and returned an empty plan. | `test_baseline_competition.py::test_equal_workload_served` (empty plan) | Compute the product size; use exhaustive only when it fits the cap, otherwise fall back to CP-SAT, then greedy. Also reorder candidate generation (replica-outer, GPU-inner) so the cap covers all GPUs. | `test_equal_workload_served` now serves 6/6 |
| 5 | **High** | `optimizer/engine.py` | `OptimizationEngine` optimised each workload independently and could recommend the same physical GPU slots to multiple deployments (double-booking in advisory output). | `test_concurrency_races.py::test_two_workloads_compete_single_gpu` | Added fleet-wide slot + memory accounting across deployments within a single `optimize()` call; placements that exhaust capacity are rejected with a reason. | Concurrency suite (levels 1–50) passes |
| 6 | **Medium** | `models/types.py` | Pydantic accepted negative `memory_capacity_mb`, `available_memory_mb`, `available_resource_slots`, `resource_cost_per_hour`, `worker_count`. | `test_boundary_cases.py::test_invalid_negative_values` | Added `Field(ge=0)` constraints. | Updated test asserts `ValidationError` |
| 7 | **Medium** | `optimizer/candidate.py` | Unhealthy GPUs were filtered out before validation, so they never appeared in `rejected_alternatives` and the rejection was invisible. | `test_boundary_cases.py::test_unavailable_node` | Offer unhealthy GPUs as candidates; the constraint validator rejects them explicitly with a `gpu_health` reason. | Updated test asserts `gpu-0` ∈ rejected with `gpu_health` |
| 8 | **Medium** | `api/app.py` | `/v1/workloads/import` raised an uncaught pydantic `ValidationError` for malformed entries → HTTP 500. | `test_security.py::test_malformed_workload_content_rejected` | Catch `ValidationError`, return HTTP 422 with error detail. | Test asserts 422 |
| 9 | **Medium** | `config/settings.py` | `api_key` was a plain `Optional[str]`, so `repr(Settings())` leaked the secret. | `test_security.py::test_secrets_not_logged` | Changed to `SecretStr`. | Test asserts secret not in repr |
| 10 | **Medium** | `policies/validator.py` | `TenantPolicy.resource_quotas` was stored but never enforced; only `max_concurrent_workers` was checked. | `test_security.py::test_resource_quota_respected` | Added `_check_resource_quotas`; also enforced in `GlobalAllocator`. | Test asserts quota violation rejected |
| 11 | **Medium** | `optimizer/engine.py`, `global_allocator.py` | Memory double-counting: `DeploymentAlternative.expected_memory_mb` is already the total across replicas, but the feasibility checks multiplied by `replica_count` again, over-rejecting valid placements. | Introduced and caught during fix #5 verification | Derive the total from the deployment's per-replica value; treat `alt.expected_memory_mb` as total. | Oracle + fleet tests pass |

## Notes

- Bugs #1–#5 were latent correctness defects that made the optimizer either
  useless (never placing workloads) or unsafe (double-booking). They were the
  single most important finding of the audit.
- Bugs #6–#11 are hardening/robustness issues.
- Three of the audit's `xfail` markers (documenting #8, #9, #10) were converted
  into ordinary passing tests after the fixes — the assertions were *strengthened*,
  not weakened.
- After the fixes the full Mac suite is **187 passed, 1 skipped, 1 deselected**,
  and the optimizer matches the independent brute-force oracle exactly
  (optimality gap = 0) on small instances.

## Unresolved defects / known limitations

These are **not** fixed in this audit and are documented rather than hidden:

1. `FleetSimulator` does not populate latency percentiles (`latency_p50/p99`,
   queue durations are always `None`); SLO violations are synthetic counters.
   → Real SLO guarantees cannot be validated without a queueing/latency model.
2. `OptimizationEngine` and `GlobalAllocator` are two separate optimizers with
   different semantics; they can disagree. The allocator is the fleet-authoritative one.
3. The API uses in-memory dictionaries for state; persistence (SQLAlchemy ORM) is
   defined but not wired into the request path.
4. No authentication, rate limiting, or input-size limits by default.
5. External adapters (Prometheus, OpenCost, Dynamo, KAI) are interface-complete
   but not validated against live upstream services.
