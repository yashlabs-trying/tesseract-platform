# 03 — Adaptive Quality Safety

Evidence: `reports/raw/adaptive_quality_safety.json`. Executed against Adaptive's
**real** code (`validation/mac-m5-audit` @ `2d826c8`) on its synthetic reference
model (qwen3-0.6b fixture; no weights downloaded).

## Results

| Check | Result | Evidence |
|---|---|---|
| **C1** quality rejection | PASS | with `allowed_bits=(4,)`, tolerance 0 → uniform-4 rejected; result has no feasible profile (`selected=None`, note "no feasible profile found; recommending baseline precision") |
| **C2** safe fallback | PASS | ladder: tol 0.0/0.01 → mixed `greedy` profile; tol ≥0.05 → `uniform-4bit-per_group`. 4-bit is not selected when it violates tolerance |
| **C3** explicit infeasible | PASS | `memory_budget_gb=0.0001` → selected baseline `uniform-16bit-per_group`, `feasible=false`, violation `memory 1.192 GB > 0.0001 GB`, note "no feasible profile found" |
| **C4** artifact availability | PASS | reference int4 `deployable=false` ("reference backend; not a production kernel"); vLLM AWQ int4 `deployable=true` |
| **C5** hardware compatibility | PASS | MPS supports only `{16,32}`; int4-on-MPS yields 2 compatibility issues and `assert_profile_supported` raises. CUDA supports `{4,8,16,32}`, `production=true` |
| **C6** measured vs estimated | PASS | quality estimator, selected profile, and cost all labelled `estimated`; never presented as measured |
| **C7** transitions | PASS (by coverage) | Adaptive controller test suite (254 tests) + platform `AdaptivePrecisionProvider` protocol tests |
| **C8** optimizer vs heuristic | **PASS (limitation preserved)** | see below |
| **C9** safe default | PASS | platform uses declared precision when Adaptive is absent (`test_adaptive_optional.py`) |

## C8 — Optimizer is not always superior (preserved limitation)

At matched storage (~448 MB), the simple **protect-all-MLP** heuristic
outperforms the sensitivity-guided selection:

| Profile | Predicted degradation | Storage |
|---|---|---|
| uniform-4 (selected) | 0.04509 | 447.1 MB |
| **protect-MLP heuristic** | **0.02783** | 448.9 MB |
| uniform-8 | 0.00541 | 745.1 MB |
| uniform-16 | 0.00000 | 1192.1 MB |

The protect-MLP heuristic achieves **~38% lower predicted degradation at
effectively identical storage**. This confirms the earlier audit finding and is
recorded as a known Adaptive limitation — it is not hidden or manipulated.

## Notes

* All quality figures are **model predictions** (`estimated`), not measured
  perplexity. The reported reference-quantizer measurements (e.g. Qwen2.5-0.5B
  +67.2% at 4-bit) are not re-derived here; they remain the motivating evidence
  that 4-bit is not automatically safe.
* No GPU or CUDA was used. MPS/CPU capability checks only.

## Verdict

Adaptive quality constraints are enforced; unsafe/unavailable profiles are
rejected or fall back to validated precision; the optimizer's limitations are
documented. **PASS (CPU).**
