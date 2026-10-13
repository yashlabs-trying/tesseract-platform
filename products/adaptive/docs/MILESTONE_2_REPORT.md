# Milestone 2 Report - Intelligence

Date: 2026-10-13
Host: Ubuntu (WSL2) x86_64, CPU-only PyTorch 2.14.1+cpu, Python 3.12.14.
No CUDA used. Primary target remains Apple Silicon M5.

## Files created

Source (`src/adaptive/`):
- `nets/reference_transformer.py`, `nets/__init__.py` - real small
  decoder-only transformer (RMSNorm, GQA, SwiGLU, causal attention) with
  HF-style layer names, causal LM loss, token batching.
- `models/sensitivity.py`, `models/quality.py`, `models/cost.py`,
  `models/optimization.py` - new domain models.
- `quantize/model.py` - apply a profile to a live `nn.Module` by simulating
  dequantized weights.
- `sensitivity/analyzer.py`, `sensitivity/__init__.py` - five sensitivity
  methods + report.
- `quality/datasets.py`, `quality/metrics.py`, `quality/harness.py`,
  `quality/__init__.py` - datasets, metrics, thresholds, experiment manifest.
- `optimizer/cost.py`, `optimizer/quality_estimator.py`, `optimizer/pareto.py`,
  `optimizer/engine.py`, `optimizer/__init__.py` - cost model, estimator,
  greedy + beam optimizers, Pareto frontier.
- `quantize/profiles.py` - extended to support 16-bit (float) mixed overrides
  and `config_for_bits`.
- `schemas.py` - sensitivity/quality/optimization serialization.
- `cli/main.py` - `sensitivity` and `optimize` commands.

Schemas: `schemas/tesseract/{sensitivity-report,quality-report,optimization-result}-v1.json`

Fixtures: `fixtures/models/tiny-llama.json`

Tests: `tests/unit/{test_sensitivity,test_quality,test_optimizer}.py`,
`tests/property/test_optimizer_properties.py`,
`tests/integration/test_m2_pipeline.py`,
`tests/contract/test_m2_schema_contract.py`, plus `tests/conftest.py` fixtures.

Examples: `examples/demo_intelligence.py`.

## Features implemented

- **Sensitivity analyzer** with five methods, cheapest first: weight
  reconstruction error, activation-weighted error, diagonal Hessian-inspired
  error, gradient-based Fisher importance, and ground-truth output divergence
  (per-layer KL). Normalized scores, deterministic rankings, and attention/MLP/
  head family attribution. No exact-Hessian accuracy is claimed.
- **Quality engine**: perplexity, long-context perplexity, output divergence,
  top-1 agreement, task accuracy, structured-output validity, and calibration
  reconstruction (diagnostic only). Structural **calibration/evaluation
  isolation** and **quality threshold enforcement** (`QualityGateError`).
  Pinned `ExperimentManifest` with fingerprint.
- **Optimizer**: analytic cost model, sensitivity-driven quality estimator,
  greedy layer upgrades and beam search, uniform 4/8-bit baselines, Pareto
  frontier, and feasibility checks for runtime compatibility, quality
  tolerance, memory budget, P95 latency and minimum throughput.

## Tests

Command: `pytest tests/ -q` plus `ruff check src tests`, `mypy src`.

- Passed: **119** (was 73 at M1)
- Failed: **0**; Skipped: **0**
- Ruff: clean. Mypy (strict): clean, 49 source files.
- Coverage: ~92%.

M2 acceptance areas covered: sensitivity ranking consistency (cross-method
Spearman, zeroed-layer ordering, determinism), calibration/evaluation dataset
isolation, quality threshold enforcement, Pareto frontier correctness
(incl. property-based antichain + domination), unsupported configuration
rejection (runtime/bit compatibility), and property-based optimizer-constraint
tests.

## Real models tested

- Synthetic reference transformer built from `fixtures/models/tiny-llama.json`
  (Llama family, 2 layers, 32 hidden). Real weights are intentionally not
  downloaded on the CPU development host; HF adapters remain available.

## Actual quality metrics

Synthetic reference model (`examples/demo_intelligence.py`), uniform profiles,
group size 16, perplexity over the evaluation split:

| profile | perplexity | output divergence (KL) |
| --- | ---: | ---: |
| baseline (16-bit) | 65.166 | 0.00000 |
| uniform 8-bit | 65.169 | 0.00000 |
| uniform 4-bit | 65.070 | 0.00007 |

These numbers characterise the *harness* on a randomly-initialised model; they
are not claims about any real pretrained model's quality.

## Actual serialized size and cost

Tiny reference model, uniform 4-bit per-group (g=16): **0.0177 MB**, analytic
cost **0.000005** units (estimated, from bytes moved). Optimizer selected the
uniform 4-bit profile under a 5% tolerance; Pareto frontier contained all 4
candidates.

## Sensitivity example (activation_weighted, tiny model)

1. `model.layers.0.self_attn.k_proj`
2. `model.layers.1.mlp.up_proj`
3. `model.layers.1.mlp.gate_proj`

## Limitations

- Quality metrics are measured on a synthetic reference model; real-model
  quality (perplexity/accuracy) and GPU timing remain unmeasured.
- The sensitivity-driven quality estimator and analytic cost model are
  heuristic/estimated, clearly labelled; they must be replaced by measured
  quality reports and Foundry benchmark imports before production claims.
- Diagonal Hessian is a diagonal second-order approximation, not an exact
  Hessian; Fisher uses empirical squared gradients on the calibration set.
- Embedding layers are not yet sensitivity-analyzed (linear layers only).
- Optimizer generates mixed 4/8/16-bit profiles; per-group scale metadata uses
  the fixed-overhead assumption from M1.

## Bugs found and fixed during M2

1. Asymmetric KL could be slightly negative from float error; clamped to 0.
2. `mixed_profile` did not support 16-bit (float) overrides; added
   `config_for_bits`.
3. `ReferenceTransformer` used a dynamically-attributed `nn.Module` container
   that defeated static typing; replaced with a typed body class.
4. Test injections based on weight outliers were unreliable (symmetric
   per-group quantisation is scale-invariant); replaced with deterministic
   zeroed-layer ordering assertions.

## Remaining GPU-only validation

Unchanged from `docs/GPU_VALIDATION_PLAN.md`: production kernels, real
TTFT/ITL/throughput/VRAM, multi-GPU placement, and real-model quality on GPU
runtimes.

## Next recommended engineering action

Begin **Milestone 3**: implement the adaptive runtime controller (telemetry
ingestion, recommendation-only decisions, hysteresis/min-dwell/cooldown/rate
limits, rollback simulation) and its acceptance tests. Do not enable automatic
switching until simulation and integration tests pass.
