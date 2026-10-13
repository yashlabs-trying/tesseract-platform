# Tesseract Adaptive - Architecture

## 1. Purpose

Adaptive selects the precision configuration that minimizes inference cost
while satisfying customer-defined quality, latency, memory, throughput and
compatibility constraints. It never assumes a model can reach 4-bit without
quality loss; a configuration is acceptable only when measured evidence
supports it.

## 2. Module map

```
                         ┌──────────────────────┐
   HF config.json ──────▶│  A. Model Analyzer   │
                         └──────────┬───────────┘
                                    │ ModelArchitecture
                         ┌──────────▼───────────┐
   calibration data ────▶│ B. Sensitivity       │
                         │    Analyzer          │
                         └──────────┬───────────┘
                                    │ SensitivityReport
                         ┌──────────▼───────────┐
                         │ C. Profile Generator │
                         │    + reference quant │
                         └──────────┬───────────┘
                                    │ PrecisionProfile[]
                         ┌──────────▼───────────┐
                         │ D. Quality Engine    │◀── eval data (separate)
                         └──────────┬───────────┘
                                    │ QualityReport
                         ┌──────────▼───────────┐
                         │ E. Optimizer         │◀── constraints
                         └──────────┬───────────┘
                                    │ Pareto frontier
                         ┌──────────▼───────────┐
   telemetry ──────────▶│ F. Runtime Controller│──▶ recommendation / plan
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ G. Integration APIs  │  Foundry / Fusion / Ignite
                         └──────────────────────┘
```

Milestones 1 and 2 implement A (Analyzer), B (Sensitivity), C (Profile
Generator + reference quantizer), D (Quality Engine), E (Optimizer) and the
storage/accounting foundation. F (Controller) and G (Integration) are
scaffolded with documented interfaces in their packages.

## 3. A. Model Analyzer

Adapters translate a Hugging Face `config.json` mapping into a normalized
`ModelArchitecture`. Adapters are registered in an `AdapterRegistry` and
selected by `model_type`; no architecture is hardcoded into the core.

Extracted fields: layer structure, weight shapes, attention/MLP projections,
total parameters, baseline precision, estimated weight storage and
quantization compatibility.

Supported families (M1): `llama`, `qwen2`, `qwen3`, `mistral` (dense
decoder-only). The dense adapter handles the shared block layout and exposes
family hooks (`attention_bias`, `has_qk_norm`, `tie_word_embeddings`).

## 4. B. Sensitivity Analyzer (M2)

Planned methods, ordered by cost: activation-weighted reconstruction error,
calibration-based perturbation, diagonal Hessian-inspired approximation, and
optional Fisher-style importance. Attention and MLP families are attributed
separately. Hessian/Fisher accuracy is never claimed before the actual
approximation is implemented and validated against reference output
divergence.

## 5. C. Quantization Profile Generator

- Reference FP16/BF16 (`baseline_profile`).
- Uniform 8-bit and 4-bit (`uniform_profile`).
- Mixed 4/8/16-bit layer configurations (`mixed_profile`).
- Configurable granularity: per-tensor, per-channel, per-group.
- Symmetric or asymmetric affine schemes with per-group scales/zero points.
- Runtime compatibility metadata (`validate_profile`, `assert_profile_supported`).

The reference quantizer (`quantize/reference.py`) is deterministic PyTorch
that runs on CPU and MPS. It is a correctness reference, not a production
kernel. Production backends (GPTQ/AWQ/bitsandbytes/TensorRT) are adapters over
existing libraries, not reimplementations.

Storage accounting (`quantize/storage.py`) tracks packed payload, scales, zero
points, per-tensor metadata and retained unquantized parameters. A profile is
never labelled deployable unless the declared runtime can execute it; the
reference integer path is explicitly non-production.

## 6. D. Quality Evaluation Engine (M2)

Configurable suites: perplexity, calibration reconstruction error,
reference-vs-quantized divergence, task accuracy, long-context quality and
structured-output correctness. Calibration and final evaluation datasets are
separate, and all datasets, model revisions, seeds and settings are pinned.
Reconstruction error alone is never accepted as proof of quality.

## 7. E. Precision Optimization Engine (M2)

Constrained optimizer minimizing estimated/measured inference cost subject to
quality tolerance, P95 latency SLO, memory budget, runtime compatibility,
minimum throughput and safe-deployment constraints. Starts with greedy layer
upgrades plus a beam-search alternative, uses sensitivity to seed candidates,
compares against uniform 4-bit and 8-bit baselines, and tracks the Pareto
frontier. Lower precision is not assumed to be faster.

## 8. F. Adaptive Runtime Controller (M3)

Consumes memory pressure, queue depth, TTFT, ITL, token throughput, model
demand, profile loading cost and quality certificates. Emits a recommended
profile, projected memory and cost impact, a switching plan and a rollback
plan. Enforces hysteresis, minimum dwell time, cooldowns and rate limits.
Operates recommendation-only until simulation and integration tests pass, and
never performs unsafe in-place weight replacement during active inference.

## 9. G. Integration Interfaces (M4)

Versioned APIs, each product independently usable:

- **Foundry** - import benchmark measurements, export profile comparisons,
  report confidence and provenance. Reuses the
  `tesseract.performance-profile.v1` schema shape.
- **Fusion** - report profile memory requirements, recommend precision under
  placement constraints, project capacity improvements.
- **Ignite** - load prevalidated profiles, perform safe replica transitions,
  expose readiness and rollback signals.

## 10. Mac M5 development strategy

- Python 3.11/3.12, PyTorch CPU, Apple MPS when supported.
- Small real transformer models and synthetic fixtures.
- Local benchmarking, deterministic simulation and mock GPU telemetry.
- Measured hardware results are always separated from simulated estimates via
  `EvidenceType`.
- No CUDA, TensorRT, bitsandbytes CUDA kernels or NCCL in the default suite.

## 11. Evidence discipline

Every quantity carries provenance: `synthetic`, `estimated`, `measured` or
`validated`. M1 produces `estimated` values from structure and arithmetic.
Nothing is promoted to `measured`/`validated` without a real run, and no fake
GPU benchmark results are generated.

## 12. Milestones

- **M1 Foundation** - architecture, analyzer, reference quantizer, storage
  accounting, schemas, integrity. *(implemented)*
- **M2 Intelligence** - sensitivity, mixed-precision optimizer, quality
  evaluations. *(implemented)*
- **M3 Adaptation** - telemetry-driven controller, policies, hysteresis,
  rollback simulation.
- **M4 Integration** - Foundry/Fusion/Ignite adapters and end-to-end Mac tests.
- **M5 GPU readiness** - production backend adapters, GPU test manifests and
  benchmark scripts.

Each milestone has explicit acceptance tests; work does not advance while
critical tests fail.
