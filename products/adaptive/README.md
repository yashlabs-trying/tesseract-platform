# Tesseract Adaptive

Autonomous, quality-constrained, workload-aware quantization and model
precision optimization.

Adaptive answers one question: **which precision configuration minimizes
inference cost while still satisfying quality, memory, throughput, latency and
compatibility constraints?** It is not another static 4-bit quantizer. It
analyzes a model, measures where precision matters, generates candidate
profiles, validates them against quality, and adapts as serving pressure
changes.

> Status: **Milestones 1-4** are implemented and tested. GPU readiness (M5) is
> planned. Nothing in this repository claims GPU performance that has not been
> measured.

## What works today (M1 + M2)

**Foundation (M1)**

- **Model Analyzer** - normalizes Hugging Face `config.json` files into a
  typed architecture via pluggable family adapters (Llama, Qwen2, Qwen3,
  Mistral). No weights are downloaded.
- **Parameter & storage accounting** - exact parameter counts by layer/family
  and serialized byte accounting at baseline precision.
- **Reference quantizer** - deterministic PyTorch affine quantization
  (symmetric/asymmetric, per-tensor/per-channel/per-group, 4/8-bit) that runs on
  CPU and Apple MPS, with int4 packing.
- **Precision profiles** - baseline, uniform and mixed-precision profile
  generation with runtime compatibility validation.
- **Versioned JSON schemas** and **artifact integrity manifests**.

**Intelligence (M2)**

- **Sensitivity analyzer** - per-layer weight error, activation-weighted error,
  diagonal Hessian-inspired error, gradient-based Fisher importance and
  ground-truth output divergence, with family attribution and rankings.
- **Quality engine** - perplexity, output divergence, task accuracy,
  long-context and structured-output metrics with enforced
  calibration/evaluation dataset isolation and threshold gating.
- **Precision optimizer** - greedy and beam-search optimizers minimizing
  estimated cost under quality/latency/memory/throughput constraints, with a
  Pareto frontier and uniform 4/8-bit baselines.
- **Synthetic reference transformer** - a real small decoder-only transformer
  so M2 runs deterministically on CPU/MPS without downloads.

**Adaptation (M3)**

- **Telemetry** - measured/simulated snapshots and deterministic scenarios
  (steady, memory pressure, traffic burst).
- **Adaptive controller** - downshift/upshift recommendations with hysteresis,
  minimum dwell time, cooldown, per-hour rate limit, safe replica-based switch
  plans, rollback plans, and recommendation-only mode until validated.
- **Simulation & fault injection** - timeline simulation with switch-failure
  injection and automatic rollback.

**Integration (M4)**

- **Foundry** - import `tesseract.performance-profile.v1` measurements; export
  profile comparisons with confidence and provenance.
- **Fusion** - profile memory requirements, placement-constrained precision
  recommendation, and capacity-improvement projection.
- **Ignite** - load prevalidated profiles, plan safe replica transitions, and
  expose readiness/rollback signals.

## Quick start

Requires Python 3.11 or 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch,dev]"

# Analyze a model architecture
adaptive analyze fixtures/models/llama-3.2-1b.json

# Report serialized storage for a uniform 4-bit profile
adaptive storage fixtures/models/llama-3.2-1b.json --bits 4 --group-size 128
```

## Tests

```bash
pytest tests/ -v
ruff check src tests
mypy src
```

Unit, integration, contract and property-based tests run entirely on CPU with
no CUDA, TensorRT, NCCL or bitsandbytes kernels required.

## Repository layout

```
src/adaptive/
  models/        domain models (architecture, quantization, storage, integrity,
                 sensitivity, quality, cost, optimization)
  analyzer/      HF config adapters + registry
  quantize/      reference quantizer, packing, profiles, storage, model apply
  nets/          synthetic reference transformer (M2)
  sensitivity/   M2 - layer sensitivity methods
  quality/       M2 - metrics, dataset isolation, thresholds, harness
  optimizer/     M2 - cost model, quality estimator, greedy/beam, Pareto
  telemetry/     M3 - telemetry snapshots, sources, scenarios
  controller/    M3 - adaptive controller, simulation, rollback
  integration/   M4 - Foundry / Fusion / Ignite adapters
  config/        settings (constraints, controller policy)
schemas/tesseract/  versioned JSON schemas
docs/ARCHITECTURE.md
```

## Development environment

Primary development target is an Apple Silicon Mac (M5) using PyTorch CPU and
MPS. MPS is **not** a substitute for NVIDIA CUDA validation; GPU-only results
are marked as unvalidated until measured.

## License

MIT
