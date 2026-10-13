# Tesseract Adaptive

Autonomous, quality-constrained, workload-aware quantization and model
precision optimization.

Adaptive answers one question: **which precision configuration minimizes
inference cost while still satisfying quality, memory, throughput, latency and
compatibility constraints?** It is not another static 4-bit quantizer. It
analyzes a model, measures where precision matters, generates candidate
profiles, validates them against quality, and adapts as serving pressure
changes.

> Status: **Milestone 1 (Foundation)** is implemented and tested. Later
> milestones (sensitivity, optimization, adaptation, integration) are
> scaffolded with documented interfaces only. Nothing in this repository claims
> GPU performance that has not been measured.

## What works today (M1)

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
- **Serialized storage accounting** - packed payload, scales, zero points,
  metadata and retained unquantized parameters.
- **Versioned JSON schemas** and **artifact integrity manifests**.

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
  models/        domain models (architecture, quantization, storage, integrity)
  analyzer/      HF config adapters + registry
  quantize/      reference quantizer, packing, profiles, storage accounting
  config/        settings (constraints, controller policy)
  sensitivity/   M2 - layer sensitivity (interface only)
  quality/       M2 - quality evaluation (interface only)
  optimizer/     M2 - constrained precision optimizer (interface only)
  controller/    M3 - adaptive runtime controller (interface only)
  telemetry/     M3 - telemetry ingestion (interface only)
  integration/   M4 - Foundry / Fusion / Ignite adapters (interface only)
schemas/tesseract/  versioned JSON schemas
docs/ARCHITECTURE.md
```

## Development environment

Primary development target is an Apple Silicon Mac (M5) using PyTorch CPU and
MPS. MPS is **not** a substitute for NVIDIA CUDA validation; GPU-only results
are marked as unvalidated until measured.

## License

MIT
