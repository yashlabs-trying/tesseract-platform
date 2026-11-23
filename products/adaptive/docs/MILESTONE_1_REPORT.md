# Milestone 1 Report - Foundation

Date: 2026-10-13
Host: Ubuntu (WSL2) x86_64, CPU-only PyTorch 2.14.1+cpu, Python 3.12.14.
Primary target remains Apple Silicon M5; this report was produced on the
development host with no CUDA.

## Files created

Source (`src/adaptive/`):
- `__init__.py`, `errors.py`, `logging.py`, `schemas.py`
- `models/`: `enums.py`, `architecture.py`, `quantization.py`, `storage.py`,
  `hardware.py`, `integrity.py`, `__init__.py`
- `analyzer/`: `base.py`, `analyzer.py`, `__init__.py`,
  `adapters/{dense,llama,qwen,mistral,__init__}.py`
- `quantize/`: `reference.py`, `packing.py`, `profiles.py`, `storage.py`,
  `__init__.py`
- `config/`: `settings.py`, `__init__.py`
- Milestone scaffolds (interfaces only): `sensitivity/`, `quality/`,
  `optimizer/`, `controller/`, `telemetry/`, `integration/`
- `cli/main.py`, `cli/__init__.py`

Schemas: `schemas/tesseract/{model-architecture,precision-profile,storage-breakdown}-v1.json`

Fixtures: `fixtures/models/{llama-3.2-1b,qwen2.5-0.5b,qwen3-0.6b}.json`

Tests: `tests/conftest.py`, `tests/unit/{test_analyzer,test_quantize,test_storage,test_profiles,test_integrity,test_cli}.py`,
`tests/property/{test_quantizer_properties,test_serialization}.py`,
`tests/integration/test_pipeline.py`, `tests/contract/test_schema_contract.py`

Docs / tooling: `README.md`, `LICENSE`, `.gitignore`, `docs/ARCHITECTURE.md`,
`docs/GPU_VALIDATION_PLAN.md`, `.github/workflows/ci.yml`,
`scripts/run_mac_tests.sh`, `examples/demo_analyze.py`, `pyproject.toml`.

No existing repository was modified or overwritten; this is a new repository.

## Features implemented

- **Model Analyzer**: adapter registry + dense-transformer adapter with
  family hooks; supports `llama`, `qwen2`, `qwen3`, `mistral`. Normalizes HF
  `config.json` into a frozen, typed `ModelArchitecture`. No weight download.
- **Parameter accounting**: per-layer, per-family, total and quantizable
  parameter counts; exact for the Llama-3.2-1B fixture (1,235,814,400).
- **Baseline weight storage**: byte-accurate at the model baseline dtype.
- **Reference quantizer**: deterministic PyTorch affine quantization -
  symmetric/asymmetric, per-tensor/per-channel/per-group, 4/8-bit, int4
  packing, reconstruction-error statistics. Runs on CPU/MPS.
- **Precision profiles**: baseline, uniform, and mixed-precision generation
  with runtime capability validation and explicit non-production labelling for
  the reference backend.
- **Serialized storage accounting**: packed payload, scales, zero points,
  per-tensor metadata, retained unquantized parameters, compression ratio.
- **Versioned JSON schemas** + atomic document writes + SHA-256 artifact
  manifests.
- **CLI**: `adaptive analyze`, `adaptive storage`, `adaptive version`.

## Tests

Command: `pytest tests/ -q` (plus `ruff check src tests`, `mypy src`).

- Passed: **73**
- Failed: **0**
- Skipped: **0**
- Ruff: clean. Mypy (strict): clean, 34 source files.
- Coverage: ~84% overall (CLI/config/logging partially covered).

Coverage of M1 acceptance areas: architecture detection, parameter accounting,
weight storage accounting, quant/dequant correctness, mixed-precision
generation, calibration/eval dataset isolation (structural, enforced by
separate suites in M2), unsupported-config rejection, model revision/artifact
integrity, plus property-based tests for optimizer-constraint-adjacent storage
monotonicity and serialization correctness.

Note: optimizer, controller, Fusion/Ignite/Foundry contract, switching and
fault-injection tests belong to M2-M4 and are not implemented yet.

## Real models tested

Small real HF configs used as fixtures (architecture only, no weights):
- `meta-llama/Llama-3.2-1B` (llama)
- `Qwen/Qwen2.5-0.5B` (qwen2)
- `Qwen/Qwen3-0.6B` (qwen3)

## Actual quality metrics

None yet. Quality evaluation is Milestone 2; no perplexity/accuracy numbers are
claimed. Only reconstruction error of the reference quantizer was measured on
synthetic tensors (relative error < 5% at int8; int4 bounded by half a
quantization step).

## Actual serialized size (Llama-3.2-1B fixture, structural estimate)

| profile | serialized | ratio |
| --- | ---: | ---: |
| baseline bf16 | 2471.6 MB | 1.00x |
| uniform 8-bit per-group (g=128) | 1274.5 MB | 1.94x |
| uniform 4-bit per-group (g=128) | 656.6 MB | 3.76x |

These are arithmetic projections from architecture structure
(`EvidenceType=estimated`), not measured on hardware.

## Memory consumption

Not measured. No runtime was executed; memory is a Milestone 3/5 concern.

## Latency and throughput

Not measured. No GPU or production kernel was run. Simulated telemetry is
Milestone 3.

## Limitations

- Reference quantizer dequantizes before matmul; it is not a production kernel
  and is marked non-deployable for integer profiles.
- GPU backend capabilities are declared but unvalidated (see
  `docs/GPU_VALIDATION_PLAN.md`).
- Adapters cover dense decoder-only families only (no MoE, no multimodal).
- Storage metadata uses a fixed per-tensor overhead assumption.
- Units: storage sizes are decimal MB (10^6 bytes).

## Bugs found and fixed during M1

1. `compute_storage` double-counted baseline bytes (init + per-layer sum).
2. Asymmetric 8-bit codes overflowed the int8 container (fixed with uint8).
3. `uniform_profile`/`mixed_profile` passed `group_size` for non-group
   granularity, raising a spurious validation error.
4. `mixed_profile` skipped validation for overrides equal to the default.
5. `QuantizedTensor.original_dtype` stored a torch dtype instead of the domain
   `DType`, breaking dequantization.
6. Unit mismatch between decimal MB and MiB in reporting; standardized on MB.

## Remaining GPU-only validation

All of `docs/GPU_VALIDATION_PLAN.md`: production kernels (GPTQ/AWQ/bnb/FP8),
real TTFT/ITL/throughput, VRAM, multi-GPU placement, and quality evaluation on
GPU runtimes.

## Next recommended engineering action

Begin **Milestone 2**: implement the sensitivity analyzer (activation-weighted
reconstruction error first) and the constrained precision optimizer (greedy +
beam search), then the quality evaluation harness with separate calibration and
evaluation datasets. Do not advance until M2 acceptance tests pass.
