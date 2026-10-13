# Milestone 5 Report - GPU Readiness

Date: 2026-10-13
Host: Ubuntu (WSL2) x86_64, CPU-only PyTorch 2.14.1+cpu, Python 3.12.14.
No CUDA device present. Primary target remains Apple Silicon M5.

## Files created

Source (`src/adaptive/`):
- `backends/availability.py` - dependency-light CUDA/package detection.
- `backends/base.py` - `BackendAdapter`, `BackendRequirements`,
  `DeploymentPlan`, `plan_document`.
- `backends/adapters.py` - GPTQ, AWQ, bitsandbytes int8/nf4 and FP8 adapters,
  plus a backend registry.
- `backends/__init__.py`.
- `benchmark/manifest.py` - pinned `GpuTestManifest` model + loaders.
- `benchmark/report.py` - measured-only `BenchmarkReport` generation, atomic
  JSON/Markdown writers.
- `benchmark/__init__.py`.
- `optimizer/measured.py` - `MeasuredCostModel` and
  `measured_costs_from_measurements` wiring Foundry imports into the optimizer.
- `cli/main.py` - `benchmark` command.

Manifests: `gpu/manifests/{qwen2.5-0.5b-int4-gptq,llama-3.2-1b-int8-bnb,qwen3-0.6b-fp8}.yaml`

Schemas: `schemas/tesseract/{benchmark-report,deployment-plan,gpu-test-manifest}-v1.json`

Scripts: `scripts/run_gpu_tests.sh`, `scripts/benchmark_gpu.sh`.

Tests: `tests/unit/{test_backends,test_benchmark}.py`,
`tests/contract/test_m5_schema_contract.py`,
`tests/integration/test_m5_plan.py`,
`tests/gpu/test_gpu_validation.py` (GPU-marked), `tests/unit/test_cli.py`
(extended).

## Features implemented

- **Production backend adapters** behind a common interface. Each adapter
  declares requirements, detects availability without importing CUDA libraries,
  validates bit-width support, and produces a `DeploymentPlan` (artifacts,
  calibration needs, expected bytes) that works with no GPU. `prepare()`
  refuses to run when unavailable or unconfigured.
- **Backend registry** (`get_backend`, `backends_for_runtime`,
  `available_backends`).
- **Benchmark report generator** that accepts only `measured`/`validated`
  measurements and rejects synthetic/estimated inputs by default, producing
  JSON and Markdown with provenance.
- **GPU test manifests** pinning model/revision, profile, hardware, workload,
  datasets and acceptance thresholds; validated on any host.
- **GPU-gated tests** (`tests/gpu/`) skipped without CUDA.
- **Measured cost model** that prefers imported Foundry measurements and falls
  back to a clearly-labelled analytic estimate.
- **CLI** `adaptive benchmark <measured.json> --out <path>`.

## Tests

Command: `pytest tests/ -q` plus `ruff check src tests`, `mypy src`.

- Passed: **192** (was 162 at M3+M4)
- Skipped: **2** (GPU-marked, require CUDA)
- Failed: **0**
- Ruff: clean. Mypy (strict): clean, 64 source files.
- Coverage: ~92%.

M5 acceptance areas covered: backend registry and availability detection,
deployment-plan generation without a GPU, unsupported-backend rejection,
`prepare()` unavailability handling, measured-only benchmark reporting (with
rejection of fabricated data), GPU manifest validation, measured-cost wiring
with analytic fallback, versioned schemas, and an end-to-end plan test.

## Backend availability on this host

No CUDA device is present, so:

- `available_backends()` returns `[]`.
- Every generated `DeploymentPlan.executable_here` is `false`, with a note
  naming the missing requirement.
- `prepare()` raises `BackendUnavailableError`.

This is the correct, honest state on a Mac/CPU host.

## Benchmark report example (measured input)

`adaptive benchmark fixtures/foundry/performance-profile-example.json`:

| profile | engine | GPU | TTFT p95 (ms) | ITL p95 (ms) | throughput (tps) |
| --- | --- | --- | ---: | ---: | ---: |
| foundry-profile-qwen2-0.5b-int4 | vllm | NVIDIA A10G | 71.00 | 14.00 | 1850.00 |

Evidence: `measured`. Note the value originates from an imported Foundry
contract document, not from a GPU run performed here.

## Limitations

- No backend has been executed on real hardware; adapters plan and gate only.
- Benchmark reports require externally produced measurements; none were
  produced in this environment.
- `scripts/benchmark_gpu.sh` is a documented workflow template, not an
  automated runner.
- Measured cost wiring is unit-tested but not exercised against live Foundry
  data.
- Multi-GPU placement and TensorRT engine building remain untested.

## Bugs found and fixed during M5

1. Backend requirements needed an "any of" package set (GPTQ ships as
   `auto_gptq` or `gptqmodel`; AWQ as `awq` or `autoawq`); added
   `requires_any_packages` and availability logic.
2. Benchmark report initially accepted any evidence type; now rejects
   synthetic/estimated inputs by default to prevent fabricated reports.

## Remaining GPU-only validation

Everything in `docs/GPU_VALIDATION_PLAN.md`: real backend execution, measured
TTFT/ITL/throughput/VRAM, real-model quality, multi-GPU placement, and
promotion of profiles to `validated`. Run `scripts/run_gpu_tests.sh` and
`scripts/benchmark_gpu.sh` on an NVIDIA host.

## Next recommended engineering action

Run the M5 workflow on a validated NVIDIA host: install the production
backends, execute `scripts/run_gpu_tests.sh`, produce measured
`performance-profile.v1` documents per manifest, build benchmark reports, and
import the measured costs into the optimizer/controller. Only then promote
profiles from `estimated` to `measured`/`validated`.
