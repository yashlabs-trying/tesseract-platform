# Testing Tesseract Adaptive

This guide covers running the full suite on a Mac/CPU host and the GPU-gated
validation on NVIDIA hardware. No test fabricates GPU results.

## Quick start

```bash
scripts/run_all_tests.sh
```

This creates `.venv` (Python 3.11/3.12), installs CPU PyTorch and dev deps, then
runs Ruff, strict Mypy, the full Pytest suite with coverage, and optional
`bandit` / `pip-audit`.

## Manual commands

```bash
ruff check src tests
mypy src
pytest tests/ -q --cov=adaptive --cov-report=term
```

GPU-marked tests are skipped automatically when no CUDA device is present.

## Test layout

| Path | Scope |
| --- | --- |
| `tests/unit/` | analyzer, quantizer, storage, profiles, sensitivity, quality, optimizer, controller, backends, benchmark, integration, CLI |
| `tests/property/` | property-based tests (quantizer, serialization, optimizer, controller) |
| `tests/integration/` | M1 pipeline, M2 pipeline, end-to-end Mac pipeline, M5 plan |
| `tests/contract/` | versioned JSON-schema contracts (M1-M5) |
| `tests/gpu/` | GPU-only execution validation (requires CUDA) |
| `tests/validation/` | rigorous correctness/adversarial suites (audit) |

## GPU validation

```bash
# On an NVIDIA host with the production backends installed:
pip install -e ".[torch,dev,gpu]"
scripts/run_gpu_tests.sh
scripts/benchmark_gpu.sh gpu/manifests/qwen2.5-0.5b-int4-gptq.yaml
adaptive benchmark measured.json --out reports/<model>-<profile>
```

## Honesty rules enforced by tests

- Benchmark reports reject synthetic/estimated measurements by default.
- Backend `prepare()` raises when a backend cannot execute here.
- `DeploymentPlan.executable_here` is `false` when requirements are unmet.
- No adapter silently falls back to full precision while reporting quantized
  performance.
