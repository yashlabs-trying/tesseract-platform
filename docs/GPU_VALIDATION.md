# GPU Validation Plan

**Status: PENDING — ready to test on GPUs, not GPU-validated.**

No NVIDIA GPU hardware validation has been performed. This document describes
what will be run later. GPU-specific CI is optional and clearly labelled; it is
never required for ordinary CI.

## Hardware requirements

* One or more NVIDIA GPUs (A100/H100/L4 or equivalent) with the NVIDIA device
  plugin installed.
* Node labels the platform/Fusion adapters understand:
  `nvidia.com/gpu.model`, `nvidia.com/gpu.memory`, `gpu-type`.
* CUDA-capable model server image (vLLM/SGLang) matching the GPU.

## Reproducible scenarios

1. Single-GPU single-model baseline (0.5B–13B).
2. Multi-GPU tensor-parallel (e.g. 70B across 4×A100-80GB).
3. Multi-model packing / contention.
4. Cold-start and wake latency under scale-to-zero.
5. Memory-aware placement at the capacity boundary (must never over-allocate).

## Procedure

```bash
# On a GPU node:
scripts/deploy_stack.sh --gpu
# Run the GPU-marked tests (skipped by default):
pytest -m gpu
# Benchmark harness with hardware provenance:
python benchmarks/runner.py --scenario benchmarks/scenarios/gpu-single.json
```

## What each benchmark records

Exact Git commits, environment, model/fixture, runtime, hardware, precision,
workload, concurrency, request count, measurement duration, failure count,
P50/P95/P99 latency, TTFT/ITL where meaningful, throughput, resource
utilisation, and known limitations.

## Cleanup

```bash
scripts/cleanup.sh --gpu
```

## Explicit non-claims

* "GPU readiness" means the scripts, manifests, scenarios and cleanup are
  prepared — **not** that GPUs have been validated.
* Results from simulated GPU capacity are labelled as simulated and must not be
  presented as hardware measurements.
* Multi-node and production-readiness claims require real multi-node GPU runs.
