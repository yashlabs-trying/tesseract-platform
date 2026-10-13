# GPU validation plan

Adaptive is developed and tested on Apple Silicon (M5) with CPU/MPS. GPU
results require NVIDIA hardware and are tracked separately. This plan defines
what must be measured before any GPU claim is promoted from `estimated` to
`measured` or `validated`.

## Scope

| Area | Mac M5 (current) | NVIDIA (required) |
| --- | --- | --- |
| Reference quantizer correctness | measured (CPU) | re-verified on CUDA |
| Storage accounting | measured (arithmetic) | unchanged |
| Latency / throughput | simulated only | measured |
| TTFT / ITL | simulated only | measured |
| Quality (perplexity, task acc.) | not yet | measured |
| Production kernels (GPTQ/AWQ/bnb/FP8) | not run | measured |
| Multi-GPU placement | not run | measured |

## Backends to validate

- `vllm` with GPTQ/AWQ/FP8 weights.
- `sglang` with the same.
- `tensorrt_llm` with INT4/INT8/FP8 engines.
- `bitsandbytes` int8/nf4 under `torch_eager`/`torch_compile`.

## Required measurements per (model, profile, hardware, workload)

- Loaded weight bytes and peak VRAM.
- TTFT p50/p95, ITL p50/p95, output tokens/second.
- Perplexity on the pinned calibration/eval split.
- Task accuracy / structured-output pass rate where applicable.
- Cost per million tokens using Foundry economics inputs.

## Method

1. Pin model revision, tokenizer, datasets, seeds and runtime versions.
2. Run each profile N>=3 times; report median and p95.
3. Import results into Foundry via the `tesseract.performance-profile.v1`
   contract, preserving `EvidenceType=measured`.
4. Reconcile against Adaptive's `estimated` projections; record deltas.
5. Only then mark profiles `validated`.

## Guardrails

- No synthetic or extrapolated numbers may be reported as measured.
- No claim that lower precision is faster without measurement.
- Rollback must be exercised for every profile promoted to production.
