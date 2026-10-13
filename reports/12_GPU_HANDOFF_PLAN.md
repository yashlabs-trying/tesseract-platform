# 12 — GPU Handoff Plan (NVIDIA)

**Not executed. Prepared only.** No cloud GPU credits were spent. See
`reports/GPU_EXECUTION_CHECKLIST.md` for the runbook.

## Hardware

| Tier | GPU | Use |
|---|---|---|
| Affordable single-GPU | RTX 3090 (24 GB) | small/medium dense models, cold-start, quantized vs BF16 |
| Primary | A100 40/80 GB | 7B–13B, multi-model packing |
| Large | H100 | 70B, multi-GPU tensor parallel (only if justified) |
| Cluster | Multi-GPU Kubernetes | multi-node, packing, SLO |

## Workloads

Small dense (0.5B–1B), medium dense (3B–8B), multiple concurrent models,
long-context, burst traffic, cold-start/scale-to-zero, quantized vs BF16,
memory-pressure.

## Fair baselines (prioritized)

1. vLLM (default and tuned).
2. SGLang.
3. Kubernetes conventional Deployment.
4. KServe / Ray Serve (if practical).
5. Precision variants: BF16/FP16, GPTQ, AWQ, bitsandbytes int8/nf4.
6. Tesseract stack (Foundry+Fusion+Ignite+Platform).

Do not require every baseline in the first session; prioritize comparable
experiments within budget.

## Metrics per experiment

TTFT, ITL, P95/P99 latency, request throughput, output-token throughput,
cold-start duration, actual GPU memory, GPU utilization, packing efficiency,
precision quality (perplexity/task), quantized model load time, cost per million
successful output tokens, cost per SLO-compliant request.

## Artifacts to prepare

* GPU manifests: `deployments/kubernetes/gpu-device-plugin.yaml` + model-server
  Deployments from `integrations/ignite/adapter.py::build_model_deployment`.
* Benchmark commands: `benchmarks/runner.py --scenario benchmarks/scenarios/gpu-*.json`.
* Baseline scripts: one per baseline engine (to be authored on the GPU host).
* Result schema: reuse `reports/raw/performance_baseline.json` shape, adding
  `hardware`, `precision`, `real|emulated|simulated|estimated`.
* Cleanup: `scripts/cleanup.sh --cluster`.
* Cost tracking: `cost_per_million_output_tokens`, `cost_per_SLO_request`.
* Failure recovery: reuse Phase G matrix at the GPU layer.

## Provenance requirement

Every GPU result must record exact commits, driver/CUDA version, GPU model and
count, engine + version, precision, workload, concurrency, request count,
duration, and failure count. Estimated values must never be labelled measured.

## Explicit non-claims

GPU readiness means scripts/manifests/scenarios/cleanup are prepared — **not**
that GPUs are validated. No performance advantage is claimed.
