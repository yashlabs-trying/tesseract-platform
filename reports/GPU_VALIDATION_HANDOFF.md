# GPU Validation Handoff

See also `reports/12_GPU_HANDOFF_PLAN.md` and `reports/GPU_EXECUTION_CHECKLIST.md`.
**Not executed. No cloud GPU credits spent.**

## What is ready
* ARM64/GPU Kubernetes manifests: `deployments/kubernetes/gpu-device-plugin.yaml`
  and model-server Deployments from `integrations/ignite/adapter.py`.
* Benchmark harness with provenance + regression detector (`benchmarks/runner.py`).
* Result schema (JSON + Markdown) recording commits, environment, hardware,
  precision, workload, concurrency, latency percentiles, throughput, failures,
  and REAL/EMULATED/SIMULATED/ESTIMATED labels.
* Cleanup: `scripts/cleanup.sh --cluster`.

## Hardware tiers
RTX 3090 (affordable single-GPU), A100 (primary), H100 (if justified),
multi-GPU Kubernetes (budget permitting).

## Baselines (prioritized)
vLLM, SGLang, plain Kubernetes Deployment, KServe/Ray Serve, precision variants
(BF16/FP16, GPTQ, AWQ, bitsandbytes int8/nf4).

## Metrics
TTFT, ITL, P95/P99 latency, request throughput, output-token throughput,
cold-start duration, GPU memory/utilization, packing efficiency, precision
quality, quantized load time, cost per million successful output tokens, cost per
SLO-compliant request.

## Prerequisites before GPU work
1. Ignite `k8s-e2e` must pass (PR #1) so the deployment path is validated.
2. Mac Kubernetes lifecycle (Phase E) should pass.
3. Foundry PR #1 merged (or the fix branch pinned).

## Explicit non-claims
GPU readiness = scripts/manifests/scenarios/cleanup prepared. Not GPU-validated.
No performance advantage claimed.
