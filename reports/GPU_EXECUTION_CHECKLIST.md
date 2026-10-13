# GPU Execution Checklist (NVIDIA) — Not Yet Executed

Run on an NVIDIA GPU host. Record everything; label REAL/EMULATED/SIMULATED/
ESTIMATED. Do not claim GPU validation until this completes.

## 0. Preconditions
- [ ] GPU host with NVIDIA driver + CUDA; `nvidia-smi` works.
- [ ] Kubernetes (or bare host) with the device plugin
      (`deployments/kubernetes/gpu-device-plugin.yaml`).
- [ ] Exact commits recorded for all five repos (`platform.manifest.yaml`).
- [ ] Model artifacts cached; disk space verified.

## 1. Bring up the stack
- [ ] `scripts/run_full_stack_validation.sh --gpu --keep-cluster`
- [ ] Confirm `reports/SCORECARD.md` shows `GPU tests` attempted.

## 2. GPU device plugin & scheduling
- [ ] `kubectl get nodes -o json | jq '.items[].status.capacity'` shows
      `nvidia.com/gpu`.
- [ ] Fusion inventory adapter sees GPUs
      (`KubernetesInventoryAdapter.get_inventory`).

## 3. Workloads (one at a time, then combined)
- [ ] Small dense (0.5B–1B) — BF16 baseline.
- [ ] Medium dense (3B–8B).
- [ ] Long-context request.
- [ ] Burst traffic.
- [ ] Cold-start / scale-to-zero wake.
- [ ] Quantized (GPTQ/AWQ/int8/nf4) vs BF16.
- [ ] Memory-pressure / packing boundary.

## 4. Baselines (prioritized)
- [ ] vLLM (default + tuned).
- [ ] SGLang.
- [ ] Plain Kubernetes Deployment.
- [ ] KServe / Ray Serve (if practical).
- [ ] Precision variants.

## 5. Metrics to capture (per experiment)
- [ ] TTFT, ITL, P95/P99 latency.
- [ ] Request throughput, output-token throughput.
- [ ] Cold-start duration; quantized model load time.
- [ ] Actual GPU memory; GPU utilization; packing efficiency.
- [ ] Precision quality (perplexity/task).
- [ ] Cost per million successful output tokens; cost per SLO-compliant request.

## 6. Reliability on GPU
- [ ] Re-run the Phase G fault matrix against the GPU deployment.
- [ ] Node loss / pod eviction / image pull failure.
- [ ] Reservation invariant holds with real GPU counts.

## 7. Cleanup
- [ ] `scripts/cleanup.sh --cluster`.
- [ ] Verify no orphaned pods/volumes; stop the GPU instance (avoid idle spend).

## 8. Evidence
- [ ] JSON + Markdown per experiment, with hardware/driver/CUDA/engine/precision.
- [ ] Copy `reports/SCORECARD.md` and raw logs back to the platform repo.
