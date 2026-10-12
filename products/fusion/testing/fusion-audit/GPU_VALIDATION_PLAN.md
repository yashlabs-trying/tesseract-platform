# GPU VALIDATION PLAN

Tesseract Fusion was audited on an Apple Silicon Mac M5 with **no NVIDIA GPU**.
This document specifies exactly what remains unvalidated and how to validate it
on real hardware. **No real-GPU certification is awarded by a Mac-based audit.**

## What was NOT validated (explicit)

| Claim area | Status on Mac | Why not validated |
|------------|---------------|-------------------|
| Real GPU memory accounting | NOT TESTED | No `nvidia.com/gpu` resources in kind |
| Real GPU scheduling (KAI / kube-scheduler GPU plugin) | NOT TESTED | No NVIDIA device plugin |
| Fractional GPU / MIG / time-slicing isolation | NOT TESTED | Requires MIG-capable hardware |
| CUDA / inference-engine behaviour | NOT TESTED | Apple Silicon, no CUDA |
| DCGM GPU utilisation metrics | NOT TESTED | No DCGM exporter, no GPU |
| Real TTFT / ITL latency percentiles | NOT TESTED | Simulator has no latency model |
| Cold-start timing | NOT TESTED | No model server |
| Power / thermal / NVLink bandwidth | NOT TESTED | Not modelled |
| Dynamo Planner ingestion | NOT TESTED | No live Dynamo |
| OpenCost real allocations | NOT TESTED | No live OpenCost |
| Prometheus DCGM scrape | NOT TESTED | No live Prometheus+DCGM |

## Minimum hardware for validation

- 2 × NVIDIA A100 (80 GB) or 2 × H100, on a single node with NVLink
- Ubuntu 22.04+, NVIDIA driver 550+, CUDA 12.x
- Kubernetes 1.29+ with the NVIDIA device plugin
- DCGM Exporter, Prometheus, Grafana, OpenCost
- vLLM or an equivalent serving engine

## Validation procedure

### Step 1 — Cluster inventory
```bash
# Verify GPUs are advertised as extended resources
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu
# Point Fusion at the cluster (read-only RBAC)
export FUSION_KUBERNETES_IN_CLUSTER=true   # or FUSION_KUBERNETES_CONFIG_FILE
python -c "from fusion.adapters.kubernetes import KubernetesInventoryAdapter as K; \
print([g.gpu_model for g in K().get_inventory()])"
```
**Expected:** Fusion reports the real GPU model, memory, and health.

### Step 2 — Real telemetry
```bash
export FUSION_PROMETHEUS_URL=http://prometheus:9090
export FUSION_OPENCOST_URL=http://opencost:9001
# DCGM metrics
curl -s "$FUSION_PROMETHEUS_URL/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL"
```
**Expected:** Fusion ingests real utilisation and real allocation cost.

### Step 3 — Measured performance profiles
Import a `tesseract.performance-profile.v1` produced from real vLLM runs:
```bash
fusion profile import profile-h100-llama70b.json
```
**Expected:** `evidence_type="measured"`, real TTFT/ITL percentiles.

### Step 4 — Validate the recommendation against reality
```bash
fusion optimize fixtures/real-fleet.yaml --mode advisory
```
Confirm, by hand, that every recommended placement:
- fits real VRAM (check `nvidia-smi`),
- respects real MIG/isolation mode,
- does not double-book a GPU,
- predicts latency consistent with measured TTFT/ITL.

### Step 5 — Fractional GPU enforcement (separate, hardware-specific)
Do **not** assume fractional requests imply memory isolation. Validate MIG
profiles or time-slicing explicitly, with an adversarial tenant test.

### Step 6 — Cloud E2E regression
Run the pytest GPU-marked suite:
```bash
pytest tests/gpu/ -m gpu --fusion-cloud-endpoint=...
```
This suite is isolated behind the `gpu` marker and is not part of the Mac run.

## Acceptance criteria for real-GPU validation

1. Fusion reads real GPU inventory and cost without fabricated values.
2. Recommendations never exceed real VRAM or real GPU counts.
3. Predicted latency is within an agreed error bound of measured TTFT/ITL.
4. Isolation requirements are enforced and adversarially tested.
5. At least one before/after replay on real traffic demonstrates a measured,
   auditable cost or margin change (not simulated).

Until all five are met, every savings figure remains **simulated**.
