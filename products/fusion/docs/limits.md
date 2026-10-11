# Limitations and Cloud-GPU Requirements

## Current Limitations (v0.1)

### Not Implemented
- Real GPU hardware integration (requires cloud GPU)
- NVIDIA Dynamo Planner external plugin (pinned version verification needed)
- KAI Scheduler placement integration
- Real-time Prometheus DCGM metrics on actual GPUs
- Production authentication/authorization
- Persistent database beyond SQLite
- Kubernetes CRD controller
- Automated execution mode

### Implementation Status
- **Production-capable**: Core data models, economics engine, constraint validator, candidate generator, CLI
- **Prototype quality**: Optimizer engine, Fleet simulator, API service
- **Interface-only**: Kubernetes adapter, Prometheus adapter, OpenCost adapter (live dependencies unavailable)
- **Simulated demonstration**: All test scenarios, demo report, fixtures

### Accuracy Constraints
- All GPU costs are user-supplied or fixture-based
- Performance profiles require measured data from Foundry or manual import
- Revenue calculations require explicit billing inputs
- Forecast quality depends on sufficient historical data
- Simulator does not model NUMA topology, PCIe bandwidth contention, or power capping

## Cloud GPU Testing Requirements

To validate Fusion with real GPU hardware:

1. **Kubernetes cluster** with NVIDIA GPU nodes
2. **Minimum hardware**: 2x NVIDIA A100 or H100 GPUs
3. **Install**:
   - NVIDIA drivers and container toolkit
   - DCGM Exporter
   - Prometheus and Grafana
   - OpenCost
4. **Run inference workloads** (e.g., vLLM with Llama models)
5. **Connect Fusion** via environment variables:
   - FUSION_KUBERNETES_IN_CLUSTER=true
   - FUSION_PROMETHEUS_URL=http://prometheus:9090
   - FUSION_OPENCOST_URL=http://opencost:9001
6. **Test markers**: `pytest tests/gpu/ -m gpu`
