"""Phase B — live service integration harness.

Exercises real product code:

* Foundry: real HTTP service (uvicorn) endpoints AND Foundry's real library
  (candidate generation + fixture profiler + ``export_performance_profile``).
* Fusion:  real HTTP service (uvicorn) endpoints (inventory, workloads,
  optimization evaluation).
* Platform: real coordinator + real ``FusionAdapter`` over HTTP against the
  live Fusion service.

Ignite's Kubernetes deployment is out of scope here (no cluster); the deployment
provider builds real Ignite ``ReplicaGroup`` manifests and marks readiness at the
Kubernetes boundary. That boundary is reported explicitly.

Run with the combined validation venv. Services must already be running on
127.0.0.1:8001 (Foundry) and 127.0.0.1:8002 (Fusion).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

REPO = Path("/home/yashs/tesseract-platform")
sys.path.insert(0, str(REPO))

from integrations.fusion import FusionAdapter  # noqa: E402
from tesseract_platform.config import Settings  # noqa: E402
from tesseract_platform.models import (  # noqa: E402
    WorkloadProfile,
    WorkloadRequest,
)
from tesseract_platform.runtime import build_platform  # noqa: E402

FOUNDRY_URL = "http://127.0.0.1:8001"
FUSION_URL = "http://127.0.0.1:8002"
FOUNDRY_FIXTURE = Path("/home/yashs/tesseract-foundry/fixtures/scenario-1-single-gpu.yaml")

evidence: dict = {"phase": "B", "generated_at": datetime.now(timezone.utc).isoformat()}


def _safe(response: httpx.Response):
    try:
        return response.json()
    except Exception:
        return response.text[:500]


def http(method: str, url: str, payload=None):
    try:
        r = httpx.request(method, url, json=payload, timeout=30.0)
        return {"status": r.status_code, "body": _safe(r)}
    except Exception as exc:  # noqa: BLE001
        return {"status": None, "error": str(exc)}


def _load_scenario() -> dict:
    data = yaml.safe_load(FOUNDRY_FIXTURE.read_text())
    data.setdefault("scenario_id", "phase-b-scenario")
    return data


# --------------------------------------------------------------------------- Foundry
def foundry_live():
    out = {}
    out["healthz"] = http("GET", f"{FOUNDRY_URL}/healthz")
    out["metrics_before"] = http("GET", f"{FOUNDRY_URL}/metrics")
    scenario = _load_scenario()
    out["recommend"] = http(
        "POST", f"{FOUNDRY_URL}/v1/services/recommend", {"scenario": scenario}
    )
    body = out["recommend"].get("body")
    rec_id = body.get("recommendation_id") if isinstance(body, dict) else None
    if rec_id:
        out["report"] = http(
            "GET", f"{FOUNDRY_URL}/v1/services/recommendations/{rec_id}/report?format=json"
        )
        out["manifest"] = http(
            "GET", f"{FOUNDRY_URL}/v1/services/recommendations/{rec_id}/manifest"
        )
    # Characterise the rest of the Foundry HTTP surface.
    out["schemas_validate"] = http(
        "POST",
        f"{FOUNDRY_URL}/v1/schemas/validate?schema_name=tesseract.performance-profile.v1",
        {
            "schema_version": "tesseract.performance-profile.v1",
            "profile_id": "p1",
            "model_id": "m",
            "model_revision": "r",
            "hardware_id": "h",
            "engine": "vllm",
            "evidence_type": "synthetic",
            "collection_timestamp": "2026-01-01T00:00:00Z",
            "source_type": "test_fixture",
        },
    )
    out["workloads_validate"] = http(
        "POST",
        f"{FOUNDRY_URL}/v1/workloads/validate",
        {
            "workload_id": "w",
            "request_rate_rps": 5.0,
            "concurrency": 8,
            "input_tokens": {"mean": 512, "p50": 384, "p95": 2048, "p99": 4096, "max": 8192},
            "output_tokens": {"mean": 256, "p50": 128, "p95": 1024, "p99": 2048, "max": 4096},
        },
    )
    out["profiling_jobs"] = http(
        "POST",
        f"{FOUNDRY_URL}/v1/profiling/jobs?mode=fixture",
        {
            "model": {
                "model_id": "Qwen/Qwen3-0.6B",
                "model_revision": "main",
                "source": "huggingface",
                "parameter_count": 600000000,
                "supported_backends": ["vllm"],
            },
            "hardware": [
                {
                    "hardware_id": "l4-1",
                    "vendor": "nvidia",
                    "gpu_model": "L4",
                    "device_count": 1,
                    "vram_per_gpu_gb": 24.0,
                    "supported_precision": ["fp16", "int8"],
                    "cost_per_gpu_hour_usd": 0.8,
                }
            ],
            "workload": {
                "workload_id": "w",
                "request_rate_rps": 5.0,
                "concurrency": 8,
                "input_tokens": {"mean": 512, "p50": 384, "p95": 2048, "p99": 4096, "max": 8192},
                "output_tokens": {"mean": 256, "p50": 128, "p95": 1024, "p99": 2048, "max": 4096},
            },
        },
    )
    return out


def foundry_library_profile():
    """Use Foundry's real library to produce a performance-profile document."""
    from foundry.benchmarking import get_runner
    from foundry.candidates import get_provider
    from foundry.models import Scenario
    from foundry.schemas import export_performance_profile

    scenario = Scenario(**_load_scenario())
    provider = get_provider("fixture")
    candidates = provider.generate_candidates(scenario.model, scenario.hardware, scenario.workload)
    runner = get_runner("fixture")
    ev = runner.run(candidates[0], scenario.workload)
    hw = scenario.hardware[0]
    profile = export_performance_profile(
        candidate_id=candidates[0].candidate_id,
        model_id=scenario.model.model_id,
        model_revision=scenario.model.model_revision,
        hardware_id=hw.hardware_id,
        hardware_gpu_model=hw.gpu_model,
        hardware_gpu_count=hw.device_count,
        hardware_vram_gb=hw.vram_per_gpu_gb,
        engine=str(candidates[0].serving_engine),
        engine_version=candidates[0].engine_version,
        evidence_type=ev.evidence_type.value,
        workload_id=scenario.workload.workload_id,
        workload_concurrency=scenario.workload.concurrency,
        workload_input_token_mean=scenario.workload.input_tokens.mean,
        workload_output_token_mean=scenario.workload.output_tokens.mean,
        ttft_p50_ms=ev.ttft_ms.p50 if ev.ttft_ms else None,
        ttft_p95_ms=ev.ttft_ms.p95 if ev.ttft_ms else None,
        ttft_p99_ms=ev.ttft_ms.p99 if ev.ttft_ms else None,
        itl_p50_ms=ev.itl_ms.p50 if ev.itl_ms else None,
        itl_p95_ms=ev.itl_ms.p95 if ev.itl_ms else None,
        request_latency_p95_ms=ev.request_latency_ms.p95 if ev.request_latency_ms else None,
        completed_requests_per_second=ev.throughput.completed_requests_per_second
        if ev.throughput
        else None,
        error_rate=ev.error_rate,
        sample_count=ev.sample_count,
        source_type="test_fixture",
    )
    return {
        "candidate_id": candidates[0].candidate_id,
        "evidence_type": ev.evidence_type.value,
        "profile_document": profile,
    }


class FoundryLibraryProfileProvider:
    """Platform ProfileProvider backed by Foundry's real library (fixture mode)."""

    def __init__(self, document: dict):
        self.document = document

    def get_profile(self, request: WorkloadRequest) -> WorkloadProfile:
        from integrations.foundry import map_performance_profile

        return map_performance_profile(self.document)


# --------------------------------------------------------------------------- Fusion
INVENTORY = [
    {
        "gpu_id": "node-a-gpu-0",
        "cluster_id": "local",
        "node_id": "node-a",
        "vendor": "nvidia",
        "gpu_model": "A100",
        "memory_capacity_mb": 81920,
        "available_memory_mb": 81920,
        "available_resource_slots": 4,
        "supported_isolation_modes": ["exclusive"],
        "health_status": "healthy",
        "sharing_policy": "exclusive_only",
        "resource_cost_per_hour": 3.0,
        "resource_cost_source": "fixture",
    }
]

WORKLOADS = [
    {
        "deployment_id": "dep-qwen",
        "tenant_id": "tenant-a",
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "main",
        "serving_engine": "vllm",
        "worker_count": 1,
        "warm_workers": 1,
        "expected_memory_mb": 40960,
    }
]


def fusion_live():
    out = {}
    out["healthz"] = http("GET", f"{FUSION_URL}/healthz")
    out["inventory_import"] = http("POST", f"{FUSION_URL}/v1/inventory/import", INVENTORY)
    out["inventory"] = http("GET", f"{FUSION_URL}/v1/inventory")
    out["workloads_import"] = http("POST", f"{FUSION_URL}/v1/workloads/import", WORKLOADS)
    out["workloads"] = http("GET", f"{FUSION_URL}/v1/workloads")
    scenario = {
        "scenario_id": "phase-b",
        "candidate_allocation_policy": "economic",
        "resource_constraints": {"min_gpu_count": 1},
        "workload": {"model_id": "Qwen/Qwen3-0.6B", "replicas": 1},
    }
    out["optimize"] = http(
        "POST", f"{FUSION_URL}/v1/optimization/evaluate", {"scenario": scenario}
    )
    body = out["optimize"].get("body")
    rec_id = body.get("recommendation_id") if isinstance(body, dict) else None
    if rec_id:
        out["recommendation_get"] = http(
            "GET", f"{FUSION_URL}/v1/optimization/recommendations/{rec_id}"
        )
    return out


# --------------------------------------------------------------------------- Ignite (manifest-only)
class LocalIgniteDeploymentProvider:
    """Builds real Ignite manifests; readiness is stubbed at the K8s boundary."""

    def __init__(self, namespace="tesseract"):
        self.namespace = namespace
        self.deployed = []
        self.released = []
        self.manifests = {}

    def deploy(self, request, reservation, precision=None):
        from integrations.base import DeploymentHandle
        from integrations.ignite import build_model_deployment, build_replicagroup

        deployment_id = f"wl-{request.workload_id}"
        self.manifests[deployment_id] = {
            "replicagroup": build_replicagroup(request, deployment_id, namespace=self.namespace),
            "deployment": build_model_deployment(request, deployment_id, namespace=self.namespace),
        }
        self.deployed.append(deployment_id)
        return DeploymentHandle(
            deployment_id=deployment_id,
            endpoint=f"/v1/workloads/{request.workload_id}",
            replicas=request.placement.replicas or 1,
            namespace=self.namespace,
        )

    def await_ready(self, deployment_id, timeout_seconds):
        return True  # K8s boundary stubbed

    def release(self, deployment_id):
        self.released.append(deployment_id)


# --------------------------------------------------------------------------- lifecycle
def platform_lifecycle(profile_document: dict):
    settings = Settings(
        state_dsn="sqlite:///:memory:",
        foundry_url=FOUNDRY_URL,
        fusion_url=FUSION_URL,
    )
    ignite = LocalIgniteDeploymentProvider()
    platform = build_platform(
        settings,
        profile_provider=FoundryLibraryProfileProvider(profile_document),
        placement_provider=FusionAdapter(base_url=FUSION_URL),
        deployment_provider=ignite,
    )
    request = WorkloadRequest.model_validate(
        {
            "workload_id": "phase-b-wl",
            "idempotency_key": "phase-b-idem-0001",
            "model": {"model_id": "Qwen/Qwen3-0.6B", "revision": "main"},
            "workload": {"concurrency": 8, "arrival_rate_rps": 5.0},
            "slo": {"ttft_p95_ms": 500.0},
            "placement": {"replicas": 1, "min_gpu_count": 1},
        }
    )
    op = platform.coordinator.submit(request)
    transitions = [t.model_dump(mode="json") for t in platform.store.list_transitions(op.operation_id)]
    reservation = platform.store.get_reservation(op.reservation_id) if op.reservation_id else None
    deployment = platform.store.get_deployment(op.deployment_id) if op.deployment_id else None
    released = platform.coordinator.release(op.operation_id)
    result = {
        "final_state": op.state.value,
        "error": op.error,
        "endpoint": op.endpoint,
        "profile_evidence": op.profile.evidence.value if op.profile else None,
        "profile_source_type": op.profile.source_type if op.profile else None,
        "reservation": reservation.model_dump(mode="json") if reservation else None,
        "deployment": deployment.model_dump(mode="json") if deployment else None,
        "released_state": released.state.value,
        "ignite_deployed": ignite.deployed,
        "ignite_released": ignite.released,
        "ignite_manifests": ignite.manifests,
        "transitions": transitions,
    }
    platform.close()
    return result


def main() -> int:
    evidence["foundry_live"] = foundry_live()
    lib = foundry_library_profile()
    evidence["foundry_library"] = lib
    evidence["fusion_live"] = fusion_live()
    evidence["platform_lifecycle"] = platform_lifecycle(lib["profile_document"])

    out = REPO / "reports" / "raw" / "live_integration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, default=str))
    print(f"wrote {out}")
    print("platform final_state:", evidence["platform_lifecycle"]["final_state"])
    print("foundry recommend status:", evidence["foundry_live"]["recommend"]["status"])
    print("fusion optimize status:", evidence["fusion_live"]["optimize"]["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
