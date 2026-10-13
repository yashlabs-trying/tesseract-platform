import pytest

from integrations.base import IntegrationError
from integrations.fusion import map_recommendation
from integrations.ignite import build_model_deployment, build_replicagroup
from tesseract_platform.testing import make_request


def _recommendation(eligible=True):
    return {
        "recommendation_id": "rec-1",
        "execution_eligibility": eligible,
        "approval_status": "approved",
        "recommended_plan": [
            {
                "deployment_id": "d1",
                "gpu_pool_id": "pool-a100",
                "gpu_type": "A100-80GB",
                "replica_count": 2,
                "expected_memory_mb": 40960,
                "estimated_cost_per_hour": "3.50",
                "warm_policy": "warm",
                "expected_performance": {"evidence_type": "measured"},
            }
        ],
    }


def test_map_recommendation_builds_placement():
    result = map_recommendation(_recommendation())
    assert result.reservation_id == "rec-1"
    assert result.capacity.gpu_count == 2
    assert result.capacity.cost_per_hour == pytest.approx(3.5)
    assert len(result.placement) == 2
    assert result.placement[0].gpu_model == "A100-80GB"
    assert result.placement[0].evidence.value == "measured"


def test_map_recommendation_rejects_ineligible():
    with pytest.raises(IntegrationError):
        map_recommendation(_recommendation(eligible=False))


def test_map_recommendation_rejects_empty_plan():
    with pytest.raises(IntegrationError):
        map_recommendation({"recommendation_id": "rec", "recommended_plan": []})


def test_build_replicagroup_manifest():
    request = make_request(replicas=2, min_gpu_count=1)
    manifest = build_replicagroup(request, "wl-1", namespace="tesseract")
    assert manifest["apiVersion"] == "kubbernetd.io/v1"
    assert manifest["kind"] == "ReplicaGroup"
    assert manifest["metadata"]["namespace"] == "tesseract"
    assert manifest["spec"]["model"]["workers"] == 2
    assert manifest["spec"]["targetRef"]["name"] == "wl-1"


def test_build_model_deployment_manifest():
    request = make_request(replicas=3)
    manifest = build_model_deployment(request, "wl-1")
    assert manifest["kind"] == "Deployment"
    assert manifest["spec"]["replicas"] == 3
    assert manifest["spec"]["template"]["spec"]["containers"][0]["readinessProbe"]
