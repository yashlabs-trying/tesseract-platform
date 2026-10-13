import pytest
from fastapi.testclient import TestClient

from integrations.base import IntegrationUnavailable
from integrations.foundry import FoundryAdapter
from integrations.ignite import build_model_deployment, build_replicagroup
from integrations.ignite.adapter import dns_name
from tesseract_platform.api.app import create_app
from tesseract_platform.config import Settings
from tesseract_platform.runtime import build_platform
from tesseract_platform.testing import (
    FakeDeploymentProvider,
    FakePlacementProvider,
    FakeProfileProvider,
    make_request,
)


def _platform(**settings_kwargs):
    settings = Settings(state_dsn="sqlite:///:memory:", **settings_kwargs)
    return build_platform(
        settings,
        profile_provider=FakeProfileProvider(),
        placement_provider=FakePlacementProvider(),
        deployment_provider=FakeDeploymentProvider(),
    )


def test_api_key_required_when_enabled():
    platform = _platform(api_key_enabled=True, api_key="s3cret")
    client = TestClient(create_app(platform))
    body = {
        "workload_id": "w", "idempotency_key": "idem-key-0001",
        "model": {"model_id": "m", "revision": "r"},
    }
    assert client.post("/v1/workloads", json=body).status_code == 401
    assert client.post("/v1/workloads", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.post("/v1/workloads", json=body, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
    platform.close()


def test_open_when_auth_disabled():
    platform = _platform(api_key_enabled=False)
    client = TestClient(create_app(platform))
    body = {
        "workload_id": "w", "idempotency_key": "idem-key-0002",
        "model": {"model_id": "m", "revision": "r"},
    }
    assert client.post("/v1/workloads", json=body).status_code == 200
    platform.close()


@pytest.mark.parametrize(
    "malicious",
    ["../../etc/passwd", "a/../b", "x; rm -rf /", "UPPER_CASE", "a" * 300, "$$$(whoami)"],
)
def test_dns_name_sanitises_resource_names(malicious):
    sanitized = dns_name(malicious)
    assert "/" not in sanitized and ".." not in sanitized and " " not in sanitized
    assert sanitized == sanitized.lower()
    assert len(sanitized) <= 52


def test_manifests_sanitise_malicious_model_id():
    request = make_request(workload_id="../../etc/passwd", model_id="x; rm -rf /")
    rg = build_replicagroup(request, "wl-../../etc")
    dep = build_model_deployment(request, "wl-../../etc")
    assert ".." not in rg["metadata"]["name"] and "/" not in rg["metadata"]["name"]
    assert ".." not in dep["metadata"]["name"] and "/" not in dep["metadata"]["name"]


def test_invalid_telemetry_schema_rejected_via_api():
    platform = _platform()
    client = TestClient(create_app(platform))
    resp = client.post("/v1/telemetry", json={"schema_version": "wrong"})
    assert resp.status_code == 422
    platform.close()


def test_unsafe_url_scheme_rejected():
    adapter = FoundryAdapter(base_url="file:///etc")
    with pytest.raises(IntegrationUnavailable):
        adapter.fetch_profile_document("p1")


def test_malformed_workload_rejected():
    platform = _platform()
    client = TestClient(create_app(platform))
    # missing required model -> 422, not a 500.
    resp = client.post("/v1/workloads", json={"workload_id": "w", "idempotency_key": "idem-key-0003"})
    assert resp.status_code == 422
    platform.close()
