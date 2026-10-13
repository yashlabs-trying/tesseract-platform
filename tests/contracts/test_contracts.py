import pytest

from integrations.base import IntegrationError
from integrations.foundry import map_performance_profile
from tesseract_platform.contracts import load_schema, validate
from tesseract_platform.models import (
    Capacity,
    DeploymentRecord,
    Evidence,
    LifecycleTransition,
    ModelDeploymentSpec,
    PlacementSlot,
    PrecisionProfileRef,
    Reservation,
    RollbackPlan,
    RollbackStep,
    TelemetryEvent,
    WorkloadRequest,
    utcnow,
)

ALL_SCHEMAS = [
    "workload",
    "telemetry",
    "reservation",
    "deployment",
    "lifecycle",
    "precision-profile",
    "rollback",
    "foundry-performance-profile",
]


@pytest.mark.parametrize("name", ALL_SCHEMAS)
def test_schema_loads(name):
    schema = load_schema(name)
    assert "$id" in schema and "properties" in schema


def test_workload_request_matches_contract():
    request = WorkloadRequest.model_validate(
        {
            "schema_version": "tesseract.platform.workload.v1",
            "workload_id": "wl-1",
            "idempotency_key": "idem-key-0001",
            "model": {"model_id": "qwen3-0.6b", "revision": "main"},
        }
    )
    assert validate("workload", request.model_dump(mode="json", exclude_none=True)) == []


def test_reservation_matches_contract():
    reservation = Reservation(
        reservation_id="res-1",
        operation_id="op-1",
        idempotency_key="idem-key-0001",
        workload_id="wl-1",
        revision=1,
        placement=[PlacementSlot(gpu_id="g0", node_id="n0", memory_mb=24576)],
        capacity=Capacity(gpu_count=1, cost_per_hour=2.5),
        expires_at=utcnow(),
    )
    assert validate("reservation", reservation.model_dump(mode="json", exclude_none=True)) == []


def test_deployment_matches_contract():
    deployment = DeploymentRecord(
        deployment_id="wl-1",
        operation_id="op-1",
        workload_id="wl-1",
        model=ModelDeploymentSpec(name="qwen3-0.6b"),
    )
    assert validate("deployment", deployment.model_dump(mode="json", exclude_none=True)) == []


def test_telemetry_matches_contract():
    event = TelemetryEvent(
        event_id="e1",
        correlation_id="c1",
        workload_id="wl-1",
        event_type="request",
        success=True,
        ttft_ms=100.0,
        evidence=Evidence.measured,
        source="test",
    )
    assert validate("telemetry", event.model_dump(mode="json", exclude_none=True)) == []


def test_lifecycle_and_rollback_match_contracts():
    transition = LifecycleTransition(
        entity_type="operation",
        entity_id="op-1",
        operation_id="op-1",
        from_state=None,
        to_state="PENDING",
    )
    assert validate("lifecycle", transition.model_dump(mode="json")) == []

    plan = RollbackPlan(
        rollback_id="rb-1",
        operation_id="op-1",
        target_type="deployment",
        target_id="wl-1",
        steps=[RollbackStep(component="ignite", action="release")],
    )
    assert validate("rollback", plan.model_dump(mode="json", exclude_none=True)) == []


def test_precision_profile_matches_contract():
    profile = PrecisionProfileRef(
        profile_ref="adaptive://m/awq-int4",
        model_id="qwen3-0.6b",
        runtime="vllm",
        method="awq",
        deployable=True,
        evidence=Evidence.validated,
    )
    assert validate("precision-profile", profile.model_dump(mode="json", exclude_none=True)) == []


def test_invalid_telemetry_rejected():
    errors = validate("telemetry", {"schema_version": "tesseract.platform.telemetry.v1"})
    assert errors


def test_foundry_profile_maps_and_validates():
    doc = {
        "schema_version": "tesseract.performance-profile.v1",
        "profile_id": "p1",
        "model_id": "qwen3-0.6b",
        "model_revision": "main",
        "hardware_id": "sim",
        "engine": "vLLM",
        "evidence_type": "measured",
        "collection_timestamp": "2026-01-01T00:00:00Z",
        "source_type": "benchmark",
        "ttft_p95_ms": 120.0,
        "output_tokens_per_second": 900.0,
        "completed_requests_per_second": 3.0,
    }
    assert validate("foundry-performance-profile", doc) == []
    profile = map_performance_profile(doc)
    assert profile.evidence is Evidence.measured
    assert profile.ttft_p95_ms == 120.0
    assert profile.output_tokens_per_second == 900.0


def test_foundry_profile_wrong_schema_rejected():
    with pytest.raises(IntegrationError):
        map_performance_profile({"schema_version": "wrong"})
