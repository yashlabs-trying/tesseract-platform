from datetime import timedelta

from tesseract_platform.coordinator import Reconciler
from tesseract_platform.models import (
    DeploymentRecord,
    DeploymentState,
    ModelDeploymentSpec,
    Operation,
    OperationState,
    Reservation,
    ReservationState,
    utcnow,
)
from tesseract_platform.testing import make_request


def test_expired_reservation_is_released(store, coordinator, placement_provider):
    request = make_request()
    op = Operation(
        operation_id="op-exp",
        idempotency_key=request.idempotency_key,
        workload_id=request.workload_id,
        correlation_id="corr-exp",
        request=request,
        state=OperationState.RESERVED,
        reservation_id="res-exp",
    )
    store.create_operation(op)
    store.save_reservation(
        Reservation(
            reservation_id="res-exp",
            operation_id="op-exp",
            idempotency_key=request.idempotency_key,
            workload_id=request.workload_id,
            state=ReservationState.RESERVED,
            revision=1,
            expires_at=utcnow() - timedelta(seconds=1),
        )
    )
    reconciler = Reconciler(coordinator, store, readiness_timeout_seconds=30)
    actions = reconciler.reconcile()
    assert any(a["type"] == "expire_reservation" for a in actions)
    assert placement_provider.released == ["res-exp"]
    assert store.get_operation("op-exp").state is OperationState.RELEASED


def test_stuck_deployment_is_rolled_back(store, coordinator, placement_provider, deployment_provider):
    request = make_request()
    op = Operation(
        operation_id="op-stuck",
        idempotency_key=request.idempotency_key,
        workload_id=request.workload_id,
        correlation_id="corr-stuck",
        request=request,
        state=OperationState.DEPLOYING,
        reservation_id="res-stuck",
        deployment_id="wl-stuck",
    )
    store.create_operation(op)
    store.save_reservation(
        Reservation(
            reservation_id="res-stuck",
            operation_id="op-stuck",
            idempotency_key=request.idempotency_key,
            workload_id=request.workload_id,
            state=ReservationState.DEPLOYING,
            revision=2,
        )
    )
    deployment = DeploymentRecord(
        deployment_id="wl-stuck",
        operation_id="op-stuck",
        reservation_id="res-stuck",
        workload_id=request.workload_id,
        state=DeploymentState.DEPLOYING,
        model=ModelDeploymentSpec(name=request.model.model_id),
    )
    deployment.updated_at = utcnow() - timedelta(seconds=1000)
    store.save_deployment(deployment, touch=False)

    reconciler = Reconciler(coordinator, store, readiness_timeout_seconds=30)
    actions = reconciler.reconcile()
    assert any(a["type"] == "stuck_deployment" for a in actions)
    assert deployment_provider.released == ["wl-stuck"]
    assert placement_provider.released == ["res-stuck"]
