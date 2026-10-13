from tesseract_platform.models import DeploymentState, OperationState, ReservationState
from tesseract_platform.testing import make_request


def test_cold_request_full_lifecycle(coordinator, store, placement_provider, deployment_provider):
    op = coordinator.submit(make_request())
    assert op.state is OperationState.ACTIVE
    assert op.endpoint

    reservation = store.get_reservation(op.reservation_id)
    assert reservation is not None and reservation.state is ReservationState.ACTIVE
    deployment = store.get_deployment(op.deployment_id)
    assert deployment is not None and deployment.state is DeploymentState.ACTIVE

    released = coordinator.release(op.operation_id)
    assert released.state is OperationState.RELEASED
    assert placement_provider.released == [op.reservation_id]
    assert deployment_provider.released == [op.deployment_id]


def test_idempotent_submit_does_not_double_reserve(coordinator, placement_provider):
    request = make_request()
    first = coordinator.submit(request)
    second = coordinator.submit(request)
    assert first.operation_id == second.operation_id
    assert placement_provider.reserved == [first.reservation_id]


def test_operation_transitions_are_recorded_in_order(coordinator, store):
    op = coordinator.submit(make_request())
    transitions = store.list_transitions(op.operation_id)
    operation_states = [t.to_state for t in transitions if t.entity_type == "operation"]
    assert operation_states == ["PROFILING", "RESERVED", "DEPLOYING", "ACTIVE"]
    reservation_states = [t.to_state for t in transitions if t.entity_type == "reservation"]
    assert reservation_states[:3] == ["RESERVING", "RESERVED", "DEPLOYING"]


def test_multi_model_requests(coordinator):
    for index, model_id in enumerate(["qwen3-0.6b", "llama-3.1-8b", "mixtral-8x22b"]):
        op = coordinator.submit(
            make_request(
                workload_id=f"wl-{index}",
                idempotency_key=f"idem-key-{index:04d}",
                model_id=model_id,
            )
        )
        assert op.state is OperationState.ACTIVE
        assert op.request.model.model_id == model_id


def test_correlation_id_preserved_across_lifecycle(coordinator, store):
    op = coordinator.submit(make_request())
    transitions = store.list_transitions(op.operation_id)
    assert transitions
    assert all(t.correlation_id == op.correlation_id for t in transitions)
