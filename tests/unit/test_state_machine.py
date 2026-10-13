from tesseract_platform.coordinator.state_machine import (
    InvalidTransition,
    assert_transition,
    can_transition,
    is_terminal,
)
from tesseract_platform.models import OperationState, ReservationState


def test_valid_reservation_path():
    path = [
        ReservationState.PENDING,
        ReservationState.RESERVING,
        ReservationState.RESERVED,
        ReservationState.DEPLOYING,
        ReservationState.READY,
        ReservationState.ACTIVE,
        ReservationState.RELEASING,
        ReservationState.RELEASED,
    ]
    for a, b in zip(path, path[1:], strict=False):
        assert can_transition(ReservationState, a, b), f"{a} -> {b} should be allowed"


def test_illegal_reservation_transition():
    assert not can_transition(ReservationState, ReservationState.PENDING, ReservationState.ACTIVE)
    try:
        assert_transition(ReservationState, ReservationState.PENDING, ReservationState.ACTIVE)
    except InvalidTransition:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected InvalidTransition")


def test_failed_reservation_can_still_release():
    assert can_transition(ReservationState, ReservationState.FAILED, ReservationState.RELEASING)
    assert can_transition(ReservationState, ReservationState.FAILED, ReservationState.RELEASED)


def test_released_is_terminal():
    assert is_terminal(ReservationState, ReservationState.RELEASED)
    assert not can_transition(ReservationState, ReservationState.RELEASED, ReservationState.ACTIVE)


def test_idempotent_same_state_allowed():
    assert can_transition(OperationState, OperationState.ACTIVE, OperationState.ACTIVE)


def test_operation_terminal():
    assert is_terminal(OperationState, OperationState.RELEASED)
