"""Reservation / deployment / operation state machine.

The lifecycle is modelled explicitly because a single database transaction
cannot atomically cover both Fusion capacity and Kubernetes objects. Every
transition is validated and recorded so the reconciler can detect drift and
drive compensating actions.

Reservation lifecycle (required by the platform contract)::

    PENDING -> RESERVING -> RESERVED -> DEPLOYING -> READY -> ACTIVE
                                   \\-> RELEASING -> RELEASED
    any (except RELEASED) -> FAILED ; FAILED -> RELEASING -> RELEASED
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Mapping, TypeVar

from tesseract_platform.models import (
    DeploymentState,
    OperationState,
    ReservationState,
)

S = TypeVar("S")


class InvalidTransition(ValueError):
    """Raised when a state transition is not permitted by the state machine."""


RESERVATION_TRANSITIONS: Mapping[ReservationState, FrozenSet[ReservationState]] = {
    ReservationState.PENDING: frozenset({ReservationState.RESERVING, ReservationState.FAILED}),
    ReservationState.RESERVING: frozenset({ReservationState.RESERVED, ReservationState.FAILED}),
    ReservationState.RESERVED: frozenset(
        {ReservationState.DEPLOYING, ReservationState.RELEASING, ReservationState.FAILED}
    ),
    ReservationState.DEPLOYING: frozenset({ReservationState.READY, ReservationState.FAILED}),
    ReservationState.READY: frozenset(
        {ReservationState.ACTIVE, ReservationState.RELEASING, ReservationState.FAILED}
    ),
    ReservationState.ACTIVE: frozenset({ReservationState.RELEASING, ReservationState.FAILED}),
    ReservationState.RELEASING: frozenset({ReservationState.RELEASED, ReservationState.FAILED}),
    ReservationState.RELEASED: frozenset(),
    # A failed reservation must still be releasable so capacity is not leaked.
    ReservationState.FAILED: frozenset({ReservationState.RELEASING, ReservationState.RELEASED}),
}

DEPLOYMENT_TRANSITIONS: Mapping[DeploymentState, FrozenSet[DeploymentState]] = {
    DeploymentState.PENDING: frozenset({DeploymentState.DEPLOYING, DeploymentState.FAILED}),
    DeploymentState.DEPLOYING: frozenset({DeploymentState.READY, DeploymentState.FAILED}),
    DeploymentState.READY: frozenset(
        {DeploymentState.ACTIVE, DeploymentState.RELEASING, DeploymentState.FAILED}
    ),
    DeploymentState.ACTIVE: frozenset({DeploymentState.RELEASING, DeploymentState.FAILED}),
    DeploymentState.RELEASING: frozenset({DeploymentState.RELEASED, DeploymentState.FAILED}),
    DeploymentState.RELEASED: frozenset(),
    DeploymentState.FAILED: frozenset({DeploymentState.RELEASING, DeploymentState.RELEASED}),
}

OPERATION_TRANSITIONS: Mapping[OperationState, FrozenSet[OperationState]] = {
    OperationState.PENDING: frozenset(
        {OperationState.PROFILING, OperationState.RELEASING, OperationState.FAILED}
    ),
    OperationState.PROFILING: frozenset(
        {OperationState.RESERVED, OperationState.RELEASING, OperationState.FAILED}
    ),
    OperationState.RESERVED: frozenset(
        {OperationState.DEPLOYING, OperationState.RELEASING, OperationState.FAILED}
    ),
    OperationState.DEPLOYING: frozenset(
        {OperationState.ACTIVE, OperationState.RELEASING, OperationState.FAILED}
    ),
    OperationState.ACTIVE: frozenset({OperationState.RELEASING, OperationState.FAILED}),
    OperationState.RELEASING: frozenset({OperationState.RELEASED, OperationState.FAILED}),
    OperationState.RELEASED: frozenset(),
    OperationState.FAILED: frozenset({OperationState.RELEASING, OperationState.RELEASED}),
}

_TERMINAL: Dict[type, FrozenSet[str]] = {
    ReservationState: frozenset({ReservationState.RELEASED.value}),
    DeploymentState: frozenset({DeploymentState.RELEASED.value}),
    OperationState: frozenset({OperationState.RELEASED.value}),
}

_TRANSITIONS: Dict[type, Mapping] = {
    ReservationState: RESERVATION_TRANSITIONS,
    DeploymentState: DEPLOYMENT_TRANSITIONS,
    OperationState: OPERATION_TRANSITIONS,
}


def _coerce(state, enum_cls):
    if isinstance(state, enum_cls):
        return state
    return enum_cls(state)


def can_transition(enum_cls, from_state, to_state) -> bool:
    """Return True if ``from_state -> to_state`` is permitted."""
    frm = _coerce(from_state, enum_cls)
    to = _coerce(to_state, enum_cls)
    if frm == to:
        # Idempotent no-op transitions are allowed (safe to re-apply).
        return True
    return to in _TRANSITIONS[enum_cls][frm]


def assert_transition(enum_cls, from_state, to_state) -> None:
    if not can_transition(enum_cls, from_state, to_state):
        raise InvalidTransition(
            f"illegal {enum_cls.__name__} transition: "
            f"{_coerce(from_state, enum_cls).value} -> {_coerce(to_state, enum_cls).value}"
        )


def is_terminal(enum_cls, state) -> bool:
    return _coerce(state, enum_cls).value in _TERMINAL[enum_cls]


__all__ = [
    "DEPLOYMENT_TRANSITIONS",
    "OPERATION_TRANSITIONS",
    "RESERVATION_TRANSITIONS",
    "InvalidTransition",
    "assert_transition",
    "can_transition",
    "is_terminal",
]
