"""Reconciliation loop.

The reconciler repairs drift between the desired and observed lifecycle:

* reservations that expired before they were committed are released;
* deployments stuck in ``DEPLOYING`` past the readiness timeout are rolled back.

Each action is a compensating action with an audited outcome; the reconciler
never leaves capacity reserved without a corresponding deployment.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from tesseract_platform.coordinator.coordinator import WorkloadCoordinator
from tesseract_platform.models import (
    DeploymentState,
    OperationState,
    ReservationState,
    utcnow,
)
from tesseract_platform.state import StateStore


class Reconciler:
    def __init__(
        self,
        coordinator: WorkloadCoordinator,
        store: StateStore,
        *,
        readiness_timeout_seconds: int = 300,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.coordinator = coordinator
        self.store = store
        self.readiness_timeout_seconds = readiness_timeout_seconds
        self.clock = clock

    def reconcile(self, now: Optional[datetime] = None) -> list[dict]:
        now = now or self.clock()
        actions: list[dict] = []

        for reservation in self.store.list_reservations_by_state(
            [ReservationState.RESERVED.value, ReservationState.RESERVING.value]
        ):
            if reservation.expires_at is not None and reservation.expires_at <= now:
                op = self.store.get_operation(reservation.operation_id)
                if op is not None and op.state not in (
                    OperationState.RELEASED,
                    OperationState.FAILED,
                ):
                    plan = self.coordinator.rollback(
                        op.operation_id, reason="reservation_expired"
                    )
                    actions.append(
                        {
                            "type": "expire_reservation",
                            "reservation_id": reservation.reservation_id,
                            "outcome": plan.state,
                        }
                    )

        for deployment in self.store.list_deployments_by_state(
            [DeploymentState.DEPLOYING.value]
        ):
            age = (now - deployment.updated_at).total_seconds()
            if age <= self.readiness_timeout_seconds:
                continue
            op = self.store.get_operation(deployment.operation_id)
            if op is not None and op.state == OperationState.DEPLOYING:
                plan = self.coordinator.rollback(
                    op.operation_id, reason="readiness_timeout"
                )
                actions.append(
                    {
                        "type": "stuck_deployment",
                        "deployment_id": deployment.deployment_id,
                        "outcome": plan.state,
                    }
                )
        return actions
