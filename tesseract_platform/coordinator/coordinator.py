"""Workload coordinator.

Owns the end-to-end lifecycle:

    submit -> profile (Foundry) -> reserve (Fusion) -> deploy (Ignite)
           -> await readiness -> ACTIVE
    failure at any step -> compensating actions -> FAILED

All state lives in the durable :class:`StateStore`; every transition is recorded.
Because Fusion capacity and Kubernetes objects cannot be updated atomically, the
coordinator uses explicit compensating actions instead of a distributed
transaction.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Optional

from tesseract_platform.coordinator.state_machine import assert_transition
from tesseract_platform.models import (
    Capacity,
    DeploymentRecord,
    DeploymentState,
    LifecycleTransition,
    ModelDeploymentSpec,
    Operation,
    OperationState,
    PrecisionProfileRef,
    Reservation,
    ReservationState,
    RollbackPlan,
    RollbackStep,
    TelemetryEvent,
    WorkloadProfile,
    WorkloadRequest,
    utcnow,
)
from tesseract_platform.policies import PlatformPolicy
from tesseract_platform.state import StateStore


class CoordinatorError(RuntimeError):
    """Base error for coordinator failures."""


class ReadinessTimeout(CoordinatorError):
    """A deployment did not become ready within the configured timeout."""


class WorkloadCoordinator:
    def __init__(
        self,
        *,
        store: StateStore,
        policy: PlatformPolicy,
        profile_provider=None,
        placement_provider=None,
        deployment_provider=None,
        precision_provider=None,
        telemetry=None,
        capacity_ledger=None,
        readiness_timeout_seconds: Optional[int] = None,
    ) -> None:
        self.store = store
        self.policy = policy
        self.profile_provider = profile_provider
        self.placement_provider = placement_provider
        self.deployment_provider = deployment_provider
        self.precision_provider = precision_provider
        self.telemetry = telemetry
        self.capacity_ledger = capacity_ledger
        self.readiness_timeout_seconds = (
            readiness_timeout_seconds or policy.readiness_timeout_seconds
        )

    # ================================================================ public API

    def submit(self, request: WorkloadRequest) -> Operation:
        """Submit a workload. Idempotent on ``idempotency_key``."""
        self.policy.validate_workload(request)
        candidate = Operation(
            operation_id=str(uuid.uuid4()),
            idempotency_key=request.idempotency_key,
            workload_id=request.workload_id,
            correlation_id=str(uuid.uuid4()),
            tenant_id=request.tenant_id,
            request=request,
        )
        op = self.store.create_operation(candidate)
        if op.operation_id != candidate.operation_id:
            # Idempotent replay: return the existing operation unchanged.
            self.store.record_audit(
                op.operation_id,
                "idempotent_replay",
                {"idempotency_key": request.idempotency_key, "state": op.state.value},
            )
            return op

        try:
            self._run(op)
        except Exception as exc:  # noqa: BLE001 - we must record every failure
            self._fail(op, exc)
        return self.store.get_operation(op.operation_id) or op

    def get(self, operation_id: str) -> Optional[Operation]:
        return self.store.get_operation(operation_id)

    def release(self, operation_id: str, reason: str = "client_release") -> Operation:
        op = self._require(operation_id)
        if op.state in (OperationState.RELEASED,):
            return op
        self._set_operation_state(op, OperationState.RELEASING, reason)
        plan = self._compensate(op, reason=reason, actor="coordinator.release")
        final = OperationState.RELEASED if plan.state == "COMPLETED" else OperationState.FAILED
        self._set_operation_state(op, final, "release complete")
        return self.store.get_operation(op.operation_id) or op

    def rollback(self, operation_id: str, reason: str = "rollback") -> RollbackPlan:
        op = self._require(operation_id)
        if op.state not in (OperationState.RELEASED,):
            self._set_operation_state(op, OperationState.RELEASING, reason)
        plan = self._compensate(op, reason=reason, actor="coordinator.rollback")
        final = OperationState.RELEASED if plan.state == "COMPLETED" else OperationState.FAILED
        if op.state != final:
            self._set_operation_state(op, final, "rollback complete")
        return plan

    # ================================================================ lifecycle

    def _run(self, op: Operation) -> None:
        self._set_operation_state(op, OperationState.PROFILING, "profiling")
        profile = self._resolve_profile(op)
        precision = self._resolve_precision(op, profile)
        op.profile = profile
        if precision is not None:
            op.precision_profile_ref = precision.profile_ref
        self.store.save_operation(op)

        reservation = self._reserve(op, profile, precision)
        deployment = self._deploy(op, reservation, precision)
        self._await_ready(op, deployment)

        self._set_operation_state(op, OperationState.ACTIVE, "deployment ready")
        self._emit(
            op,
            "cold_start",
            cold_start_ms=deployment.updated_at and self._ms_between(op.created_at, deployment.updated_at),
            readiness_ms=self._ms_between(deployment.created_at, deployment.updated_at),
            replica_count=deployment.replicas,
        )
        self.store.record_audit(
            op.operation_id,
            "workload_active",
            {"endpoint": op.endpoint, "reservation_id": op.reservation_id},
        )

    def _resolve_profile(self, op: Operation) -> WorkloadProfile:
        profile: Optional[WorkloadProfile] = None
        if self.profile_provider is not None:
            try:
                profile = self.profile_provider.get_profile(op.request)
            except Exception as exc:  # noqa: BLE001 - provider unavailable is expected
                self.store.record_audit(
                    op.operation_id, "foundry_unavailable", {"error": str(exc)}
                )
                profile = None
        return self.policy.resolve_profile(op.request, profile)

    def _resolve_precision(
        self, op: Operation, profile: WorkloadProfile
    ) -> Optional[PrecisionProfileRef]:
        if self.precision_provider is None:
            return None
        try:
            recommended = self.precision_provider.recommend_profile(
                op.request, capacity=None
            )
        except Exception as exc:  # noqa: BLE001 - optional provider
            self.store.record_audit(
                op.operation_id, "adaptive_unavailable", {"error": str(exc)}
            )
            return None
        if recommended is not None and not self.precision_provider.validate_profile(recommended):
            self.store.record_audit(
                op.operation_id, "adaptive_profile_rejected",
                {"profile_ref": recommended.profile_ref},
            )
            return None
        return recommended

    def _reserve(
        self, op: Operation, profile: WorkloadProfile, precision: Optional[PrecisionProfileRef]
    ) -> Reservation:
        if self.placement_provider is None:
            raise CoordinatorError("no placement provider configured")
        result = self.placement_provider.reserve(op.request, profile, precision)
        if self.capacity_ledger is not None:
            # Enforce committed + active_reserved <= total_usable_capacity.
            self.capacity_ledger.reserve(
                op.operation_id, result.capacity.gpu_count if result.capacity else 0
            )
        reservation = Reservation(
            reservation_id=result.reservation_id or str(uuid.uuid4()),
            operation_id=op.operation_id,
            idempotency_key=op.idempotency_key,
            workload_id=op.workload_id,
            state=ReservationState.RESERVING,
            revision=0,
            placement=result.placement,
            capacity=result.capacity or Capacity(),
            expires_at=result.expires_at
            or (utcnow() + timedelta(seconds=self.policy.reservation_ttl_seconds)),
        )
        self.store.save_reservation(reservation)
        self._record_transition("reservation", reservation.reservation_id, op, None, "RESERVING")
        # Commit the reservation.
        reservation.state = ReservationState.RESERVED
        reservation.revision = 1
        self.store.save_reservation(reservation, expected_revision=0)
        self._record_transition(
            "reservation", reservation.reservation_id, op, "RESERVING", "RESERVED"
        )
        if self.capacity_ledger is not None:
            self.capacity_ledger.commit(op.operation_id)
        op.reservation_id = reservation.reservation_id
        self._set_operation_state(op, OperationState.RESERVED, "capacity reserved")
        return reservation

    def _deploy(
        self, op: Operation, reservation: Reservation, precision: Optional[PrecisionProfileRef]
    ) -> DeploymentRecord:
        if self.deployment_provider is None:
            raise CoordinatorError("no deployment provider configured")
        handle = self.deployment_provider.deploy(op.request, reservation, precision)
        deployment = DeploymentRecord(
            deployment_id=handle.deployment_id,
            operation_id=op.operation_id,
            reservation_id=reservation.reservation_id,
            workload_id=op.workload_id,
            namespace=handle.namespace,
            state=DeploymentState.DEPLOYING,
            replicas=handle.replicas,
            model=ModelDeploymentSpec(
                name=op.request.model.model_id,
                precision=precision.method if precision else op.request.model.precision,
                gpuPerWorker=op.request.placement.min_gpu_count,
                workers=max(1, op.request.placement.replicas or 1),
            ),
            precision_profile_ref=handle.precision_profile_ref,
            endpoint=handle.endpoint,
        )
        self.store.save_deployment(deployment)
        self._record_transition("deployment", deployment.deployment_id, op, None, "DEPLOYING")
        # Move the reservation to DEPLOYING.
        reservation.state = ReservationState.DEPLOYING
        reservation.revision = 2
        self.store.save_reservation(reservation, expected_revision=1)
        self._record_transition(
            "reservation", reservation.reservation_id, op, "RESERVED", "DEPLOYING"
        )
        op.deployment_id = deployment.deployment_id
        self._set_operation_state(op, OperationState.DEPLOYING, "deploying")
        return deployment

    def _await_ready(self, op: Operation, deployment: DeploymentRecord) -> None:
        ready = self.deployment_provider.await_ready(
            deployment.deployment_id, self.readiness_timeout_seconds
        )
        if not ready:
            raise ReadinessTimeout(
                f"deployment {deployment.deployment_id} not ready within "
                f"{self.readiness_timeout_seconds}s"
            )
        deployment.state = DeploymentState.READY
        self.store.save_deployment(deployment)
        self._record_transition(
            "deployment", deployment.deployment_id, op, "DEPLOYING", "READY"
        )
        reservation = self.store.get_reservation(op.reservation_id) if op.reservation_id else None
        if reservation is not None:
            reservation.state = ReservationState.READY
            reservation.revision = 3
            self.store.save_reservation(reservation, expected_revision=2)
            self._record_transition(
                "reservation", reservation.reservation_id, op, "DEPLOYING", "READY"
            )
            reservation.state = ReservationState.ACTIVE
            reservation.revision = 4
            self.store.save_reservation(reservation, expected_revision=3)
            self._record_transition(
                "reservation", reservation.reservation_id, op, "READY", "ACTIVE"
            )
        deployment.state = DeploymentState.ACTIVE
        self.store.save_deployment(deployment)
        self._record_transition(
            "deployment", deployment.deployment_id, op, "READY", "ACTIVE"
        )
        op.endpoint = deployment.endpoint

    # ================================================================ compensation

    def _compensate(self, op: Operation, *, reason: str, actor: str) -> RollbackPlan:
        plan = RollbackPlan(
            rollback_id=str(uuid.uuid4()),
            operation_id=op.operation_id,
            target_type="deployment" if op.deployment_id else "reservation",
            target_id=op.deployment_id or op.reservation_id or op.operation_id,
            reason=reason,
            state="IN_PROGRESS",
        )
        if op.deployment_id:
            plan.steps.append(
                RollbackStep(component="ignite", action=f"release deployment {op.deployment_id}")
            )
        if op.reservation_id:
            plan.steps.append(
                RollbackStep(component="fusion", action=f"release reservation {op.reservation_id}")
            )
        plan.steps.append(RollbackStep(component="platform", action="mark operation released"))

        ok = True
        for step in plan.steps:
            try:
                self._execute_step(op, step)
                step.status = "done"
            except Exception as exc:  # noqa: BLE001 - record but keep going
                step.status = "failed"
                step.detail = str(exc)
                ok = False

        plan.state = "COMPLETED" if ok else "FAILED"
        plan.completed_at = utcnow()
        self.store.record_audit(
            op.operation_id, "compensation", plan.model_dump(mode="json")
        )
        return plan

    def _execute_step(self, op: Operation, step: RollbackStep) -> None:
        if step.component == "ignite" and op.deployment_id:
            self.deployment_provider.release(op.deployment_id)
            deployment = self.store.get_deployment(op.deployment_id)
            if deployment is not None and deployment.state != DeploymentState.RELEASED:
                self._record_transition(
                    "deployment", deployment.deployment_id, op,
                    deployment.state.value, "RELEASING", actor="reconciler",
                )
                deployment.state = DeploymentState.RELEASED
                self.store.save_deployment(deployment)
                self._record_transition(
                    "deployment", deployment.deployment_id, op, "RELEASING", "RELEASED"
                )
        elif step.component == "fusion" and op.reservation_id:
            self.placement_provider.release(op.reservation_id)
            if self.capacity_ledger is not None:
                self.capacity_ledger.release(op.operation_id)
            reservation = self.store.get_reservation(op.reservation_id)
            if reservation is not None and reservation.state != ReservationState.RELEASED:
                prev = reservation.state.value
                reservation.state = ReservationState.RELEASING
                reservation.revision += 1
                self.store.save_reservation(reservation)
                self._record_transition(
                    "reservation", reservation.reservation_id, op, prev, "RELEASING"
                )
                reservation.state = ReservationState.RELEASED
                reservation.revision += 1
                self.store.save_reservation(reservation)
                self._record_transition(
                    "reservation", reservation.reservation_id, op, "RELEASING", "RELEASED"
                )
        elif step.component == "platform":
            self._emit(op, "release", replica_count=0)

    def _fail(self, op: Operation, exc: Exception) -> None:
        op.error = str(exc)
        self.store.save_operation(op)
        self.store.record_audit(op.operation_id, "operation_failed", {"error": str(exc)})
        if op.state not in (OperationState.FAILED, OperationState.RELEASED):
            try:
                self._set_operation_state(op, OperationState.FAILED, f"failed: {exc}")
            except Exception:  # noqa: BLE001 - state may already be terminal
                pass
        # Compensate: release any resources that were acquired.
        self._compensate(op, reason=str(exc), actor="coordinator.failure")

    # ================================================================ helpers

    def _require(self, operation_id: str) -> Operation:
        op = self.store.get_operation(operation_id)
        if op is None:
            raise CoordinatorError(f"unknown operation {operation_id!r}")
        return op

    def _set_operation_state(self, op: Operation, new_state: OperationState, reason: str) -> None:
        assert_transition(OperationState, op.state, new_state)
        previous = op.state.value
        op.state = new_state
        self.store.save_operation(op)
        if previous != new_state.value:
            self._record_transition(
                "operation", op.operation_id, op, previous, new_state.value, reason=reason
            )

    def _record_transition(
        self,
        entity_type: str,
        entity_id: str,
        op: Operation,
        from_state: Optional[str],
        to_state: str,
        reason: str = "",
        actor: str = "coordinator",
    ) -> None:
        self.store.record_transition(
            LifecycleTransition(
                entity_type=entity_type,
                entity_id=entity_id,
                operation_id=op.operation_id,
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                actor=actor,
                correlation_id=op.correlation_id,
            )
        )

    def _emit(self, op: Operation, event_type: str, **kwargs) -> None:
        if self.telemetry is None:
            return
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            correlation_id=op.correlation_id,
            workload_id=op.workload_id,
            deployment_id=op.deployment_id,
            reservation_id=op.reservation_id,
            event_type=event_type,
            evidence=kwargs.pop("evidence", "measured"),
            source=kwargs.pop("source", "platform-coordinator"),
            **kwargs,
        )
        self.telemetry.record(event)

    @staticmethod
    def _ms_between(start, end) -> Optional[float]:
        if start is None or end is None:
            return None
        return max(0.0, (end - start).total_seconds() * 1000.0)
