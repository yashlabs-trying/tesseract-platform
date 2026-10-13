"""FastAPI control-plane application.

Endpoints cover the full lifecycle: submit, inspect, release/rollback,
telemetry ingest/summary, reconciliation and controller evaluation.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import ValidationError

from tesseract_platform import __version__
from tesseract_platform.coordinator import CoordinatorError
from tesseract_platform.models import SLOSpec, WorkloadRequest
from tesseract_platform.policies import PolicyError
from tesseract_platform.runtime import Platform, build_platform


def create_app(platform: Optional[Platform] = None) -> FastAPI:
    app = FastAPI(
        title="Tesseract Platform",
        version=__version__,
        description="Unified orchestration and optimization layer for Tesseract.",
    )
    plat = platform or build_platform()
    app.state.platform = plat
    settings = plat.settings

    def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
        if settings.api_key_enabled:
            if not settings.api_key or x_api_key != settings.api_key:
                raise HTTPException(status_code=401, detail="invalid or missing API key")
        return x_api_key

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def readyz():
        return {
            "status": "ready",
            "foundry": bool(plat.coordinator.profile_provider),
            "fusion": bool(plat.coordinator.placement_provider),
            "ignite": bool(plat.coordinator.deployment_provider),
            "adaptive": bool(getattr(plat.precision_provider, "available", False)),
        }

    @app.post("/v1/workloads")
    def submit_workload(body: dict, api_key: str = Depends(require_api_key)):
        try:
            request = WorkloadRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        try:
            op = plat.coordinator.submit(request)
        except PolicyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return op.model_dump(mode="json")

    @app.get("/v1/workloads/{operation_id}")
    def get_workload(operation_id: str):
        op = plat.coordinator.get(operation_id)
        if op is None:
            raise HTTPException(status_code=404, detail="operation not found")
        return op.model_dump(mode="json")

    @app.get("/v1/workloads/{operation_id}/lifecycle")
    def get_lifecycle(operation_id: str):
        return [t.model_dump(mode="json") for t in plat.store.list_transitions(operation_id)]

    @app.post("/v1/workloads/{operation_id}/release")
    def release_workload(operation_id: str, api_key: str = Depends(require_api_key)):
        try:
            op = plat.coordinator.release(operation_id)
        except CoordinatorError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return op.model_dump(mode="json")

    @app.post("/v1/workloads/{operation_id}/rollback")
    def rollback_workload(
        operation_id: str, reason: str = "api_rollback", api_key: str = Depends(require_api_key)
    ):
        try:
            plan = plat.coordinator.rollback(operation_id, reason=reason)
        except CoordinatorError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return plan.model_dump(mode="json")

    @app.post("/v1/telemetry")
    def ingest_telemetry(body: dict, api_key: str = Depends(require_api_key)):
        try:
            event = plat.telemetry.ingest(body)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return event.model_dump(mode="json")

    @app.get("/v1/telemetry/{workload_id}/summary")
    def telemetry_summary(workload_id: str):
        return plat.telemetry.aggregate(workload_id).model_dump(mode="json")

    @app.post("/v1/reconcile")
    def reconcile(api_key: str = Depends(require_api_key)):
        return {"actions": plat.reconciler.reconcile()}

    @app.post("/v1/controller/evaluate")
    def controller_evaluate(body: dict, api_key: str = Depends(require_api_key)):
        workload_id = body.get("workload_id")
        if not workload_id:
            raise HTTPException(status_code=422, detail="workload_id is required")
        slo = SLOSpec.model_validate(body.get("slo", {}))
        return plat.controller.evaluate(workload_id, slo).model_dump(mode="json")

    return app
