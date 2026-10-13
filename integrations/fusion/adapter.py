"""Fusion -> Platform adapter.

Fusion produces a ``FusionRecommendation`` (a *plan*, not a measurement). This
adapter maps the recommended plan into the platform's placement/reservation
contract and, in live mode, requests an evaluation from Fusion's
``/v1/optimization/evaluate`` endpoint.

Fusion is advisory: it does not hold a database lock over Kubernetes. The
platform therefore maintains the authoritative capacity ledger in its durable
state store and uses Fusion for the placement decision. This is exactly why the
reservation lifecycle is a reconcilable state machine rather than one
transaction.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, Optional

import httpx

from integrations.base import (
    IntegrationError,
    IntegrationUnavailable,
    PlacementResult,
)
from tesseract_platform.models import (
    Capacity,
    Evidence,
    PlacementSlot,
    PrecisionProfileRef,
    WorkloadProfile,
    WorkloadRequest,
)

# Fusion EvidenceType -> platform Evidence
_EVIDENCE_MAP = {
    "measured": Evidence.measured,
    "estimated": Evidence.estimated,
    "simulated": Evidence.synthetic,
    "extrapolated": Evidence.estimated,
    "unknown": Evidence.estimated,
}


def map_evidence(value: Any) -> Evidence:
    return _EVIDENCE_MAP.get(str(value).lower(), Evidence.estimated)


def _dec(value: Any) -> float:
    try:
        return float(Decimal(str(value)))
    except Exception:
        return 0.0


def map_recommendation(
    recommendation: Mapping[str, Any],
    *,
    require_execution_eligible: bool = False,
) -> PlacementResult:
    """Map a Fusion recommendation document into a :class:`PlacementResult`.

    Fusion is advisory by default (``execution_eligibility=False``,
    ``approval_status=advisory_only``); the platform is the execution authority,
    so advisory plans are accepted. Set ``require_execution_eligible=True`` to
    require Fusion to mark the plan executable instead.

    Raises :class:`IntegrationError` when Fusion reports no plan at all, so the
    coordinator fails the operation instead of reserving phantom capacity.
    """
    plan = recommendation.get("recommended_plan") or []
    if not plan:
        raise IntegrationError("Fusion recommendation has no recommended_plan")

    if require_execution_eligible and not recommendation.get("execution_eligibility", True):
        raise IntegrationError(
            "Fusion recommendation is not execution eligible "
            f"(approval_status={recommendation.get('approval_status')!r})"
        )

    placement: list[PlacementSlot] = []
    gpu_count = 0
    cost_per_hour = 0.0
    for alternative in plan:
        replicas = int(alternative.get("replica_count", 1) or 1)
        pool = str(alternative.get("gpu_pool_id", "unknown"))
        gpu_type = alternative.get("gpu_type")
        memory_mb = alternative.get("expected_memory_mb")
        evidence = Evidence.estimated
        perf = alternative.get("expected_performance") or {}
        if isinstance(perf, Mapping) and perf.get("evidence_type"):
            evidence = map_evidence(perf.get("evidence_type"))
        for index in range(replicas):
            placement.append(
                PlacementSlot(
                    gpu_id=f"{pool}-{index}",
                    node_id=pool,
                    gpu_model=gpu_type,
                    memory_mb=int(memory_mb) if memory_mb is not None else None,
                    isolation=str(alternative.get("warm_policy", "warm")),
                    evidence=evidence,
                )
            )
        gpu_count += replicas
        cost_per_hour += _dec(alternative.get("estimated_cost_per_hour", 0))

    return PlacementResult(
        reservation_id=str(recommendation.get("recommendation_id", "reservation")),
        placement=placement,
        capacity=Capacity(gpu_count=gpu_count, cost_per_hour=cost_per_hour),
        evidence=Evidence.estimated,
    )


class FusionAdapter:
    """Fusion :class:`PlacementProvider` implementation."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        evaluate_path: str = "/v1/optimization/evaluate",
        release_path: str = "/v1/reservations/{reservation_id}/release",
        http: Optional[httpx.Client] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.evaluate_path = evaluate_path
        self.release_path = release_path
        self._http = http
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(self, method: str, path: str, json_body: Optional[dict] = None) -> httpx.Response:
        if not self.base_url:
            raise IntegrationUnavailable("Fusion base_url is not configured")
        client = self._http or httpx.Client(timeout=self.timeout)
        try:
            response = client.request(
                method, f"{self.base_url}{path}", headers=self._headers(), json=json_body
            )
        except httpx.HTTPError as exc:
            raise IntegrationUnavailable(f"Fusion request failed: {exc}") from exc
        finally:
            if self._http is None:
                client.close()
        return response

    def build_scenario(
        self,
        request: WorkloadRequest,
        profile: WorkloadProfile,
        precision: Optional[PrecisionProfileRef] = None,
    ) -> dict:
        """Translate a platform workload request into a Fusion scenario."""
        model = request.model
        return {
            "scenario": {
                "scenario_id": request.workload_id,
                "candidate_allocation_policy": "economic",
                "resource_constraints": {
                    "min_gpu_count": request.placement.min_gpu_count,
                    "preferred_gpu_model": request.placement.preferred_gpu_model,
                },
                "slo_constraints": {
                    "ttft_p95_ms": request.slo.ttft_p95_ms,
                    "itl_p95_ms": request.slo.itl_p95_ms,
                    "request_p99_ms": request.slo.request_p99_ms,
                },
                "workload": {
                    "model_id": model.model_id,
                    "model_revision": model.revision,
                    "precision": (precision.method if precision else model.precision),
                    "concurrency": request.workload.concurrency,
                    "arrival_rate_rps": request.workload.arrival_rate_rps,
                    "input_tokens_mean": request.workload.input_tokens_mean,
                    "output_tokens_mean": request.workload.output_tokens_mean,
                    "replicas": request.placement.replicas,
                    "profile_id": profile.profile_id,
                    "evidence_type": profile.evidence.value,
                },
            }
        }

    def reserve(
        self,
        request: WorkloadRequest,
        profile: WorkloadProfile,
        precision: Optional[PrecisionProfileRef] = None,
    ) -> PlacementResult:
        response = self._request(
            "POST", self.evaluate_path, self.build_scenario(request, profile, precision)
        )
        if response.status_code >= 400:
            raise IntegrationError(f"Fusion evaluate returned HTTP {response.status_code}")
        return map_recommendation(response.json())

    def release(self, reservation_id: str) -> None:
        """Release a reservation.

        Fusion is advisory and has no native release endpoint today. When one is
        configured the adapter calls it; otherwise releasing is a logical no-op
        and the platform's own capacity ledger is authoritative.
        """
        if not self.base_url or not self.release_path:
            return
        try:
            response = self._request(
                "POST", self.release_path.format(reservation_id=reservation_id)
            )
        except IntegrationUnavailable:
            return
        if response.status_code >= 500:
            raise IntegrationError(f"Fusion release returned HTTP {response.status_code}")
