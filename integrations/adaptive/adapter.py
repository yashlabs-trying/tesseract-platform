"""Adaptive provider implementations.

* :class:`DisabledAdaptiveProvider` — Adaptive intentionally off. Safe default.
* :class:`UnavailableAdaptiveProvider` — Adaptive expected but unreachable.
* :class:`HttpAdaptiveProvider` — scaffold for a live Adaptive service.

None of these fabricate quantization results. With no provider the platform
simply deploys the model's declared precision.
"""

from __future__ import annotations

from typing import Optional

import httpx

from integrations.base import IntegrationUnavailable
from tesseract_platform.models import Capacity, PrecisionProfileRef, WorkloadRequest


class DisabledAdaptiveProvider:
    """Adaptive is disabled. All queries return 'nothing available'."""

    available = False
    enabled = False

    def list_profiles(self, model_id: str) -> list[PrecisionProfileRef]:
        return []

    def evaluate_constraints(self, request: WorkloadRequest) -> list[str]:
        return []

    def recommend_profile(
        self, request: WorkloadRequest, capacity: Optional[Capacity] = None
    ) -> Optional[PrecisionProfileRef]:
        return None

    def validate_profile(self, profile: PrecisionProfileRef) -> bool:
        return False

    def plan_transition(
        self, current: Optional[PrecisionProfileRef], target: PrecisionProfileRef
    ) -> dict:
        return {"schema_version": "tesseract.platform.precision-transition.v1",
                "status": "unavailable", "reason": "adaptive disabled"}

    def plan_rollback(self, transition: dict) -> dict:
        return {"schema_version": "tesseract.platform.precision-transition.v1",
                "status": "unavailable", "reason": "adaptive disabled"}


class UnavailableAdaptiveProvider:
    """Adaptive is enabled but not reachable. Fails loudly instead of silently."""

    available = False
    enabled = True

    def _fail(self):
        raise IntegrationUnavailable("Adaptive provider is enabled but unavailable")

    def list_profiles(self, model_id: str) -> list[PrecisionProfileRef]:
        self._fail()

    def evaluate_constraints(self, request: WorkloadRequest) -> list[str]:
        self._fail()

    def recommend_profile(self, request: WorkloadRequest, capacity=None):
        self._fail()

    def validate_profile(self, profile: PrecisionProfileRef) -> bool:
        self._fail()

    def plan_transition(self, current, target) -> dict:
        self._fail()

    def plan_rollback(self, transition: dict) -> dict:
        self._fail()


class HttpAdaptiveProvider:
    """Scaffold for a live Adaptive service.

    Endpoint paths are placeholders until the Adaptive repository publishes its
    service interface; they are configurable and never assumed to exist.
    """

    available = True
    enabled = True

    def __init__(
        self,
        base_url: str,
        *,
        http: Optional[httpx.Client] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = http
        self.timeout = timeout

    def _get(self, path: str) -> httpx.Response:
        client = self._http or httpx.Client(timeout=self.timeout)
        try:
            return client.get(f"{self.base_url}{path}")
        except httpx.HTTPError as exc:
            raise IntegrationUnavailable(f"Adaptive request failed: {exc}") from exc
        finally:
            if self._http is None:
                client.close()

    def list_profiles(self, model_id: str) -> list[PrecisionProfileRef]:
        response = self._get(f"/v1/models/{model_id}/precision-profiles")
        if response.status_code >= 400:
            raise IntegrationUnavailable(f"Adaptive returned HTTP {response.status_code}")
        return [PrecisionProfileRef.model_validate(item) for item in response.json()]

    def evaluate_constraints(self, request: WorkloadRequest) -> list[str]:
        return []

    def recommend_profile(self, request: WorkloadRequest, capacity=None):
        return None

    def validate_profile(self, profile: PrecisionProfileRef) -> bool:
        return profile.deployable and profile.evidence.value in ("measured", "validated")

    def plan_transition(self, current, target) -> dict:
        return {"schema_version": "tesseract.platform.precision-transition.v1",
                "status": "planned", "from": current.profile_ref if current else None,
                "to": target.profile_ref}

    def plan_rollback(self, transition: dict) -> dict:
        return {"schema_version": "tesseract.platform.precision-transition.v1",
                "status": "planned", "reverse_of": transition.get("to")}


def build_adaptive_provider(settings) -> object:
    """Select an Adaptive provider from configuration.

    Returns a *disabled* provider unless Adaptive is explicitly enabled.
    """
    if not getattr(settings, "adaptive_enabled", False):
        return DisabledAdaptiveProvider()
    url = getattr(settings, "adaptive_url", None)
    if not url:
        return UnavailableAdaptiveProvider()
    return HttpAdaptiveProvider(url)
