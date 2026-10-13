"""Foundry -> Platform adapter.

Foundry emits ``tesseract.performance-profile.v1`` documents. This adapter:

1. validates the document against the contract,
2. maps it to the platform's :class:`WorkloadProfile`, and
3. can retrieve documents from a live Foundry over HTTP.

Important: ``evidence_type`` is preserved verbatim. Synthetic/estimated
documents are NOT real GPU measurements and must never be presented as such.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional

import httpx
from jsonschema import Draft7Validator

from integrations.base import IntegrationError, IntegrationUnavailable
from tesseract_platform.models import Evidence, WorkloadProfile, WorkloadRequest

FOUNDRY_PERFORMANCE_SCHEMA = "tesseract.performance-profile.v1"


def _to_evidence(value: Any) -> Evidence:
    try:
        return Evidence(str(value))
    except ValueError as exc:  # pragma: no cover - guarded by schema
        raise IntegrationError(f"unknown evidence_type: {value!r}") from exc


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def map_performance_profile(document: Mapping[str, Any]) -> WorkloadProfile:
    """Map a Foundry performance-profile document to a platform WorkloadProfile.

    Raises :class:`IntegrationError` if the document is not a valid
    ``tesseract.performance-profile.v1`` document.
    """
    schema_version = document.get("schema_version")
    if schema_version != FOUNDRY_PERFORMANCE_SCHEMA:
        raise IntegrationError(
            f"expected schema_version {FOUNDRY_PERFORMANCE_SCHEMA!r}, got {schema_version!r}"
        )
    for required in ("profile_id", "model_id"):
        if not document.get(required):
            raise IntegrationError(f"performance profile missing required field {required!r}")

    return WorkloadProfile(
        profile_id=str(document["profile_id"]),
        model_id=str(document["model_id"]),
        model_revision=str(document.get("model_revision", "unknown")),
        hardware_gpu_model=document.get("hardware_gpu_model"),
        hardware_gpu_count=_opt_int(document.get("hardware_gpu_count")) or 1,
        hardware_vram_gb=_opt_float(document.get("hardware_vram_gb")),
        engine=document.get("engine"),
        engine_version=document.get("engine_version"),
        ttft_p50_ms=_opt_float(document.get("ttft_p50_ms")),
        ttft_p95_ms=_opt_float(document.get("ttft_p95_ms")),
        ttft_p99_ms=_opt_float(document.get("ttft_p99_ms")),
        itl_p50_ms=_opt_float(document.get("itl_p50_ms")),
        itl_p95_ms=_opt_float(document.get("itl_p95_ms")),
        request_latency_p95_ms=_opt_float(document.get("request_latency_p95_ms")),
        completed_requests_per_second=_opt_float(document.get("completed_requests_per_second")),
        output_tokens_per_second=_opt_float(document.get("output_tokens_per_second")),
        input_tokens_per_second=_opt_float(document.get("input_tokens_per_second")),
        error_rate=_opt_float(document.get("error_rate")),
        concurrency=_opt_int(document.get("workload_concurrency")) or 1,
        evidence=_to_evidence(document.get("evidence_type", "estimated")),
        collection_timestamp=_opt_dt(document.get("collection_timestamp")),
        source_type=document.get("source_type"),
    )


class FoundryAdapter:
    """Foundry :class:`ProfileProvider` implementation.

    Document resolution order:

    1. an explicitly injected document (per ``model_id`` or a default);
    2. a live Foundry HTTP fetch by ``profile_id``.

    If neither yields a document the adapter raises
    :class:`IntegrationUnavailable`; the coordinator then applies explicit
    conservative defaults or fails, per policy.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        endpoint_template: str = "/v1/performance-profiles/{profile_id}",
        documents: Optional[dict[str, Mapping[str, Any]]] = None,
        default_document: Optional[Mapping[str, Any]] = None,
        profile_ids: Optional[dict[str, str]] = None,
        http: Optional[httpx.Client] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.endpoint_template = endpoint_template
        self.documents = documents or {}
        self.default_document = default_document
        self.profile_ids = profile_ids or {}
        self._http = http
        self.timeout = timeout
        self._schema_validator: Optional[Draft7Validator] = None

    # ---------------------------------------------------------------- schema

    def validate_document(self, document: Mapping[str, Any]) -> list[str]:
        """Validate a document against the vendored contract. Returns error strings."""
        if self._schema_validator is None:
            from tesseract_platform.contracts import load_schema

            self._schema_validator = Draft7Validator(load_schema("foundry-performance-profile"))
        return [f"{list(e.path)}: {e.message}" for e in self._schema_validator.iter_errors(document)]

    # ---------------------------------------------------------------- provider

    def get_profile(self, request: WorkloadRequest) -> WorkloadProfile:
        document = self._resolve_document(request)
        if document is None:
            raise IntegrationUnavailable(
                f"no Foundry performance profile available for model {request.model.model_id!r}"
            )
        return map_performance_profile(document)

    def _resolve_document(self, request: WorkloadRequest) -> Optional[Mapping[str, Any]]:
        doc = self.documents.get(request.model.model_id) or self.default_document
        if doc is not None:
            return doc
        profile_id = self.profile_ids.get(request.model.model_id)
        if profile_id and self.base_url:
            return self.fetch_profile_document(profile_id)
        return None

    def fetch_profile_document(self, profile_id: str) -> Mapping[str, Any]:
        if not self.base_url:
            raise IntegrationUnavailable("Foundry base_url is not configured")
        path = self.endpoint_template.format(profile_id=profile_id)
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        client = self._http or httpx.Client(timeout=self.timeout)
        try:
            response = client.get(f"{self.base_url}{path}", headers=headers)
        except httpx.HTTPError as exc:
            raise IntegrationUnavailable(f"Foundry request failed: {exc}") from exc
        finally:
            if self._http is None:
                client.close()
        if response.status_code == 404:
            raise IntegrationUnavailable(f"Foundry profile {profile_id!r} not found")
        if response.status_code >= 400:
            raise IntegrationError(f"Foundry returned HTTP {response.status_code}")
        return response.json()
