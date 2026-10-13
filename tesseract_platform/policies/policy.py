"""Orchestration policy.

Policies encode the safety rules the coordinator must enforce:

* input validation (reject malformed workloads early);
* evidence policy: if Foundry cannot provide a real measurement, the platform
  either applies **explicit conservative defaults** or rejects the operation —
  it never invents a measured profile.
"""

from __future__ import annotations

from typing import Optional

from tesseract_platform.models import Evidence, WorkloadProfile, WorkloadRequest

# Deliberately pessimistic defaults used only when no measurement is available.
CONSERVATIVE_DEFAULTS = {
    "ttft_p95_ms": 2000.0,
    "itl_p95_ms": 100.0,
    "completed_requests_per_second": 0.1,
    "output_tokens_per_second": 1.0,
}


class PolicyError(ValueError):
    """A workload or profile violates platform policy."""


class PlatformPolicy:
    def __init__(
        self,
        *,
        require_measured_evidence: bool = False,
        allow_conservative_defaults: bool = True,
        reservation_ttl_seconds: int = 900,
        readiness_timeout_seconds: int = 300,
        default_gpu_model: str = "unknown",
    ) -> None:
        self.require_measured_evidence = require_measured_evidence
        self.allow_conservative_defaults = allow_conservative_defaults
        self.reservation_ttl_seconds = reservation_ttl_seconds
        self.readiness_timeout_seconds = readiness_timeout_seconds
        self.default_gpu_model = default_gpu_model

    def validate_workload(self, request: WorkloadRequest) -> None:
        errors: list[str] = []
        if not request.workload_id:
            errors.append("workload_id is required")
        if len(request.idempotency_key) < 8:
            errors.append("idempotency_key must be at least 8 characters")
        if request.placement.min_gpu_count < 1:
            errors.append("placement.min_gpu_count must be >= 1")
        if request.workload.concurrency < 1:
            errors.append("workload.concurrency must be >= 1")
        if request.workload.arrival_rate_rps < 0:
            errors.append("workload.arrival_rate_rps must be >= 0")
        for name in ("ttft_p95_ms", "itl_p95_ms", "request_p99_ms"):
            value = getattr(request.slo, name)
            if value is not None and value < 0:
                errors.append(f"slo.{name} must be >= 0")
        if errors:
            raise PolicyError("; ".join(errors))

    def resolve_profile(
        self, request: WorkloadRequest, profile: Optional[WorkloadProfile]
    ) -> WorkloadProfile:
        """Return a usable profile or raise :class:`PolicyError`.

        Applies conservative defaults when telemetry is missing, unless the
        platform is configured to require real measurements.
        """
        if profile is None:
            if not self.allow_conservative_defaults:
                raise PolicyError(
                    "no Foundry profile available and conservative defaults are disabled"
                )
            return self.conservative_default(request)
        if self.require_measured_evidence and not profile.evidence.is_real_measurement:
            raise PolicyError(
                f"profile {profile.profile_id!r} has evidence={profile.evidence.value!r}; "
                "measured evidence is required"
            )
        return profile

    def conservative_default(self, request: WorkloadRequest) -> WorkloadProfile:
        return WorkloadProfile(
            profile_id=f"conservative-{request.model.model_id}",
            model_id=request.model.model_id,
            model_revision=request.model.revision,
            hardware_gpu_model=self.default_gpu_model,
            engine=None,
            concurrency=request.workload.concurrency,
            evidence=Evidence.estimated,
            source_type="conservative_default",
            ttft_p95_ms=CONSERVATIVE_DEFAULTS["ttft_p95_ms"],
            itl_p95_ms=CONSERVATIVE_DEFAULTS["itl_p95_ms"],
            completed_requests_per_second=CONSERVATIVE_DEFAULTS["completed_requests_per_second"],
            output_tokens_per_second=CONSERVATIVE_DEFAULTS["output_tokens_per_second"],
        )
