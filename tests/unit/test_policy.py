import pytest

from tesseract_platform.models import Evidence, WorkloadProfile
from tesseract_platform.policies import PlatformPolicy, PolicyError
from tesseract_platform.testing import make_request


def test_rejects_short_idempotency_key():
    policy = PlatformPolicy()
    request = make_request()
    request.idempotency_key = "short"
    with pytest.raises(PolicyError):
        policy.validate_workload(request)


def test_conservative_default_when_no_profile():
    policy = PlatformPolicy()
    profile = policy.resolve_profile(make_request(), None)
    assert profile.evidence is Evidence.estimated
    assert profile.source_type == "conservative_default"
    assert profile.ttft_p95_ms == 2000.0


def test_require_measured_evidence_rejects_estimated():
    policy = PlatformPolicy(require_measured_evidence=True)
    estimated = WorkloadProfile(
        profile_id="p1",
        model_id="m",
        model_revision="r",
        evidence=Evidence.estimated,
    )
    with pytest.raises(PolicyError):
        policy.resolve_profile(make_request(), estimated)


def test_missing_profile_rejected_when_defaults_disabled():
    policy = PlatformPolicy(allow_conservative_defaults=False)
    with pytest.raises(PolicyError):
        policy.resolve_profile(make_request(), None)


def test_measured_profile_passes_through():
    policy = PlatformPolicy(require_measured_evidence=True)
    measured = WorkloadProfile(
        profile_id="p1", model_id="m", model_revision="r", evidence=Evidence.measured
    )
    assert policy.resolve_profile(make_request(), measured) is measured
