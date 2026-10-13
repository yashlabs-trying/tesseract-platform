import pytest

from integrations.adaptive import (
    DisabledAdaptiveProvider,
    UnavailableAdaptiveProvider,
    build_adaptive_provider,
)
from integrations.base import IntegrationUnavailable
from tesseract_platform.config import Settings
from tesseract_platform.models import Evidence, OperationState, PrecisionProfileRef
from tesseract_platform.testing import make_request


def test_platform_operates_without_adaptive(coordinator):
    op = coordinator.submit(make_request())
    assert op.state is OperationState.ACTIVE
    assert op.precision_profile_ref is None


def test_disabled_provider_is_safe():
    provider = DisabledAdaptiveProvider()
    assert provider.available is False
    assert provider.list_profiles("m") == []
    assert provider.recommend_profile(make_request()) is None
    profile = PrecisionProfileRef(
        profile_ref="x", model_id="m", runtime="vllm", method="awq",
        deployable=True, evidence=Evidence.validated,
    )
    assert provider.validate_profile(profile) is False
    assert provider.plan_transition(None, profile)["status"] == "unavailable"
    assert provider.plan_rollback({})["status"] == "unavailable"


def test_unavailable_provider_fails_loudly():
    provider = UnavailableAdaptiveProvider()
    with pytest.raises(IntegrationUnavailable):
        provider.list_profiles("m")


def test_build_adaptive_provider_defaults_disabled():
    assert isinstance(build_adaptive_provider(Settings(adaptive_enabled=False)), DisabledAdaptiveProvider)


def test_enabled_without_url_is_unavailable():
    assert isinstance(
        build_adaptive_provider(Settings(adaptive_enabled=True, adaptive_url=None)),
        UnavailableAdaptiveProvider,
    )


class _FakeAdaptiveProvider:
    available = True
    enabled = True

    def recommend_profile(self, request, capacity=None):
        return PrecisionProfileRef(
            profile_ref="adaptive://qwen3-0.6b/awq-int4",
            model_id=request.model.model_id,
            runtime="vllm",
            method="awq",
            deployable=True,
            evidence=Evidence.validated,
            bits=4,
        )

    def validate_profile(self, profile):
        return profile.deployable and profile.evidence is Evidence.validated

    def list_profiles(self, model_id):
        return []

    def evaluate_constraints(self, request):
        return []

    def plan_transition(self, current, target):
        return {"status": "planned"}

    def plan_rollback(self, transition):
        return {"status": "planned"}


def test_precision_provider_flows_into_deployment(store, policy, telemetry):
    from tesseract_platform.coordinator import WorkloadCoordinator
    from tesseract_platform.testing import (
        FakeDeploymentProvider,
        FakePlacementProvider,
        FakeProfileProvider,
    )

    coordinator = WorkloadCoordinator(
        store=store,
        policy=policy,
        profile_provider=FakeProfileProvider(),
        placement_provider=FakePlacementProvider(),
        deployment_provider=FakeDeploymentProvider(),
        precision_provider=_FakeAdaptiveProvider(),
        telemetry=telemetry,
    )
    op = coordinator.submit(make_request())
    assert op.state is OperationState.ACTIVE
    assert op.precision_profile_ref == "adaptive://qwen3-0.6b/awq-int4"
    deployment = store.get_deployment(op.deployment_id)
    assert deployment.precision_profile_ref == "adaptive://qwen3-0.6b/awq-int4"
    assert deployment.model.precision == "awq"
