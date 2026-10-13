from integrations.adaptive import DisabledAdaptiveProvider
from tesseract_platform.coordinator import WorkloadCoordinator
from tesseract_platform.models import OperationState
from tesseract_platform.testing import (
    FakeDeploymentProvider,
    FakePlacementProvider,
    FakeProfileProvider,
    make_request,
)


def _coordinator(store, policy, telemetry, *, profile, placement, deployment):
    return WorkloadCoordinator(
        store=store,
        policy=policy,
        profile_provider=profile,
        placement_provider=placement,
        deployment_provider=deployment,
        precision_provider=DisabledAdaptiveProvider(),
        telemetry=telemetry,
    )


def test_fusion_failure_fails_without_reserving(store, policy, telemetry):
    placement = FakePlacementProvider(fail=True)
    coordinator = _coordinator(
        store,
        policy,
        telemetry,
        profile=FakeProfileProvider(),
        placement=placement,
        deployment=FakeDeploymentProvider(),
    )
    op = coordinator.submit(make_request())
    assert op.state is OperationState.FAILED
    assert op.reservation_id is None
    assert placement.reserved == []


def test_deployment_failure_releases_reservation(store, policy, telemetry):
    placement = FakePlacementProvider()
    deployment = FakeDeploymentProvider(fail=True)
    coordinator = _coordinator(
        store,
        policy,
        telemetry,
        profile=FakeProfileProvider(),
        placement=placement,
        deployment=deployment,
    )
    op = coordinator.submit(make_request())
    assert op.state is OperationState.FAILED
    assert op.reservation_id is not None
    assert placement.released == [op.reservation_id]


def test_readiness_timeout_compensates(store, policy, telemetry):
    placement = FakePlacementProvider()
    deployment = FakeDeploymentProvider(ready=False)
    coordinator = _coordinator(
        store,
        policy,
        telemetry,
        profile=FakeProfileProvider(),
        placement=placement,
        deployment=deployment,
    )
    op = coordinator.submit(make_request())
    assert op.state is OperationState.FAILED
    assert placement.released == [op.reservation_id]
    assert deployment.released == [op.deployment_id]


def test_foundry_unavailable_uses_conservative_default(store, policy, telemetry):
    placement = FakePlacementProvider()
    deployment = FakeDeploymentProvider()
    coordinator = _coordinator(
        store,
        policy,
        telemetry,
        profile=FakeProfileProvider(fail=True),
        placement=placement,
        deployment=deployment,
    )
    op = coordinator.submit(make_request())
    assert op.state is OperationState.ACTIVE
    assert op.profile is not None
    assert op.profile.source_type == "conservative_default"


def test_explicit_rollback_completes(store, policy, telemetry, coordinator):
    op = coordinator.submit(make_request())
    plan = coordinator.rollback(op.operation_id, reason="test")
    assert plan.state == "COMPLETED"
    final = coordinator.get(op.operation_id)
    assert final.state is OperationState.RELEASED
