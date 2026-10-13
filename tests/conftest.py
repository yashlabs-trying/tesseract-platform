"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from integrations.adaptive import DisabledAdaptiveProvider
from tesseract_platform.coordinator import WorkloadCoordinator
from tesseract_platform.observability import TelemetryCollector
from tesseract_platform.policies import PlatformPolicy
from tesseract_platform.state import StateStore
from tesseract_platform.testing import (
    FakeDeploymentProvider,
    FakePlacementProvider,
    FakeProfileProvider,
)


@pytest.fixture
def store():
    s = StateStore("sqlite:///:memory:")
    yield s
    s.close()


@pytest.fixture
def telemetry(store):
    return TelemetryCollector(store)


@pytest.fixture
def policy():
    return PlatformPolicy(reservation_ttl_seconds=900, readiness_timeout_seconds=30)


@pytest.fixture
def profile_provider():
    return FakeProfileProvider()


@pytest.fixture
def placement_provider():
    return FakePlacementProvider()


@pytest.fixture
def deployment_provider():
    return FakeDeploymentProvider()


@pytest.fixture
def coordinator(store, policy, telemetry, profile_provider, placement_provider, deployment_provider):
    return WorkloadCoordinator(
        store=store,
        policy=policy,
        profile_provider=profile_provider,
        placement_provider=placement_provider,
        deployment_provider=deployment_provider,
        precision_provider=DisabledAdaptiveProvider(),
        telemetry=telemetry,
    )
