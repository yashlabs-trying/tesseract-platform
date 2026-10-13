from concurrent.futures import ThreadPoolExecutor

import pytest

from integrations.adaptive import DisabledAdaptiveProvider
from tesseract_platform.coordinator import WorkloadCoordinator
from tesseract_platform.models import OperationState
from tesseract_platform.state import CapacityExhausted, CapacityLedger
from tesseract_platform.testing import (
    FakeDeploymentProvider,
    FakePlacementProvider,
    FakeProfileProvider,
    make_request,
)


def test_ledger_invariant_basic():
    ledger = CapacityLedger(10)
    ledger.reserve("a", 4)
    assert ledger.active_reserved == 4 and ledger.available == 6
    ledger.commit("a")
    assert ledger.committed == 4
    ledger.reserve("b", 6)
    with pytest.raises(CapacityExhausted):
        ledger.reserve("c", 1)
    assert ledger.invariant_ok()
    ledger.release("a")
    assert ledger.available == 4  # "b" still holds 6


def test_concurrent_reservations_never_exceed_capacity():
    ledger = CapacityLedger(10)

    def attempt(index: int):
        try:
            ledger.reserve(f"owner-{index}", 1)
            return True
        except CapacityExhausted:
            return False

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(attempt, range(50)))

    assert sum(results) == 10
    assert ledger.active_reserved == 10
    assert ledger.invariant_ok()


def _coordinator(store, policy, telemetry, ledger, gpu_count=1):
    return WorkloadCoordinator(
        store=store,
        policy=policy,
        profile_provider=FakeProfileProvider(),
        placement_provider=FakePlacementProvider(gpu_count=gpu_count),
        deployment_provider=FakeDeploymentProvider(),
        precision_provider=DisabledAdaptiveProvider(),
        telemetry=telemetry,
        capacity_ledger=ledger,
    )


def test_coordinator_enforces_capacity_under_concurrency(store, policy, telemetry):
    ledger = CapacityLedger(2)
    coordinator = _coordinator(store, policy, telemetry, ledger, gpu_count=1)

    def submit(index: int):
        return coordinator.submit(
            make_request(workload_id=f"wl-{index}", idempotency_key=f"idem-key-{index:06d}")
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        ops = list(pool.map(submit, range(8)))

    active = [op for op in ops if op.state is OperationState.ACTIVE]
    failed = [op for op in ops if op.state is OperationState.FAILED]
    assert len(active) == 2
    assert len(failed) == 6
    assert ledger.committed == 2
    assert ledger.invariant_ok()


def test_failed_deployment_releases_capacity(store, policy, telemetry):
    ledger = CapacityLedger(1)
    coordinator = WorkloadCoordinator(
        store=store,
        policy=policy,
        profile_provider=FakeProfileProvider(),
        placement_provider=FakePlacementProvider(),
        deployment_provider=FakeDeploymentProvider(fail=True),
        precision_provider=DisabledAdaptiveProvider(),
        telemetry=telemetry,
        capacity_ledger=ledger,
    )
    op = coordinator.submit(make_request())
    assert op.state is OperationState.FAILED
    assert ledger.used == 0  # capacity released by compensation
    assert ledger.invariant_ok()


def test_released_capacity_is_reusable(store, policy, telemetry):
    ledger = CapacityLedger(1)
    coordinator = _coordinator(store, policy, telemetry, ledger)
    first = coordinator.submit(make_request(workload_id="wl-1", idempotency_key="idem-key-0001"))
    assert first.state is OperationState.ACTIVE
    # Second request cannot fit while the first holds capacity.
    second = coordinator.submit(make_request(workload_id="wl-2", idempotency_key="idem-key-0002"))
    assert second.state is OperationState.FAILED
    # Release the first, then the second can fit.
    coordinator.release(first.operation_id)
    assert ledger.used == 0
    third = coordinator.submit(make_request(workload_id="wl-3", idempotency_key="idem-key-0003"))
    assert third.state is OperationState.ACTIVE


def test_released_owner_cannot_commit():
    ledger = CapacityLedger(4)
    ledger.reserve("x", 2)
    ledger.release("x")
    with pytest.raises(KeyError):
        ledger.commit("x")
