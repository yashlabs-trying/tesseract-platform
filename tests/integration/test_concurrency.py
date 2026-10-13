from concurrent.futures import ThreadPoolExecutor

from tesseract_platform.models import OperationState
from tesseract_platform.testing import make_request


def test_100_concurrent_distinct_requests(coordinator):
    def submit(index: int):
        return coordinator.submit(
            make_request(workload_id=f"wl-{index}", idempotency_key=f"idem-key-{index:06d}")
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(submit, range(100)))

    assert len(results) == 100
    assert all(op.state is OperationState.ACTIVE for op in results)
    assert len({op.operation_id for op in results}) == 100


def test_concurrent_duplicate_idempotency_key(coordinator, placement_provider):
    request = make_request()

    def submit(_: int):
        return coordinator.submit(request)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(16)))

    assert len({op.operation_id for op in results}) == 1
    assert len(placement_provider.reserved) == 1
