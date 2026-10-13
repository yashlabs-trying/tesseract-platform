"""Property-based tests for optimizer constraints and Pareto correctness."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from adaptive.models.enums import DType, EvidenceType, RuntimeBackend
from adaptive.models.quantization import PrecisionProfile, baseline_config, int_config
from adaptive.models.sensitivity import SensitivityMethod, SensitivityReport
from adaptive.optimizer.pareto import dominates, is_antichain, pareto_frontier
from adaptive.optimizer.quality_estimator import SensitivityQualityEstimator, bit_penalty

settings.register_profile("ci", max_examples=50, deadline=None)
settings.load_profile("ci")

_finite = st.floats(0.0, 1000.0, allow_nan=False, allow_infinity=False)


@given(values=st.lists(st.tuples(_finite, _finite), min_size=1, max_size=8))
def test_pareto_frontier_is_valid(values: list[tuple[float, float]]) -> None:
    points = [(f"p{i}", cost, loss) for i, (cost, loss) in enumerate(values)]
    frontier = set(pareto_frontier(points))
    frontier_points = [p for p in points if p[0] in frontier]
    assert is_antichain(frontier_points)
    for point in points:
        if point[0] in frontier:
            continue
        assert any(dominates(member, point) for member in frontier_points)


@given(values=st.lists(st.tuples(_finite, _finite), min_size=1, max_size=6))
def test_frontier_nonempty(values: list[tuple[float, float]]) -> None:
    points = [(f"p{i}", cost, loss) for i, (cost, loss) in enumerate(values)]
    assert pareto_frontier(points)


@given(
    a=st.tuples(_finite, _finite),
    b=st.tuples(_finite, _finite),
)
def test_dominance_is_asymmetric(a: tuple[float, float], b: tuple[float, float]) -> None:
    pa = ("a", a[0], a[1])
    pb = ("b", b[0], b[1])
    if dominates(pa, pb):
        assert not dominates(pb, pa)


@given(bits=st.sampled_from([4, 8, 16, 32]))
def test_bit_penalty_monotonic(bits: int) -> None:
    assert bit_penalty(bits) >= bit_penalty(bits + 4)


def _report() -> SensitivityReport:
    layers = ["model.layers.0.mlp.up_proj", "model.layers.0.self_attn.q_proj"]
    scores = {
        layers[0]: {SensitivityMethod.activation_weighted.value: 1.0},
        layers[1]: {SensitivityMethod.activation_weighted.value: 0.2},
    }
    return SensitivityReport(
        model_id="m",
        structural_fingerprint="fp",
        calibration_fingerprint="cal",
        methods=(SensitivityMethod.activation_weighted,),
        bits=4,
        scores=scores,
        raw_scores=scores,
        rankings={SensitivityMethod.activation_weighted.value: layers},
        family_attribution={},
        evidence=EvidenceType.estimated,
    )


@given(bits=st.sampled_from([4, 8, 16]))
def test_estimator_degradation_nonnegative(bits: int) -> None:
    estimator = SensitivityQualityEstimator(_report(), baseline_value=100.0)
    config = int_config(bits) if bits < 16 else baseline_config(DType.fp16)
    profile = PrecisionProfile(
        profile_id="p",
        name="p",
        default_config=config,
        runtime=RuntimeBackend.cuda,
    )
    assert estimator.degradation(profile) >= 0.0


def test_estimator_zero_at_full_precision() -> None:
    estimator = SensitivityQualityEstimator(_report(), baseline_value=100.0)
    profile = PrecisionProfile(
        profile_id="p",
        name="p",
        default_config=baseline_config(DType.fp16),
        runtime=RuntimeBackend.cuda,
    )
    assert estimator.degradation(profile) == 0.0
    assert estimator.predict(profile) == 100.0
