"""Property-based tests for controller anti-oscillation behaviour."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from adaptive.controller import Controller, ControllerAction, ControllerPolicy, ProfileOption
from adaptive.models.controller import TelemetrySnapshot
from adaptive.models.enums import EvidenceType

settings.register_profile("ci", max_examples=50, deadline=None)
settings.load_profile("ci")


def _options() -> list[ProfileOption]:
    return [
        ProfileOption(
            profile_id="p4", precision_bits=4, estimated_cost=1.0, weight_bytes=4_000_000_000
        ),
        ProfileOption(
            profile_id="p8", precision_bits=8, estimated_cost=2.0, weight_bytes=8_000_000_000
        ),
        ProfileOption(
            profile_id="p16", precision_bits=16, estimated_cost=4.0, weight_bytes=16_000_000_000
        ),
    ]


@given(pressure=st.floats(0.65, 0.85, allow_nan=False, allow_infinity=False))
def test_deadband_pressure_holds(pressure: float) -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    controller = Controller(policy, _options(), "p16")
    snapshot = TelemetrySnapshot(
        memory_pressure=pressure,
        ttft_ms=100.0,
        itl_ms=20.0,
        tokens_per_second=200.0,
        evidence=EvidenceType.synthetic,
    )
    decision = controller.recommend(snapshot, now=0.0)
    assert decision.action is ControllerAction.hold


@given(
    pressure=st.floats(0.96, 1.0, allow_nan=False, allow_infinity=False),
)
def test_high_pressure_always_downshifts(pressure: float) -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    controller = Controller(policy, _options(), "p16")
    snapshot = TelemetrySnapshot(
        memory_pressure=pressure,
        ttft_ms=100.0,
        itl_ms=20.0,
        tokens_per_second=200.0,
        evidence=EvidenceType.synthetic,
    )
    decision = controller.recommend(snapshot, now=0.0)
    assert decision.action is ControllerAction.downshift


def test_recommend_is_deterministic() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    snapshot = TelemetrySnapshot(
        memory_pressure=0.5, ttft_ms=100.0, itl_ms=20.0, tokens_per_second=200.0
    )
    first = Controller(policy, _options(), "p4").recommend(snapshot, now=0.0)
    second = Controller(policy, _options(), "p4").recommend(snapshot, now=0.0)
    assert first.model_dump() == second.model_dump()
