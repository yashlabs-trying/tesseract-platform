"""Phase 5 - adaptive controller safety and adversarial telemetry."""

from __future__ import annotations

import pytest

from adaptive.controller import (
    Controller,
    ControllerAction,
    ControllerPolicy,
    ProfileOption,
    run_simulation,
)
from adaptive.models.controller import TelemetrySnapshot
from adaptive.models.enums import EvidenceType


def _options() -> list[ProfileOption]:
    return [
        ProfileOption(
            profile_id="p4", precision_bits=4, estimated_cost=1.0,
            weight_bytes=4_000_000_000, quality_degradation=0.06,
        ),
        ProfileOption(
            profile_id="p8", precision_bits=8, estimated_cost=2.0,
            weight_bytes=8_000_000_000, quality_degradation=0.01,
        ),
        ProfileOption(
            profile_id="p16", precision_bits=16, estimated_cost=4.0,
            weight_bytes=16_000_000_000, quality_degradation=0.0,
        ),
    ]


def _snap(ts: float, pressure: float, **kw: float) -> TelemetrySnapshot:
    base = dict(
        timestamp_s=ts,
        memory_pressure=pressure,
        memory_used_gb=pressure * 80,
        memory_total_gb=80,
        ttft_ms=120.0,
        itl_ms=20.0,
        tokens_per_second=200.0,
        model_demand=10.0,
        evidence=EvidenceType.synthetic,
    )
    base.update(kw)
    return TelemetrySnapshot(**base)


def test_d01_progressive_memory_pressure() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    controller = Controller(policy, _options(), "p16")
    actions = []
    for i, pressure in enumerate([0.5, 0.7, 0.85, 0.9, 0.96]):
        actions.append(controller.recommend(_snap(i * 60.0, pressure)).action)
    # no switch until pressure crosses high + hysteresis (0.95)
    assert actions[0] is ControllerAction.hold
    assert actions[-1] is ControllerAction.downshift


@pytest.mark.parametrize("multiplier", [2, 5, 10])
def test_d02_traffic_bursts(multiplier: int) -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    demand = 10.0 * multiplier
    snapshots = [
        _snap(0.0, 0.5),
        _snap(
            30.0,
            0.92,
            model_demand=demand,
            ttft_ms=120.0 * multiplier,
            itl_ms=20.0 * multiplier,
            tokens_per_second=200.0,
        ),
    ]
    result = run_simulation(policy, _options(), snapshots, automatic=True)
    assert result.oscillated is False
    if multiplier >= 5:
        # 5x/10x breach the TTFT/ITL SLO and must trigger a downshift
        assert result.switch_count >= 1
    else:
        # a 2x burst stays within SLO; the controller must not thrash
        assert result.switch_count == 0


def test_d03_quality_regression_rejected_for_upshift() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0, quality_tolerance=0.05)
    controller = Controller(policy, _options(), "p4")
    # p16 is lossless and eligible; the only higher option
    decision = controller.recommend(_snap(0.0, 0.3))
    assert decision.action is ControllerAction.upshift
    assert decision.recommended_profile_id == "p16"


def test_d04_hysteresis_band_no_oscillation() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0, hysteresis=0.05)
    snapshots = [_snap(i * 60.0, 0.9 if i % 2 else 0.92) for i in range(6)]
    result = run_simulation(policy, _options(), snapshots, automatic=True)
    assert result.switch_count == 0


def test_d05_duplicate_transition_is_idempotent() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    controller = Controller(policy, _options(), "p16")
    controller.enable_automatic()
    controller.record_switch("p8", now=0.0)
    controller.record_switch("p8", now=1.0)  # duplicate
    assert controller.current_profile_id == "p8"
    assert len(controller.switch_timestamps) == 1


def test_d05_failed_load_rolls_back() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    from adaptive.telemetry import memory_pressure_scenario

    result = run_simulation(
        policy, _options(), memory_pressure_scenario(), automatic=True, switch_outcomes=[False]
    )
    assert result.rollback_count == 1


def test_d06_determinism() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    snaps = [_snap(0.0, 0.5), _snap(60.0, 0.97), _snap(120.0, 0.5)]
    a = run_simulation(policy, _options(), snaps, automatic=True)
    b = run_simulation(policy, _options(), snaps, automatic=True)
    assert a.switch_count == b.switch_count
    assert [d.action for d in a.decisions] == [d.action for d in b.decisions]


def test_d07_stale_telemetry_fails_safe() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    controller = Controller(policy, _options(), "p16")
    controller.recommend(_snap(100.0, 0.5))
    decision = controller.recommend(_snap(50.0, 0.99))  # older timestamp
    assert decision.action is ControllerAction.hold
    assert "stale" in decision.reason or "out-of-order" in decision.reason


def test_d07_missing_telemetry_fields_default_safely() -> None:
    # a snapshot with zero demand/throughput and healthy pressure must hold
    controller = Controller(ControllerPolicy(min_dwell_s=0.0), _options(), "p16")
    snapshot = TelemetrySnapshot(memory_pressure=0.5, evidence=EvidenceType.synthetic)
    assert controller.recommend(snapshot).action is ControllerAction.hold
