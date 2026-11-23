"""M3: adaptive runtime controller."""

from __future__ import annotations

import pytest

from adaptive.controller import (
    Controller,
    ControllerAction,
    ControllerPolicy,
    ProfileOption,
    run_simulation,
)
from adaptive.errors import ConfigurationError
from adaptive.models.controller import TelemetrySnapshot
from adaptive.models.enums import EvidenceType
from adaptive.telemetry import (
    ScriptedTelemetry,
    memory_pressure_scenario,
    steady_scenario,
    traffic_burst_scenario,
)


def _options() -> list[ProfileOption]:
    return [
        ProfileOption(
            profile_id="p4",
            precision_bits=4,
            estimated_cost=1.0,
            weight_bytes=4_000_000_000,
            quality_degradation=0.05,
        ),
        ProfileOption(
            profile_id="p8",
            precision_bits=8,
            estimated_cost=2.0,
            weight_bytes=8_000_000_000,
            quality_degradation=0.01,
        ),
        ProfileOption(
            profile_id="p16",
            precision_bits=16,
            estimated_cost=4.0,
            weight_bytes=16_000_000_000,
            quality_degradation=0.0,
        ),
    ]


def _snapshot(pressure: float, **kwargs: float) -> TelemetrySnapshot:
    base = dict(
        memory_pressure=pressure,
        memory_used_gb=pressure * 80.0,
        memory_total_gb=80.0,
        ttft_ms=120.0,
        itl_ms=20.0,
        tokens_per_second=200.0,
        model_demand=10.0,
        evidence=EvidenceType.synthetic,
    )
    base.update(kwargs)
    return TelemetrySnapshot(**base)


def test_memory_pressure_triggers_downshift() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0)
    controller = Controller(policy, _options(), "p16")
    decision = controller.recommend(_snapshot(0.97), now=0.0)
    assert decision.action is ControllerAction.downshift
    assert decision.recommended_profile_id == "p4"
    assert decision.switch_plan is not None
    assert decision.switch_plan.in_place is False
    assert decision.switch_plan.requires_replica_transition is True


def test_ttft_breach_triggers_downshift() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, ttft_slo_ms=500.0)
    controller = Controller(policy, _options(), "p16")
    decision = controller.recommend(_snapshot(0.5, ttft_ms=900.0), now=0.0)
    assert decision.action is ControllerAction.downshift


def test_healthy_headroom_triggers_upshift() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, quality_tolerance=0.05)
    controller = Controller(policy, _options(), "p4")
    decision = controller.recommend(_snapshot(0.3), now=0.0)
    assert decision.action is ControllerAction.upshift
    assert decision.recommended_profile_id == "p16"


def test_upshift_respects_quality_tolerance() -> None:
    options = _options()
    options[2] = options[2].model_copy(update={"quality_degradation": 0.5})
    policy = ControllerPolicy(min_dwell_s=0.0, quality_tolerance=0.05)
    controller = Controller(policy, options, "p4")
    decision = controller.recommend(_snapshot(0.3), now=0.0)
    assert decision.action is ControllerAction.upshift
    assert decision.recommended_profile_id == "p8"


def test_min_dwell_blocks_upshift() -> None:
    policy = ControllerPolicy(min_dwell_s=300.0)
    controller = Controller(policy, _options(), "p4")
    decision = controller.recommend(_snapshot(0.3), now=10.0)
    assert decision.action is ControllerAction.hold
    assert "dwell" in decision.reason
    assert decision.dwell_remaining_s > 0.0


def test_cooldown_blocks_switch_after_change() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=120.0)
    controller = Controller(policy, _options(), "p4")
    controller.last_change_ts = 0.0
    decision = controller.recommend(_snapshot(0.3), now=10.0)
    assert decision.action is ControllerAction.hold
    assert "cooldown" in decision.reason


def test_rate_limit_blocks_non_emergency_switch() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0, rate_limit_per_hour=1)
    controller = Controller(policy, _options(), "p16")
    controller.switch_timestamps.extend([0.0])
    decision = controller.recommend(_snapshot(0.5, ttft_ms=900.0), now=10.0)
    assert decision.action is ControllerAction.hold
    assert decision.rate_limited


def test_hysteresis_prevents_oscillation() -> None:
    # pressures oscillate around the high watermark within the hysteresis band
    pressures = [0.5, 0.92, 0.5, 0.92, 0.5, 0.92]
    snapshots = [_snapshot(p, timestamp_s=index * 60.0) for index, p in enumerate(pressures)]

    wide = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0, hysteresis=0.05)
    result_wide = run_simulation(wide, _options(), snapshots, automatic=True)
    assert result_wide.switch_count == 0
    assert result_wide.oscillated is False

    narrow = ControllerPolicy(
        min_dwell_s=0.0, cooldown_s=0.0, hysteresis=0.0, rate_limit_per_hour=100
    )
    result_narrow = run_simulation(narrow, _options(), snapshots, automatic=True)
    assert result_narrow.switch_count >= 2


def test_traffic_burst_response() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    result = run_simulation(
        policy, _options(), traffic_burst_scenario(), automatic=True
    )
    assert result.switch_count >= 1
    actions = [decision.action for decision in result.decisions]
    assert ControllerAction.downshift in actions
    assert result.oscillated is False


def test_safe_rollback_after_failed_upgrade() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    controller = Controller(policy, _options(), "p4")
    controller.record_switch("p16", now=0.0)
    decision = controller.evaluate_rollback(_snapshot(0.95), now=30.0)
    assert decision is not None
    assert decision.action is ControllerAction.rollback
    assert decision.recommended_profile_id == "p4"
    assert decision.rollback_plan is not None


def test_no_rollback_for_downshift_under_pressure() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    controller = Controller(policy, _options(), "p16")
    controller.record_switch("p4", now=0.0)
    assert controller.evaluate_rollback(_snapshot(0.95), now=30.0) is None


def test_no_rollback_outside_window() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0, rollback_window_s=60.0)
    controller = Controller(policy, _options(), "p4")
    controller.record_switch("p16", now=0.0)
    assert controller.evaluate_rollback(_snapshot(0.95), now=120.0) is None


def test_fault_injection_and_recovery() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    snapshots = memory_pressure_scenario()
    result = run_simulation(
        policy,
        _options(),
        snapshots,
        automatic=True,
        switch_outcomes=[False],
    )
    assert result.switch_count >= 1
    assert result.rollback_count == 1
    assert not result.oscillated


def test_recommendation_only_blocks_apply() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0)
    controller = Controller(policy, _options(), "p16", automatic=False)
    decision = controller.recommend(_snapshot(0.97), now=0.0)
    with pytest.raises(ConfigurationError):
        controller.apply(decision, now=0.0)


def test_enable_automatic_requires_validation() -> None:
    controller = Controller(ControllerPolicy(), _options(), "p16")
    with pytest.raises(ConfigurationError):
        controller.enable_automatic(validated=False)
    controller.enable_automatic(validated=True)
    assert controller.automatic and controller.validated


def test_decision_is_advisory_in_recommendation_mode() -> None:
    controller = Controller(ControllerPolicy(min_dwell_s=0.0), _options(), "p16")
    decision = controller.recommend(_snapshot(0.97), now=0.0)
    assert decision.automatic is False


def test_scripted_telemetry_iterates() -> None:
    source = ScriptedTelemetry(steady_scenario(steps=3))
    assert len(list(source)) == 3
    assert source.snapshot().memory_pressure == 0.5


def test_steady_scenario_holds() -> None:
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    result = run_simulation(policy, _options(), steady_scenario(), automatic=True)
    assert result.switch_count == 0
    assert all(decision.action is ControllerAction.hold for decision in result.decisions)
