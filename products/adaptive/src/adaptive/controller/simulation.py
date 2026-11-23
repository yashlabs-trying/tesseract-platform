"""Controller simulation and fault injection (Milestone 3)."""

from __future__ import annotations

from collections.abc import Sequence

from adaptive.controller.controller import Controller
from adaptive.models.controller import (
    ControlDecision,
    ControllerAction,
    ControllerPolicy,
    ProfileOption,
    SimulationResult,
    TelemetrySnapshot,
)
from adaptive.models.enums import EvidenceType


def _initial_profile(options: Sequence[ProfileOption], preferred: str | None) -> str:
    if preferred is not None:
        return preferred
    return max(options, key=lambda option: option.precision_bits).profile_id


def run_simulation(
    policy: ControllerPolicy,
    options: Sequence[ProfileOption],
    snapshots: Sequence[TelemetrySnapshot],
    *,
    initial_profile_id: str | None = None,
    automatic: bool = False,
    switch_outcomes: Sequence[bool] | None = None,
) -> SimulationResult:
    """Run the controller over a telemetry timeline.

    ``switch_outcomes`` injects faults: a ``False`` entry means the switch was
    applied but failed verification, triggering a rollback.
    """
    controller = Controller(
        policy, options, _initial_profile(options, initial_profile_id), automatic=False
    )
    if automatic:
        controller.enable_automatic()

    outcomes = list(switch_outcomes) if switch_outcomes is not None else []
    outcome_index = 0
    decisions: list[ControlDecision] = []
    switch_count = 0
    rollback_count = 0
    direction_changes = 0
    last_direction: str | None = None

    for snapshot in snapshots:
        now = snapshot.timestamp_s
        decision = controller.recommend(snapshot, now)
        decisions.append(decision)
        if not automatic:
            continue

        if decision.action is ControllerAction.rollback:
            controller.record_rollback(now)
            rollback_count += 1
            continue
        if decision.action in (ControllerAction.downshift, ControllerAction.upshift):
            success = outcomes[outcome_index] if outcome_index < len(outcomes) else True
            outcome_index += 1
            controller.record_switch(decision.recommended_profile_id, now, success=True)
            switch_count += 1
            direction = "down" if decision.action is ControllerAction.downshift else "up"
            if last_direction is not None and direction != last_direction:
                direction_changes += 1
            last_direction = direction
            if not success:
                controller.record_rollback(now)
                rollback_count += 1
                decisions.append(
                    ControlDecision(
                        action=ControllerAction.rollback,
                        current_profile_id=decision.recommended_profile_id,
                        recommended_profile_id=decision.current_profile_id,
                        reason="switch verification failed; rolling back",
                        automatic=True,
                        rollback_plan=None,
                        evidence=EvidenceType.synthetic,
                    )
                )

    oscillated = switch_count > policy.rate_limit_per_hour
    return SimulationResult(
        decisions=tuple(decisions),
        switch_count=switch_count,
        rollback_count=rollback_count,
        direction_changes=direction_changes,
        oscillated=oscillated,
        final_profile_id=controller.current_profile_id,
        evidence=EvidenceType.synthetic,
    )


__all__ = ["run_simulation"]
