"""Adaptive runtime controller (Milestone 3).

Consumes telemetry (memory pressure, queue depth, TTFT, ITL, throughput,
demand) and quality certificates, and emits a recommended precision profile
with projected memory/cost impact, a switching plan and a rollback plan.

Anti-oscillation controls: hysteresis, minimum dwell time, cooldown after a
change, and a per-hour rate limit. The controller is recommendation-only until
:meth:`Controller.enable_automatic` is called after validation; it never
mutates live weights and always plans a replica-based transition.
"""

from __future__ import annotations

from collections.abc import Sequence

from adaptive.errors import ConfigurationError
from adaptive.models.controller import (
    ControlDecision,
    ControllerAction,
    ControllerPolicy,
    ProfileOption,
    RollbackPlan,
    SwitchPlan,
    TelemetrySnapshot,
)


class Controller:
    """Stateful precision controller."""

    def __init__(
        self,
        policy: ControllerPolicy,
        options: Sequence[ProfileOption],
        current_profile_id: str,
        *,
        automatic: bool = False,
    ) -> None:
        if not options:
            raise ConfigurationError("controller requires at least one profile option")
        self.policy = policy
        self.options = {option.profile_id: option for option in options}
        if current_profile_id not in self.options:
            raise ConfigurationError(f"unknown current profile: {current_profile_id}")
        self.current_profile_id = current_profile_id
        self.current_since = 0.0
        self.last_change_ts: float | None = None
        self.last_seen_ts: float | None = None
        self.switch_timestamps: list[float] = []
        self.previous_profile_id: str | None = None
        self.automatic = automatic
        self.validated = False

    # -- helpers ----------------------------------------------------------
    def _option(self, profile_id: str) -> ProfileOption:
        return self.options[profile_id]

    def _downshift_target(self, current: ProfileOption) -> ProfileOption | None:
        cheaper = [
            option
            for option in self.options.values()
            if option.precision_bits < current.precision_bits
        ]
        if not cheaper:
            return None
        return min(cheaper, key=lambda option: (option.estimated_cost, option.precision_bits))

    def _upshift_target(self, current: ProfileOption) -> ProfileOption | None:
        eligible = [
            option
            for option in self.options.values()
            if option.precision_bits > current.precision_bits
            and option.quality_degradation <= self.policy.quality_tolerance
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda option: (option.precision_bits, -option.estimated_cost))

    def _rate_limited(self, now: float) -> bool:
        window_start = now - 3600.0
        recent = [ts for ts in self.switch_timestamps if ts > window_start]
        return len(recent) >= self.policy.rate_limit_per_hour

    def _dwell_remaining(self, now: float) -> float:
        return max(0.0, self.policy.min_dwell_s - (now - self.current_since))

    def _cooldown_remaining(self, now: float) -> float:
        if self.last_change_ts is None:
            return 0.0
        return max(0.0, self.policy.cooldown_s - (now - self.last_change_ts))

    # -- decision ---------------------------------------------------------
    def recommend(self, snapshot: TelemetrySnapshot, now: float | None = None) -> ControlDecision:
        now = snapshot.timestamp_s if now is None else now
        current = self._option(self.current_profile_id)
        automatic = self.automatic and self.validated

        # fail safe on stale / out-of-order telemetry: never switch on a
        # snapshot older than the last one observed
        if self.last_seen_ts is not None and now < self.last_seen_ts:
            return self._hold(current, "stale or out-of-order telemetry", now, automatic)
        self.last_seen_ts = now

        pressure = snapshot.memory_pressure
        ttft_bad = snapshot.ttft_ms > self.policy.ttft_slo_ms
        itl_bad = snapshot.itl_ms > self.policy.itl_slo_ms
        tps_bad = (
            self.policy.min_throughput_tps > 0.0
            and snapshot.tokens_per_second < self.policy.min_throughput_tps
        )
        emergency = pressure >= self.policy.memory_high + 2.0 * self.policy.hysteresis
        down = (
            pressure >= self.policy.memory_high + self.policy.hysteresis
            or ttft_bad
            or itl_bad
            or tps_bad
        )
        up = (
            pressure <= self.policy.memory_low - self.policy.hysteresis
            and not ttft_bad
            and not itl_bad
            and not tps_bad
        )

        rollback = self.evaluate_rollback(snapshot, now)
        if rollback is not None:
            return rollback

        dwell_remaining = self._dwell_remaining(now)
        cooldown_remaining = self._cooldown_remaining(now)

        if down:
            target = self._downshift_target(current)
            if target is None:
                return self._hold(current, "no lower-precision profile available", now, automatic)
            if dwell_remaining > 0.0 and not emergency:
                return self._hold(current, "minimum dwell time not elapsed", now, automatic)
            if cooldown_remaining > 0.0 and not emergency:
                return self._hold(current, "cooldown active", now, automatic)
            if self._rate_limited(now) and not emergency:
                return self._hold(
                    current, "switch rate limit reached", now, automatic, rate_limited=True
                )
            return self._switch(current, target, "pressure or SLO breach", now, automatic)

        if up:
            target = self._upshift_target(current)
            if target is None:
                return self._hold(
                    current,
                    "no eligible higher-precision profile within quality tolerance",
                    now,
                    automatic,
                )
            if dwell_remaining > 0.0:
                return self._hold(current, "minimum dwell time not elapsed", now, automatic)
            if cooldown_remaining > 0.0:
                return self._hold(current, "cooldown active", now, automatic)
            if self._rate_limited(now):
                return self._hold(
                    current, "switch rate limit reached", now, automatic, rate_limited=True
                )
            return self._switch(
                current, target, "headroom available and quality allows upgrade", now, automatic
            )

        return self._hold(current, "stable", now, automatic)

    def evaluate_rollback(
        self, snapshot: TelemetrySnapshot, now: float | None = None
    ) -> ControlDecision | None:
        """Recommend a rollback if a recent switch is failing."""
        now = snapshot.timestamp_s if now is None else now
        if self.previous_profile_id is None or self.last_change_ts is None:
            return None
        if now - self.last_change_ts > self.policy.rollback_window_s:
            return None
        current = self._option(self.current_profile_id)
        previous = self._option(self.previous_profile_id)
        # only roll back a recent *upgrade* that is now breaching; a downshift
        # under pressure is already the safer direction.
        if previous.precision_bits >= current.precision_bits:
            return None
        failing = (
            snapshot.memory_pressure >= self.policy.memory_high
            or snapshot.ttft_ms > self.policy.ttft_slo_ms
            or snapshot.itl_ms > self.policy.itl_slo_ms
        )
        if not failing:
            return None
        return ControlDecision(
            action=ControllerAction.rollback,
            current_profile_id=current.profile_id,
            recommended_profile_id=previous.profile_id,
            reason="recent switch is failing validation",
            automatic=self.automatic and self.validated,
            expected_memory_gb=previous.weight_bytes / 1e9,
            expected_cost=previous.estimated_cost,
            expected_quality_degradation=previous.quality_degradation,
            rollback_plan=RollbackPlan(
                from_profile_id=current.profile_id,
                to_profile_id=previous.profile_id,
                triggers=("memory pressure", "ttft breach", "itl breach"),
            ),
            evidence=snapshot.evidence,
        )

    # -- decision builders ------------------------------------------------
    def _hold(
        self,
        current: ProfileOption,
        reason: str,
        now: float,
        automatic: bool,
        *,
        rate_limited: bool = False,
    ) -> ControlDecision:
        return ControlDecision(
            action=ControllerAction.hold,
            current_profile_id=current.profile_id,
            recommended_profile_id=current.profile_id,
            reason=reason,
            automatic=automatic,
            rate_limited=rate_limited,
            dwell_remaining_s=self._dwell_remaining(now),
            expected_memory_gb=current.weight_bytes / 1e9,
            expected_cost=current.estimated_cost,
            expected_quality_degradation=current.quality_degradation,
        )

    def _switch(
        self,
        current: ProfileOption,
        target: ProfileOption,
        reason: str,
        now: float,
        automatic: bool,
    ) -> ControlDecision:
        action = (
            ControllerAction.downshift
            if target.precision_bits < current.precision_bits
            else ControllerAction.upshift
        )
        return ControlDecision(
            action=action,
            current_profile_id=current.profile_id,
            recommended_profile_id=target.profile_id,
            reason=reason,
            automatic=automatic,
            dwell_remaining_s=0.0,
            expected_memory_gb=target.weight_bytes / 1e9,
            expected_cost=target.estimated_cost,
            expected_quality_degradation=target.quality_degradation,
            switch_plan=SwitchPlan(
                from_profile_id=current.profile_id,
                to_profile_id=target.profile_id,
                estimated_downtime_s=0.0,
                requires_replica_transition=True,
                in_place=False,
                reason=reason,
            ),
            rollback_plan=RollbackPlan(
                from_profile_id=target.profile_id,
                to_profile_id=current.profile_id,
                triggers=("quality gate failed", "slo breach", "readiness timeout"),
            ),
        )

    # -- state transitions ------------------------------------------------
    def enable_automatic(self, *, validated: bool = True) -> None:
        """Enable automatic switching (only after validation passes)."""
        if not validated:
            raise ConfigurationError("automatic switching requires validation")
        self.automatic = True
        self.validated = True

    def apply(
        self, decision: ControlDecision, now: float, *, success: bool = True
    ) -> None:
        """Apply a decision; requires automatic mode."""
        if not (self.automatic and self.validated):
            raise ConfigurationError("automatic switching is disabled")
        if decision.action in (ControllerAction.downshift, ControllerAction.upshift):
            self.record_switch(decision.recommended_profile_id, now, success=success)
        elif decision.action is ControllerAction.rollback:
            self.record_rollback(now)

    def record_switch(self, profile_id: str, now: float, *, success: bool = True) -> None:
        if profile_id not in self.options:
            raise ConfigurationError(f"unknown profile: {profile_id}")
        if profile_id == self.current_profile_id:
            return  # idempotent: duplicate transition request is a no-op
        self.last_change_ts = now
        self.switch_timestamps.append(now)
        if not success:
            return
        self.previous_profile_id = self.current_profile_id
        self.current_profile_id = profile_id
        self.current_since = now

    def record_rollback(self, now: float) -> None:
        if self.previous_profile_id is None:
            return
        self.current_profile_id = self.previous_profile_id
        self.previous_profile_id = None
        self.last_change_ts = now
        self.current_since = now
        self.switch_timestamps.append(now)


__all__ = ["Controller"]
