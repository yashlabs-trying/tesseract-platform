"""Optimization controller.

Compares measured performance against desired objectives and, depending on
mode, may recommend or (only when explicitly enabled) drive reconfiguration.

Safety features required by the platform:

* **observe-only by default** — never acts;
* **hysteresis** — a violation must exceed a threshold before it counts;
* **cooldowns** — at most one action per workload per cooldown window;
* **action budget** — a hard cap on actions, default 0.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, Field

from tesseract_platform.config import ControllerMode
from tesseract_platform.models import SLOSpec
from tesseract_platform.observability.telemetry import TelemetryCollector, TelemetrySummary


class ControllerDecision(BaseModel):
    workload_id: str
    mode: ControllerMode
    action: str = "none"  # none | recommend_reconfigure | reconfigure
    reason: str = ""
    current_ttft_p95_ms: Optional[float] = None
    target_ttft_p95_ms: Optional[float] = None
    violation_pct: Optional[float] = None
    cooldown_active: bool = False
    budget_remaining: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OptimizationController:
    def __init__(
        self,
        store,
        telemetry: TelemetryCollector,
        *,
        mode: ControllerMode = ControllerMode.OBSERVE_ONLY,
        cooldown_seconds: int = 120,
        action_budget: int = 0,
        ttft_hysteresis_pct: float = 15.0,
    ) -> None:
        self.store = store
        self.telemetry = telemetry
        self.mode = mode
        self.cooldown_seconds = cooldown_seconds
        self.action_budget = action_budget
        self.ttft_hysteresis_pct = ttft_hysteresis_pct
        self._last_action_at: dict[str, datetime] = {}
        self._actions_used = 0

    @property
    def budget_remaining(self) -> int:
        return max(0, self.action_budget - self._actions_used)

    def evaluate(
        self,
        workload_id: str,
        slo: SLOSpec,
        *,
        now: Optional[datetime] = None,
    ) -> ControllerDecision:
        now = now or datetime.now(timezone.utc)
        summary: TelemetrySummary = self.telemetry.aggregate(workload_id)
        decision = ControllerDecision(
            workload_id=workload_id,
            mode=self.mode,
            budget_remaining=self.budget_remaining,
            target_ttft_p95_ms=slo.ttft_p95_ms,
            current_ttft_p95_ms=summary.ttft_p95_ms,
        )
        if summary.request_count == 0 or summary.ttft_p95_ms is None:
            decision.reason = "no_successful_requests"
            return decision
        if slo.ttft_p95_ms is None:
            decision.reason = "no_ttft_objective"
            return decision

        target = slo.ttft_p95_ms
        current = summary.ttft_p95_ms
        violation_pct = ((current - target) / target * 100.0) if target > 0 else 0.0
        decision.violation_pct = violation_pct

        if violation_pct <= self.ttft_hysteresis_pct:
            decision.reason = "within_hysteresis"
            return decision

        decision.reason = "ttft_p95_exceeds_objective"
        if self.mode is ControllerMode.OBSERVE_ONLY:
            decision.action = "none"
            return decision

        decision.action = "recommend_reconfigure"
        if self.mode is ControllerMode.CONTROLLED_ACTION:
            if self._cooldown_active(workload_id, now):
                decision.cooldown_active = True
                decision.reason = "cooldown_active"
            elif self.budget_remaining <= 0:
                decision.reason = "action_budget_exhausted"
            else:
                decision.action = "reconfigure"
                self._record_action(workload_id, now)
                decision.budget_remaining = self.budget_remaining
        self.store.record_audit(
            None, "controller_decision", decision.model_dump(mode="json")
        )
        return decision

    def _cooldown_active(self, workload_id: str, now: datetime) -> bool:
        last = self._last_action_at.get(workload_id)
        if last is None:
            return False
        return now - last < timedelta(seconds=self.cooldown_seconds)

    def _record_action(self, workload_id: str, now: datetime) -> None:
        self._last_action_at[workload_id] = now
        self._actions_used += 1
