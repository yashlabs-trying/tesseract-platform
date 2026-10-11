from __future__ import annotations
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from fusion.models.types import (
    GPUInventory, ModelDeployment, TenantPolicy, PerformanceProfile,
    WorkloadDemand, ForecastResult, DeploymentAlternative,
    ExpectedOutcome, AllocationStatus,
)
from fusion.demand.forecaster import DemandForecaster
from fusion.optimizer.candidate import CandidateGenerator
from fusion.optimizer.evaluator import AlternativeEvaluator
from fusion.policies.validator import ConstraintValidator
from fusion.economics.accounting import CostAccountingEngine


class DemandAwareOptimizer:
    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        self._forecaster = DemandForecaster()
        self._candidate_generator = CandidateGenerator()
        self._evaluator = AlternativeEvaluator()
        self._constraint_validator = ConstraintValidator()
        self._cost_engine = CostAccountingEngine()

        self._hysteresis_threshold = float(
            self._config.get("hysteresis_threshold", 0.15)
        )
        self._min_decision_interval = timedelta(
            minutes=self._config.get("min_decision_interval_minutes", 15)
        )
        self._warm_cost_factor = float(
            self._config.get("warm_cost_factor", 0.8)
        )
        self._cold_start_seconds = float(
            self._config.get("cold_start_seconds", 30.0)
        )

        self._last_decisions: Dict[str, Tuple[int, datetime]] = {}

    def evaluate_with_demand(
        self,
        deployment: ModelDeployment,
        gpus: List[GPUInventory],
        tenant_policy: TenantPolicy,
        profile: Optional[PerformanceProfile],
        demand: Optional[WorkloadDemand],
        demand_history: List[WorkloadDemand],
    ) -> Tuple[Optional[DeploymentAlternative], Optional[ExpectedOutcome], str]:
        gpu_map = {g.gpu_id: g for g in gpus}

        forecast = self._forecaster.forecast(demand_history, horizon_hours=1)

        alternatives = self._candidate_generator.generate_alternatives(
            deployment, gpus, profile,
            max_candidates=self._config.get("max_candidates_per_workload", 15),
        )

        cold_alts = []
        for alt in alternatives:
            if alt.replica_count > 0:
                cold_alt = alt.model_copy(update={
                    "warm_policy": "cold",
                    "cold_start_risk": f"~{self._cold_start_seconds}s startup delay",
                    "estimated_cost_per_hour": alt.estimated_cost_per_hour * Decimal(str(self._warm_cost_factor)),
                })
                cold_alts.append(cold_alt)

        alternatives.extend(cold_alts)

        best_alt: Optional[DeploymentAlternative] = None
        best_outcome: Optional[ExpectedOutcome] = None
        reason = "no_feasible_alternative"

        for alt in alternatives:
            gpu = gpu_map.get(alt.gpu_pool_id)
            if gpu is None and alt.gpu_pool_id != "__baseline__":
                continue

            dep_id = deployment.deployment_id
            if dep_id in self._last_decisions:
                last_count, last_time = self._last_decisions[dep_id]
                if (datetime.utcnow() - last_time) < self._min_decision_interval:
                    if abs(alt.replica_count - last_count) / max(last_count, 1) < self._hysteresis_threshold:
                        continue

            if forecast and alt.warm_policy == "cold":
                peak_forecast = max(forecast.forecast_rps) if forecast.forecast_rps else 0
                current_throughput = (profile.throughput_rps if profile else 10.0) * alt.replica_count
                if peak_forecast * 1.2 < current_throughput and alt.replica_count > 1:
                    continue

            outcome = self._evaluator.evaluate(alt, gpu, deployment, tenant_policy, profile)

            if forecast and outcome.expected_contribution_margin is not None:
                uncertainty = max(forecast.forecast_uncertainty) if forecast.forecast_uncertainty else 0
                risk_penalty = Decimal(str(uncertainty * 0.1)) * outcome.expected_operating_cost
                adjusted = outcome.model_copy(update={
                    "expected_contribution_margin": outcome.expected_contribution_margin - risk_penalty,
                    "modeled_risk_adjustment": outcome.modeled_risk_adjustment + risk_penalty,
                })
                outcome = adjusted

            if best_alt is None or (outcome.expected_contribution_margin or Decimal("-inf")) > (best_outcome.expected_contribution_margin or Decimal("-inf")):
                best_alt = alt
                best_outcome = outcome
                reason = "demand_aware_selection"

        if best_alt and best_outcome and best_alt.gpu_pool_id != "__baseline__":
            self._last_decisions[deployment.deployment_id] = (
                best_alt.replica_count,
                datetime.utcnow(),
            )

        return best_alt, best_outcome, reason

    def should_scale_to_zero(
        self,
        demand: Optional[WorkloadDemand],
        forecast: Optional[ForecastResult],
        warm_cost: Decimal,
        cold_start_cost: Decimal,
        idle_hours: float = 1.0,
    ) -> Tuple[bool, str]:
        if demand is None or demand.requests_per_second == 0:
            projected_idle_cost = warm_cost * Decimal(str(idle_hours))
            if projected_idle_cost > cold_start_cost:
                return True, (
                    f"Idle cost ${projected_idle_cost:.2f} exceeds "
                    f"reactivation cost ${cold_start_cost:.2f} — scale to zero"
                )
            return False, (
                f"Reactivation cost ${cold_start_cost:.2f} exceeds "
                f"idle cost ${projected_idle_cost:.2f} — keep warm"
            )
        if forecast and forecast.burst_probability > 0.3:
            return False, "High burst probability — retain warm capacity"
        return False, "Active demand — keep serving"
