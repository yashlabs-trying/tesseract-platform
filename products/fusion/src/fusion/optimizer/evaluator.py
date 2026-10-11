from __future__ import annotations
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from fusion.models.types import (
    DeploymentAlternative, GPUInventory, ModelDeployment, TenantPolicy,
    PerformanceProfile, ExpectedOutcome, ConstraintResult, FusionRecommendation,
    OptimizationScenario,
)
from fusion.policies.validator import PolicyValidator
from fusion.economics.accounting import CostAccountingEngine


class AlternativeEvaluator:
    def __init__(self):
        self._policy_validator = PolicyValidator()
        self._cost_engine = CostAccountingEngine()

    def evaluate(
        self,
        alternative: DeploymentAlternative,
        gpu: Optional[GPUInventory],
        deployment: ModelDeployment,
        tenant_policy: TenantPolicy,
        profile: Optional[PerformanceProfile] = None,
    ) -> ExpectedOutcome:
        cost = alternative.estimated_cost_per_hour
        revenue = alternative.estimated_revenue_per_hour or Decimal("0")
        risk = Decimal("0")
        uncertainty = None
        slo_violation = None

        # Risk adjustment: no profile means higher uncertainty
        if profile is None or profile.evidence_type == "unknown":
            risk = cost * Decimal("0.2")
            slo_violation = 0.10
            uncertainty = "No performance profile — risk adjustment applied"
        else:
            slo_violation = 0.01

        # Contribution margin = revenue - cost
        contribution = revenue - cost

        return ExpectedOutcome(
            expected_revenue=revenue,
            expected_operating_cost=cost,
            expected_contribution_margin=contribution,
            modeled_risk_adjustment=risk,
            expected_p95_latency_ms=profile.ttft_p50_ms if profile else None,
            expected_p99_latency_ms=profile.ttft_p99_ms if profile else None,
            uncertainty_description=uncertainty,
            slo_violation_probability=slo_violation,
        )

    def rank_alternatives(
        self,
        alternatives: List[DeploymentAlternative],
        gpu_map: Dict[str, GPUInventory],
        deployment: ModelDeployment,
        tenant_policy: TenantPolicy,
        profiles: Dict[str, PerformanceProfile],
    ) -> List[Tuple[DeploymentAlternative, ExpectedOutcome, List]]:
        results = []
        for alt in alternatives:
            gpu = gpu_map.get(alt.gpu_pool_id)
            profile = profiles.get(alt.deployment_id)
            outcome = self.evaluate(alt, gpu, deployment, tenant_policy, profile)
            results.append((alt, outcome, []))
        results.sort(key=lambda x: x[1].expected_contribution_margin or Decimal("-inf"), reverse=True)
        return results
