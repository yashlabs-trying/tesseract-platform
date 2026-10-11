from __future__ import annotations
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from fusion.models.types import (
    GPUInventory, ModelDeployment, TenantPolicy, PerformanceProfile,
    OptimizationScenario, FusionRecommendation, DeploymentAlternative,
    ExpectedOutcome, AllocationStatus, RecommendationStatus,
    WorkloadDemand,
)
from fusion.optimizer.candidate import CandidateGenerator
from fusion.optimizer.evaluator import AlternativeEvaluator
from fusion.policies.validator import PolicyValidator, ConstraintValidator
from fusion.economics.accounting import CostAccountingEngine, RevenueCalculator


class OptimizationEngine:
    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        self._candidate_generator = CandidateGenerator()
        self._evaluator = AlternativeEvaluator()
        self._policy_validator = PolicyValidator()
        self._constraint_validator = ConstraintValidator()
        self._cost_engine = CostAccountingEngine()
        self._revenue_calc = RevenueCalculator()

    def optimize(
        self,
        deployments: List[ModelDeployment],
        gpus: List[GPUInventory],
        tenant_policies: Dict[str, TenantPolicy],
        profiles: Dict[str, PerformanceProfile],
        demands: Dict[str, WorkloadDemand],
        scenario: Optional[OptimizationScenario] = None,
    ) -> FusionRecommendation:
        gpu_map = {g.gpu_id: g for g in gpus}
        gpu_map.update({g.gpu_model: g for g in gpus})
        gpu_map.update({g.cluster_id: g for g in gpus})

        recommended_plan: List[DeploymentAlternative] = []
        baseline_plan: List[DeploymentAlternative] = []
        alternative_lists: list = []
        rejected: List[str] = []
        rejected_reasons: Dict[str, str] = {}
        all_constraint_checks: Dict[str, str] = {}
        total_baseline_revenue = Decimal("0")
        total_baseline_cost = Decimal("0")
        total_rec_revenue = Decimal("0")
        total_rec_cost = Decimal("0")

        for dep in deployments:
            policy = tenant_policies.get(dep.tenant_id, TenantPolicy(tenant_id=dep.tenant_id))
            profile = profiles.get(dep.deployment_id)
            demand = demands.get(dep.deployment_id)

            alternatives = self._candidate_generator.generate_alternatives(
                dep, gpus, profile,
                max_candidates=self._config.get("max_candidates_per_workload", 20),
            )

            validated_alts: List[Tuple[DeploymentAlternative, ExpectedOutcome]] = []
            for alt in alternatives:
                gpu = gpu_map.get(alt.gpu_pool_id) or gpu_map.get(alt.gpu_type)
                checks = []
                if gpu:
                    checks = self._constraint_validator.validate_all(alt, gpu, dep, policy, profile)
                failures = [c for c in checks if c.result == "fail"]
                if not failures:
                    outcome = self._evaluator.evaluate(alt, gpu, dep, policy, profile)
                    validated_alts.append((alt, outcome))
                else:
                    rejected.append(alt.gpu_pool_id)
                    rejected_reasons[alt.gpu_pool_id] = "; ".join(
                        c.details or c.name for c in failures
                    )

            if not validated_alts:
                continue

            validated_alts.sort(key=lambda x: x[1].expected_contribution_margin or Decimal("-inf"), reverse=True)

            # Baseline = first alternative (usually current allocation)
            baseline = validated_alts[0][1] if validated_alts else ExpectedOutcome()
            baseline_alt = validated_alts[0][0] if validated_alts else None
            if baseline_alt:
                baseline_plan.append(baseline_alt)
                total_baseline_revenue += baseline.expected_revenue
                total_baseline_cost += baseline.expected_operating_cost

            # Recommended = best feasible alternative
            recommended = validated_alts[0][1] if validated_alts else ExpectedOutcome()
            rec_alt = validated_alts[0][0] if validated_alts else None
            if rec_alt:
                rec_alt = rec_alt.model_copy(update={"allocation_status": AllocationStatus.RECOMMENDED})
                recommended_plan.append(rec_alt)
                total_rec_revenue += recommended.expected_revenue
                total_rec_cost += recommended.expected_operating_cost

            alt_list = [a for a, _ in validated_alts[1:]]
            alternative_lists.append(alt_list)

        baseline_econ = ExpectedOutcome(
            expected_revenue=total_baseline_revenue,
            expected_operating_cost=total_baseline_cost,
        )
        rec_econ = ExpectedOutcome(
            expected_revenue=total_rec_revenue,
            expected_operating_cost=total_rec_cost,
        )

        return FusionRecommendation(
            scenario_id=scenario.scenario_id if scenario else "default",
            recommended_plan=recommended_plan,
            baseline_plan=baseline_plan,
            alternative_plans=alternative_lists,
            expected_economics=rec_econ,
            baseline_economics=baseline_econ,
            confidence=0.7 if all(demands.values()) else 0.4,
            rejected_alternatives=rejected,
            rejected_reasons=rejected_reasons,
            approval_status=RecommendationStatus.ADVISORY_ONLY,
            execution_eligibility=False,
            provenance={"optimizer_version": "0.1.0", "engine": "deterministic_candidate"},
        )
