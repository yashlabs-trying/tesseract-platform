from __future__ import annotations
from decimal import Decimal
from itertools import product
from typing import Dict, List, Optional, Tuple, Any
from fusion.models.types import (
    GPUInventory, ModelDeployment, TenantPolicy, PerformanceProfile,
    DeploymentAlternative, ExpectedOutcome, AllocationStatus,
    WorkloadDemand, FusionRecommendation, RecommendationStatus,
)
from fusion.optimizer.candidate import CandidateGenerator
from fusion.optimizer.evaluator import AlternativeEvaluator
from fusion.policies.validator import ConstraintValidator


def _has_ortools() -> bool:
    try:
        import ortools
        return True
    except ImportError:
        return False


class GlobalAllocator:
    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        self._candidate_generator = CandidateGenerator()
        self._evaluator = AlternativeEvaluator()
        self._constraint_validator = ConstraintValidator()
        self._max_combinations = self._config.get("max_exhaustive_combinations", 10000)

    def allocate(
        self,
        deployments: List[ModelDeployment],
        gpus: List[GPUInventory],
        tenant_policies: Dict[str, TenantPolicy],
        profiles: Dict[str, PerformanceProfile],
        demands: Dict[str, WorkloadDemand],
    ) -> FusionRecommendation:
        gpu_map = {g.gpu_id: g for g in gpus}

        # Generate all alternatives for each deployment
        all_alts: Dict[str, List[DeploymentAlternative]] = {}
        for dep in deployments:
            policy = tenant_policies.get(dep.tenant_id, TenantPolicy(tenant_id=dep.tenant_id))
            profile = profiles.get(dep.deployment_id)
            alts = self._candidate_generator.generate_alternatives(
                dep, gpus, profile,
                max_candidates=self._config.get("max_candidates_per_workload", 10),
            )
            all_alts[dep.deployment_id] = alts

        # Build baseline plan
        baseline_plan: List[DeploymentAlternative] = []
        for dep in deployments:
            for alt in all_alts.get(dep.deployment_id, []):
                if alt.gpu_pool_id == "__baseline__":
                    baseline_plan.append(alt)
                    break

        if len(deployments) <= 9:
            return self._exhaustive_search(
                deployments, gpus, gpu_map, tenant_policies, profiles,
                all_alts, baseline_plan,
            )
        elif _has_ortools():
            return self._ortools_solve(
                deployments, gpus, gpu_map, tenant_policies, profiles,
                all_alts, baseline_plan,
            )
        else:
            return self._greedy_allocate(
                deployments, gpus, gpu_map, tenant_policies, profiles,
                all_alts, baseline_plan,
            )

    def _exhaustive_search(
        self,
        deployments: List[ModelDeployment],
        gpus: List[GPUInventory],
        gpu_map: Dict[str, GPUInventory],
        tenant_policies: Dict[str, TenantPolicy],
        profiles: Dict[str, PerformanceProfile],
        all_alts: Dict[str, List[DeploymentAlternative]],
        baseline_plan: List[DeploymentAlternative],
    ) -> FusionRecommendation:
        alt_lists = [all_alts.get(d.deployment_id, []) for d in deployments]
        # Filter out empty lists
        alt_lists = [l for l in alt_lists if l]

        if not alt_lists:
            return self._make_empty_recommendation()

        best_combination: Optional[Tuple[List[DeploymentAlternative], List[ExpectedOutcome]]] = None
        best_score = Decimal("-inf")
        combinations_tried = 0

        for combo in product(*alt_lists):
            combinations_tried += 1
            if combinations_tried > self._max_combinations:
                break

            feasible, outcomes = self._check_global_feasibility(
                list(combo), deployments, gpus, gpu_map, tenant_policies, profiles
            )
            if not feasible:
                continue

            total_contribution = sum(
                (o.expected_contribution_margin or Decimal("0")) for o in outcomes
            )
            if total_contribution > best_score:
                best_score = total_contribution
                best_combination = (list(combo), outcomes)

        if best_combination is None:
            return self._make_empty_recommendation()

        recommended_plan, outcomes = best_combination
        for alt in recommended_plan:
            alt = alt.model_copy(update={"allocation_status": AllocationStatus.RECOMMENDED})

        total_rev = sum(o.expected_revenue for o in outcomes)
        total_cost = sum(o.expected_operating_cost for o in outcomes)

        baseline_outcomes = []
        for alt in baseline_plan:
            dep = next(d for d in deployments if d.deployment_id == alt.deployment_id)
            policy = tenant_policies.get(dep.tenant_id, TenantPolicy(tenant_id=dep.tenant_id))
            profile = profiles.get(dep.deployment_id)
            baseline_outcomes.append(self._evaluator.evaluate(alt, None, dep, policy, profile))

        baseline_rev = sum(o.expected_revenue for o in baseline_outcomes)
        baseline_cost = sum(o.expected_operating_cost for o in baseline_outcomes)

        return FusionRecommendation(
            scenario_id="global_allocator",
            recommended_plan=recommended_plan,
            baseline_plan=baseline_plan,
            expected_economics=ExpectedOutcome(
                expected_revenue=total_rev,
                expected_operating_cost=total_cost,
                expected_contribution_margin=total_rev - total_cost,
            ),
            baseline_economics=ExpectedOutcome(
                expected_revenue=baseline_rev,
                expected_operating_cost=baseline_cost,
                expected_contribution_margin=baseline_rev - baseline_cost,
            ),
            confidence=0.6,
            approval_status=RecommendationStatus.ADVISORY_ONLY,
            execution_eligibility=False,
            provenance={"optimizer_version": "0.2.0", "engine": "exhaustive_global"},
        )

    def _greedy_allocate(
        self,
        deployments: List[ModelDeployment],
        gpus: List[GPUInventory],
        gpu_map: Dict[str, GPUInventory],
        tenant_policies: Dict[str, TenantPolicy],
        profiles: Dict[str, PerformanceProfile],
        all_alts: Dict[str, List[DeploymentAlternative]],
        baseline_plan: List[DeploymentAlternative],
    ) -> FusionRecommendation:
        """Greedy: assign workloads to GPUs sorted by priority then best contribution."""
        gpu_slots = {g.gpu_id: g.available_resource_slots for g in gpus}
        recommended_plan = []

        sorted_deps = sorted(
            deployments,
            key=lambda d: tenant_policies.get(d.tenant_id, TenantPolicy(tenant_id=d.tenant_id)).priority,
            reverse=True,
        )

        for dep in sorted_deps:
            alts = all_alts.get(dep.deployment_id, [])
            best_alt = None
            best_outcome = None

            for alt in alts:
                if alt.gpu_pool_id == "__baseline__":
                    continue
                gpu = gpu_map.get(alt.gpu_pool_id)
                if gpu is None:
                    continue
                if gpu_slots.get(gpu.gpu_id, 0) < alt.replica_count:
                    continue

                policy = tenant_policies.get(dep.tenant_id, TenantPolicy(tenant_id=dep.tenant_id))
                profile = profiles.get(dep.deployment_id)
                outcome = self._evaluator.evaluate(alt, gpu, dep, policy, profile)

                if best_alt is None or (outcome.expected_contribution_margin or Decimal("-inf")) > (best_outcome.expected_contribution_margin or Decimal("-inf")):
                    best_alt = alt
                    best_outcome = outcome

            if best_alt and best_outcome:
                gpu_slots[best_alt.gpu_pool_id] = gpu_slots.get(best_alt.gpu_pool_id, 0) - best_alt.replica_count
                recommended_plan.append(best_alt.model_copy(update={"allocation_status": AllocationStatus.RECOMMENDED}))

        rec_rev = sum(a.estimated_revenue_per_hour or Decimal("0") for a in recommended_plan)
        rec_cost = sum(a.estimated_cost_per_hour for a in recommended_plan)

        return FusionRecommendation(
            scenario_id="global_allocator_greedy",
            recommended_plan=recommended_plan,
            baseline_plan=baseline_plan,
            expected_economics=ExpectedOutcome(
                expected_revenue=rec_rev,
                expected_operating_cost=rec_cost,
                expected_contribution_margin=rec_rev - rec_cost,
            ),
            baseline_economics=ExpectedOutcome(),
            confidence=0.4,
            approval_status=RecommendationStatus.ADVISORY_ONLY,
            execution_eligibility=False,
            provenance={"optimizer_version": "0.2.0", "engine": "greedy_global"},
        )

    def _check_global_feasibility(
        self,
        combo: List[DeploymentAlternative],
        deployments: List[ModelDeployment],
        gpus: List[GPUInventory],
        gpu_map: Dict[str, GPUInventory],
        tenant_policies: Dict[str, TenantPolicy],
        profiles: Dict[str, PerformanceProfile],
    ) -> Tuple[bool, List[ExpectedOutcome]]:
        gpu_usage: Dict[str, int] = {g.gpu_id: 0 for g in gpus}
        tenant_usage: Dict[str, int] = {}
        outcomes = []

        for alt in combo:
            dep = next((d for d in deployments if d.deployment_id == alt.deployment_id), None)
            if dep is None:
                return False, []
            policy = tenant_policies.get(dep.tenant_id, TenantPolicy(tenant_id=dep.tenant_id))
            profile = profiles.get(dep.deployment_id)

            if alt.gpu_pool_id == "__baseline__":
                outcome = self._evaluator.evaluate(alt, None, dep, policy, profile)
                outcomes.append(outcome)
                continue

            gpu = gpu_map.get(alt.gpu_pool_id)
            if gpu is None:
                return False, []

            # Check GPU slot count
            current_usage = gpu_usage.get(gpu.gpu_id, 0)
            if current_usage + alt.replica_count > gpu.available_resource_slots:
                return False, []

            # Check per-GPU memory
            per_replica_memory = alt.expected_memory_mb or dep.expected_memory_mb or 0
            total_memory = per_replica_memory * alt.replica_count
            available_gpu_memory = gpu.available_memory_mb or gpu.memory_capacity_mb
            if total_memory > 0 and total_memory > available_gpu_memory:
                return False, []

            # Check tenant quota
            if policy.max_concurrent_workers is not None:
                current_tenant = tenant_usage.get(dep.tenant_id, 0) + alt.replica_count
                if current_tenant > policy.max_concurrent_workers:
                    return False, []

            gpu_usage[gpu.gpu_id] = current_usage + alt.replica_count
            tenant_usage[dep.tenant_id] = tenant_usage.get(dep.tenant_id, 0) + alt.replica_count

            outcome = self._evaluator.evaluate(alt, gpu, dep, policy, profile)
            outcomes.append(outcome)

        return True, outcomes

    def _make_empty_recommendation(self) -> FusionRecommendation:
        return FusionRecommendation(
            scenario_id="global_allocator",
            recommended_plan=[],
            baseline_plan=[],
            expected_economics=ExpectedOutcome(),
            baseline_economics=ExpectedOutcome(),
            confidence=0.0,
            approval_status=RecommendationStatus.ADVISORY_ONLY,
            execution_eligibility=False,
            provenance={"error": "no_feasible_combination", "optimizer_version": "0.2.0"},
        )
