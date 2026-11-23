from __future__ import annotations
from decimal import Decimal
from itertools import product
from typing import Dict, List, Optional, Tuple, Any
from fusion.models.types import (
    GPUInventory, ModelDeployment, TenantPolicy, PerformanceProfile,
    DeploymentAlternative, ExpectedOutcome, AllocationStatus,
    WorkloadDemand, FusionRecommendation, RecommendationStatus, HealthStatus,
)
from fusion.optimizer.candidate import CandidateGenerator
from fusion.optimizer.evaluator import AlternativeEvaluator
from fusion.policies.validator import ConstraintValidator


def _has_ortools() -> bool:
    try:
        import ortools  # noqa: F401
        return True
    except ImportError:
        return False


class GlobalAllocator:
    """Fleet-wide allocation optimizer.

    Unlike the per-workload OptimizationEngine, this allocator considers all
    workloads simultaneously and enforces that the *combined* demand never
    exceeds the physical fleet. The baseline (current allocation) is a
    reference point only and is NEVER selected as a candidate: every workload
    that can be placed must be placed on a real GPU.
    """

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

        # Baseline plan: current allocation, for comparison only.
        baseline_plan: List[DeploymentAlternative] = [
            self._candidate_generator.build_baseline(dep, gpu_map)
            for dep in deployments
        ]

        # Candidate space per deployment: real GPU placements, plus an explicit
        # "unplaced" option so the search can still return a partial plan when
        # capacity is insufficient. The objective strongly prefers serving
        # (maximising placed replicas) over not serving.
        all_alts: Dict[str, List[DeploymentAlternative]] = {}
        for dep in deployments:
            profile = profiles.get(dep.deployment_id)
            alts = self._candidate_generator.generate_alternatives(
                dep, gpus, profile,
                max_candidates=self._config.get("max_candidates_per_workload", 64),
            )
            candidates = [a for a in alts if a.gpu_pool_id != "__baseline__"]
            candidates.append(
                DeploymentAlternative(
                    deployment_id=dep.deployment_id,
                    gpu_pool_id="__unplaced__",
                    gpu_type="none",
                    replica_count=0,
                    estimated_cost_per_hour=Decimal("0"),
                    estimated_revenue_per_hour=None,
                    estimated_contribution=Decimal("0"),
                    allocation_status=AllocationStatus.REJECTED,
                    warm_policy="cold",
                    explanation="Not placed: insufficient feasible capacity",
                )
            )
            all_alts[dep.deployment_id] = candidates

        # Exhaustive search is only tractable when the combined candidate space
        # fits within the configured cap. Otherwise fall back to CP-SAT, then
        # greedy. (Without this guard a large space is silently under-explored.)
        total_combos = 1
        for dep in deployments:
            total_combos *= max(1, len(all_alts.get(dep.deployment_id, [])))

        if len(deployments) <= 9 and total_combos <= self._max_combinations:
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

    # ------------------------------------------------------------------ #
    # Exhaustive search
    # ------------------------------------------------------------------ #
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
        placeable = [d for d in deployments if all_alts.get(d.deployment_id)]
        alt_lists = [all_alts[d.deployment_id] for d in placeable]

        if not alt_lists:
            return self._make_empty_recommendation(baseline_plan)

        # Large weight so that serving a workload always dominates cost.
        serve_weight = Decimal("1000000")
        priority_by_dep = {
            d.deployment_id: tenant_policies.get(
                d.tenant_id, TenantPolicy(tenant_id=d.tenant_id)
            ).priority
            for d in deployments
        }
        # A workload is "satisfied" once it is placed with at least its required
        # replica count. Reward is per satisfied workload (weighted by priority),
        # NOT per replica, so the optimizer never over-provisions to inflate the
        # score.
        required_replicas = {
            d.deployment_id: max(1, d.worker_count) for d in deployments
        }

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

            # Primary: maximise the number of satisfied workloads, weighted by
            # tenant priority so that when capacity is scarce the higher-priority
            # workload wins. Serving one more workload dominates any cost delta.
            weighted_served = sum(
                Decimal(1 + priority_by_dep.get(a.deployment_id, 0))
                for a in combo
                if a.gpu_pool_id != "__unplaced__"
                and a.replica_count >= required_replicas.get(a.deployment_id, 1)
            )
            total_cost = sum(o.expected_operating_cost for o in outcomes)
            total_revenue = sum(o.expected_revenue for o in outcomes)
            score = (
                weighted_served * serve_weight
                + total_revenue
                - total_cost
            )
            if score > best_score:
                best_score = score
                best_combination = (list(combo), outcomes)

        if best_combination is None:
            return self._make_empty_recommendation(baseline_plan)

        recommended_plan, outcomes = best_combination
        recommended_plan = [
            alt.model_copy(update={"allocation_status": AllocationStatus.RECOMMENDED})
            for alt in recommended_plan
            if alt.gpu_pool_id != "__unplaced__"
        ]

        total_rev = sum(o.expected_revenue for o in outcomes)
        total_cost = sum(o.expected_operating_cost for o in outcomes)

        baseline_rev = sum(b.estimated_revenue_per_hour or Decimal("0") for b in baseline_plan)
        baseline_cost = sum(b.estimated_cost_per_hour for b in baseline_plan)

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
            provenance={
                "optimizer_version": "0.3.0",
                "engine": "exhaustive_global",
                "combinations_evaluated": combinations_tried,
                "placed": len(recommended_plan),
                "unplaceable": len(deployments) - len(placeable),
            },
        )

    # ------------------------------------------------------------------ #
    # OR-Tools CP-SAT
    # ------------------------------------------------------------------ #
    def _ortools_solve(
        self,
        deployments: List[ModelDeployment],
        gpus: List[GPUInventory],
        gpu_map: Dict[str, GPUInventory],
        tenant_policies: Dict[str, TenantPolicy],
        profiles: Dict[str, PerformanceProfile],
        all_alts: Dict[str, List[DeploymentAlternative]],
        baseline_plan: List[DeploymentAlternative],
    ) -> FusionRecommendation:
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        # One boolean per (deployment, candidate alternative)
        choices: Dict[Tuple[str, int], Any] = {}
        for dep in deployments:
            for idx, _alt in enumerate(all_alts.get(dep.deployment_id, [])):
                choices[(dep.deployment_id, idx)] = model.NewBoolVar(
                    f"{dep.deployment_id}_{idx}"
                )
            # Exactly one alternative per placeable deployment
            row = [
                choices[(dep.deployment_id, idx)]
                for idx in range(len(all_alts.get(dep.deployment_id, [])))
            ]
            if row:
                model.AddExactlyOne(row)

        # Capacity constraints per GPU
        for gpu in gpus:
            terms = []
            for dep in deployments:
                for idx, alt in enumerate(all_alts.get(dep.deployment_id, [])):
                    if alt.gpu_pool_id == gpu.gpu_id:
                        terms.append(choices[(dep.deployment_id, idx)] * alt.replica_count)
            if terms:
                model.Add(sum(terms) <= gpu.available_resource_slots)

        # Objective: maximise served replicas (weighted by tenant priority)
        # first, then (revenue - cost).
        serve_weight = 1000000  # cents-equivalent weight per satisfied workload
        objective_terms = []
        for dep in deployments:
            prio = tenant_policies.get(
                dep.tenant_id, TenantPolicy(tenant_id=dep.tenant_id)
            ).priority
            required = max(1, dep.worker_count)
            for idx, alt in enumerate(all_alts.get(dep.deployment_id, [])):
                # scale decimal to integer cents
                cost_cents = int((alt.estimated_cost_per_hour * 100).to_integral_value())
                rev_cents = int(
                    ((alt.estimated_revenue_per_hour or Decimal("0")) * 100).to_integral_value()
                )
                # Reward once per satisfied workload, not per replica.
                serve_reward = (
                    serve_weight * (1 + prio)
                    if alt.replica_count >= required
                    else 0
                )
                # CP-SAT minimizes: subtract the serve reward so serving wins.
                objective_terms.append(
                    choices[(dep.deployment_id, idx)]
                    * (cost_cents - rev_cents - serve_reward)
                )
        if objective_terms:
            model.Minimize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(
            self._config.get("optimizer_timeout_seconds", 30)
        )
        solver.parameters.random_seed = int(self._config.get("optimizer_seed", 42))
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._greedy_allocate(
                deployments, gpus, gpu_map, tenant_policies, profiles,
                all_alts, baseline_plan,
            )

        recommended_plan: List[DeploymentAlternative] = []
        outcomes: List[ExpectedOutcome] = []
        for dep in deployments:
            for idx, alt in enumerate(all_alts.get(dep.deployment_id, [])):
                if solver.Value(choices[(dep.deployment_id, idx)]) == 1:
                    recommended_plan.append(
                        alt.model_copy(update={"allocation_status": AllocationStatus.RECOMMENDED})
                    )
                    gpu = gpu_map.get(alt.gpu_pool_id)
                    policy = tenant_policies.get(dep.tenant_id, TenantPolicy(tenant_id=dep.tenant_id))
                    outcomes.append(
                        self._evaluator.evaluate(
                            alt, gpu, dep, policy, profiles.get(dep.deployment_id)
                        )
                    )
                    break

        total_rev = sum(o.expected_revenue for o in outcomes)
        total_cost = sum(o.expected_operating_cost for o in outcomes)
        baseline_cost = sum(b.estimated_cost_per_hour for b in baseline_plan)

        return FusionRecommendation(
            scenario_id="global_allocator_cpsat",
            recommended_plan=recommended_plan,
            baseline_plan=baseline_plan,
            expected_economics=ExpectedOutcome(
                expected_revenue=total_rev,
                expected_operating_cost=total_cost,
                expected_contribution_margin=total_rev - total_cost,
            ),
            baseline_economics=ExpectedOutcome(
                expected_operating_cost=baseline_cost,
                expected_contribution_margin=-baseline_cost,
            ),
            confidence=0.7,
            approval_status=RecommendationStatus.ADVISORY_ONLY,
            execution_eligibility=False,
            provenance={
                "optimizer_version": "0.3.0",
                "engine": "cpsat_global",
                "solver_status": str(status),
            },
        )

    # ------------------------------------------------------------------ #
    # Greedy fallback
    # ------------------------------------------------------------------ #
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
        """Greedy: assign workloads to GPUs by priority, then best contribution."""
        gpu_slots = {g.gpu_id: g.available_resource_slots for g in gpus}
        gpu_mem_used: Dict[str, int] = {}
        tenant_used: Dict[str, int] = {}
        recommended_plan: List[DeploymentAlternative] = []
        outcomes: List[ExpectedOutcome] = []

        sorted_deps = sorted(
            deployments,
            key=lambda d: tenant_policies.get(
                d.tenant_id, TenantPolicy(tenant_id=d.tenant_id)
            ).priority,
            reverse=True,
        )

        for dep in sorted_deps:
            policy = tenant_policies.get(dep.tenant_id, TenantPolicy(tenant_id=dep.tenant_id))
            profile = profiles.get(dep.deployment_id)
            best_alt = None
            best_outcome = None

            for alt in all_alts.get(dep.deployment_id, []):
                if alt.gpu_pool_id == "__unplaced__":
                    continue
                gpu = gpu_map.get(alt.gpu_pool_id)
                if gpu is None:
                    continue
                if gpu.health_status != HealthStatus.HEALTHY:
                    continue
                if gpu_slots.get(gpu.gpu_id, 0) < alt.replica_count:
                    continue
                # Memory feasibility
                per = dep.expected_memory_mb or 0
                total_mem = per * alt.replica_count
                avail_mem = gpu.available_memory_mb or gpu.memory_capacity_mb
                if total_mem and gpu_mem_used.get(gpu.gpu_id, 0) + total_mem > avail_mem:
                    continue
                # Tenant quota
                if policy.max_concurrent_workers is not None:
                    if tenant_used.get(dep.tenant_id, 0) + alt.replica_count > policy.max_concurrent_workers:
                        continue
                gpu_quota = policy.resource_quotas.get("gpu") if policy.resource_quotas else None
                if gpu_quota is not None:
                    if tenant_used.get(dep.tenant_id, 0) + alt.replica_count > gpu_quota:
                        continue
                outcome = self._evaluator.evaluate(alt, gpu, dep, policy, profile)
                if best_alt is None or (
                    outcome.expected_contribution_margin or Decimal("-inf")
                ) > (best_outcome.expected_contribution_margin or Decimal("-inf")):
                    best_alt = alt
                    best_outcome = outcome

            if best_alt and best_outcome:
                gpu_slots[best_alt.gpu_pool_id] = (
                    gpu_slots.get(best_alt.gpu_pool_id, 0) - best_alt.replica_count
                )
                per = dep.expected_memory_mb or 0
                gpu_mem_used[best_alt.gpu_pool_id] = (
                    gpu_mem_used.get(best_alt.gpu_pool_id, 0)
                    + per * best_alt.replica_count
                )
                tenant_used[dep.tenant_id] = (
                    tenant_used.get(dep.tenant_id, 0) + best_alt.replica_count
                )
                recommended_plan.append(
                    best_alt.model_copy(update={"allocation_status": AllocationStatus.RECOMMENDED})
                )
                outcomes.append(best_outcome)

        rec_rev = sum(o.expected_revenue for o in outcomes)
        rec_cost = sum(o.expected_operating_cost for o in outcomes)
        baseline_cost = sum(b.estimated_cost_per_hour for b in baseline_plan)

        return FusionRecommendation(
            scenario_id="global_allocator_greedy",
            recommended_plan=recommended_plan,
            baseline_plan=baseline_plan,
            expected_economics=ExpectedOutcome(
                expected_revenue=rec_rev,
                expected_operating_cost=rec_cost,
                expected_contribution_margin=rec_rev - rec_cost,
            ),
            baseline_economics=ExpectedOutcome(
                expected_operating_cost=baseline_cost,
                expected_contribution_margin=-baseline_cost,
            ),
            confidence=0.4,
            approval_status=RecommendationStatus.ADVISORY_ONLY,
            execution_eligibility=False,
            provenance={
                "optimizer_version": "0.3.0",
                "engine": "greedy_global",
                "placed": len(recommended_plan),
            },
        )

    # ------------------------------------------------------------------ #
    # Feasibility
    # ------------------------------------------------------------------ #
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
        outcomes: List[ExpectedOutcome] = []
        dep_by_id = {d.deployment_id: d for d in deployments}

        for alt in combo:
            dep = dep_by_id.get(alt.deployment_id)
            if dep is None:
                return False, []
            policy = tenant_policies.get(dep.tenant_id, TenantPolicy(tenant_id=dep.tenant_id))
            profile = profiles.get(dep.deployment_id)

            # Explicitly unplaced: consumes no capacity.
            if alt.gpu_pool_id == "__unplaced__":
                outcomes.append(ExpectedOutcome())
                continue

            gpu = gpu_map.get(alt.gpu_pool_id)
            if gpu is None:
                return False, []

            # Unhealthy / unknown hardware is never a valid placement target.
            if gpu.health_status not in (HealthStatus.HEALTHY,):
                return False, []

            # GPU slot count
            current_usage = gpu_usage.get(gpu.gpu_id, 0)
            if current_usage + alt.replica_count > gpu.available_resource_slots:
                return False, []

            # Per-GPU memory. Note: alt.expected_memory_mb is already the TOTAL
            # across replicas (per-replica × replica_count), so derive the total
            # from the deployment's per-replica value to avoid double counting.
            per_replica_memory = dep.expected_memory_mb or 0
            total_memory = per_replica_memory * alt.replica_count
            available_gpu_memory = gpu.available_memory_mb or gpu.memory_capacity_mb
            if total_memory > 0 and total_memory > available_gpu_memory:
                return False, []

            # Tenant worker quota
            if policy.max_concurrent_workers is not None:
                current_tenant = tenant_usage.get(dep.tenant_id, 0) + alt.replica_count
                if current_tenant > policy.max_concurrent_workers:
                    return False, []

            # Tenant resource_quotas (e.g. {"gpu": N})
            gpu_quota = policy.resource_quotas.get("gpu") if policy.resource_quotas else None
            if gpu_quota is not None:
                current_tenant = tenant_usage.get(dep.tenant_id, 0) + alt.replica_count
                if current_tenant > gpu_quota:
                    return False, []

            gpu_usage[gpu.gpu_id] = current_usage + alt.replica_count
            tenant_usage[dep.tenant_id] = tenant_usage.get(dep.tenant_id, 0) + alt.replica_count

            outcomes.append(self._evaluator.evaluate(alt, gpu, dep, policy, profile))

        return True, outcomes

    def _make_empty_recommendation(
        self, baseline_plan: Optional[List[DeploymentAlternative]] = None
    ) -> FusionRecommendation:
        return FusionRecommendation(
            scenario_id="global_allocator",
            recommended_plan=[],
            baseline_plan=baseline_plan or [],
            expected_economics=ExpectedOutcome(),
            baseline_economics=ExpectedOutcome(),
            confidence=0.0,
            approval_status=RecommendationStatus.ADVISORY_ONLY,
            execution_eligibility=False,
            provenance={"error": "no_feasible_combination", "optimizer_version": "0.3.0"},
        )
