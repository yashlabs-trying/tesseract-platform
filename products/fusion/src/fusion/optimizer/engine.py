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
        # Build lookup: only by gpu_id to avoid collisions
        gpu_map = {g.gpu_id: g for g in gpus}

        recommended_plan: List[DeploymentAlternative] = []
        baseline_plan: List[DeploymentAlternative] = []
        alternative_lists: list = []
        rejected: List[str] = []
        rejected_reasons: Dict[str, str] = {}
        constraint_warnings: Dict[str, List[str]] = {}

        total_baseline_revenue = Decimal("0")
        total_baseline_cost = Decimal("0")
        total_baseline_contribution = Decimal("0")
        total_rec_revenue = Decimal("0")
        total_rec_cost = Decimal("0")
        total_rec_contribution = Decimal("0")

        # Fleet-wide accounting so two workloads are never recommended onto the
        # same physical GPU slots (prevents double-booking in advisory output).
        gpu_slots_used: Dict[str, int] = {}
        gpu_mem_used: Dict[str, int] = {}

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
                gpu = gpu_map.get(alt.gpu_pool_id)
                checks = []

                # When no GPU matched, still record it but with a FAIL
                if gpu is None:
                    if alt.gpu_pool_id == "__baseline__":
                        # Baseline: allow through without GPU check
                        outcome = self._evaluator.evaluate(alt, None, dep, policy, profile)
                        validated_alts.append((alt, outcome))
                        continue
                    else:
                        rejected.append(alt.gpu_pool_id)
                        rejected_reasons[alt.gpu_pool_id] = "No matching GPU found in inventory"
                        continue

                checks = self._constraint_validator.validate_all(alt, gpu, dep, policy, profile)
                failures = [c for c in checks if c.result == "fail"]
                unknowns = [c for c in checks if c.result == "unknown"]

                if not failures:
                    outcome = self._evaluator.evaluate(alt, gpu, dep, policy, profile)
                    validated_alts.append((alt, outcome))
                    if unknowns:
                        constraint_warnings.setdefault(alt.gpu_pool_id, [])
                        for u in unknowns:
                            constraint_warnings[alt.gpu_pool_id].append(
                                f"UNKNOWN: {u.name} — {u.details}"
                            )
                else:
                    rejected.append(alt.gpu_pool_id)
                    rejected_reasons[alt.gpu_pool_id] = "; ".join(
                        f"FAIL {c.name}: {c.details}" for c in failures
                    )

            if not validated_alts:
                # No feasible alternative for this workload — skip
                continue

            # Sort by contribution margin descending
            validated_alts.sort(
                key=lambda x: x[1].expected_contribution_margin or Decimal("-inf"),
                reverse=True,
            )

            # Identify the baseline alternative (marked with "__baseline__")
            baseline_entry: Optional[Tuple[DeploymentAlternative, ExpectedOutcome]] = None
            best_entry: Optional[Tuple[DeploymentAlternative, ExpectedOutcome]] = None

            for entry in validated_alts:
                alt, _outcome = entry
                if alt.gpu_pool_id == "__baseline__":
                    baseline_entry = entry
                    continue
                # Enforce remaining fleet capacity before accepting a placement.
                gpu = gpu_map.get(alt.gpu_pool_id)
                if gpu is not None:
                    used_slots = gpu_slots_used.get(gpu.gpu_id, 0)
                    if used_slots + alt.replica_count > gpu.available_resource_slots:
                        rejected.append(f"{dep.deployment_id}:{alt.gpu_pool_id}")
                        rejected_reasons[
                            f"{dep.deployment_id}:{alt.gpu_pool_id}"
                        ] = "Fleet capacity exhausted by higher-ranked workload"
                        continue
                    per_replica_mem = dep.expected_memory_mb or 0
                    total_mem = per_replica_mem * alt.replica_count
                    used_mem = gpu_mem_used.get(gpu.gpu_id, 0)
                    avail_mem = gpu.available_memory_mb or gpu.memory_capacity_mb
                    if total_mem and used_mem + total_mem > avail_mem:
                        rejected.append(f"{dep.deployment_id}:{alt.gpu_pool_id}")
                        rejected_reasons[
                            f"{dep.deployment_id}:{alt.gpu_pool_id}"
                        ] = "Fleet GPU memory exhausted by higher-ranked workload"
                        continue
                if best_entry is None:
                    best_entry = entry

            # If no explicit baseline found, use the first entry as baseline
            if baseline_entry is None:
                baseline_entry = validated_alts[0]

            # If no better alternative found, keep baseline as recommended too
            if best_entry is None:
                best_entry = baseline_entry

            baseline_alt, baseline_outcome = baseline_entry
            rec_alt, rec_outcome = best_entry

            # Record the committed placement against fleet capacity.
            if rec_alt.gpu_pool_id != "__baseline__":
                gpu = gpu_map.get(rec_alt.gpu_pool_id)
                if gpu is not None:
                    gpu_slots_used[gpu.gpu_id] = (
                        gpu_slots_used.get(gpu.gpu_id, 0) + rec_alt.replica_count
                    )
                    per_replica_mem = dep.expected_memory_mb or 0
                    gpu_mem_used[gpu.gpu_id] = (
                        gpu_mem_used.get(gpu.gpu_id, 0)
                        + per_replica_mem * rec_alt.replica_count
                    )

            baseline_plan.append(baseline_alt)
            total_baseline_revenue += baseline_outcome.expected_revenue
            total_baseline_cost += baseline_outcome.expected_operating_cost
            if baseline_outcome.expected_contribution_margin is not None:
                total_baseline_contribution += baseline_outcome.expected_contribution_margin

            rec_alt = rec_alt.model_copy(update={"allocation_status": AllocationStatus.RECOMMENDED})
            recommended_plan.append(rec_alt)
            total_rec_revenue += rec_outcome.expected_revenue
            total_rec_cost += rec_outcome.expected_operating_cost
            if rec_outcome.expected_contribution_margin is not None:
                total_rec_contribution += rec_outcome.expected_contribution_margin

            # Other alternatives
            other_alts = []
            for a, _ in validated_alts:
                if a.gpu_pool_id == "__baseline__":
                    continue
                if best_entry is not None and a is best_entry[0]:
                    continue
                other_alts.append(a)
            alternative_lists.append(other_alts)

        baseline_econ = ExpectedOutcome(
            expected_revenue=total_baseline_revenue,
            expected_operating_cost=total_baseline_cost,
            expected_contribution_margin=total_baseline_contribution,
        )
        rec_econ = ExpectedOutcome(
            expected_revenue=total_rec_revenue,
            expected_operating_cost=total_rec_cost,
            expected_contribution_margin=total_rec_contribution,
        )

        # Confidence depends on whether we have real demand/profiles
        has_profiles = len(profiles) > 0
        has_demand = len(demands) > 0
        if has_profiles and has_demand:
            confidence = 0.7
        elif has_profiles or has_demand:
            confidence = 0.5
        else:
            confidence = 0.3

        return FusionRecommendation(
            scenario_id=scenario.scenario_id if scenario else "default",
            recommended_plan=recommended_plan,
            baseline_plan=baseline_plan,
            alternative_plans=alternative_lists,
            expected_economics=rec_econ,
            baseline_economics=baseline_econ,
            confidence=confidence,
            constraint_checks={k: "unknown" for k in constraint_warnings.keys()},
            rejected_alternatives=rejected,
            rejected_reasons=rejected_reasons,
            risk_explanation="; ".join(
                f"{k}: {', '.join(v)}" for k, v in constraint_warnings.items()
            ) if constraint_warnings else None,
            approval_status=RecommendationStatus.ADVISORY_ONLY,
            execution_eligibility=False,
            provenance={"optimizer_version": "0.2.0", "engine": "deterministic_candidate_fixed"},
        )
