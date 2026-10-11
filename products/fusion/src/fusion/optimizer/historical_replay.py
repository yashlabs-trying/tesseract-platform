from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from fusion.models.types import (
    GPUInventory, ModelDeployment, TenantPolicy, PerformanceProfile,
    WorkloadDemand, FusionRecommendation, RecommendationStatus,
    ExpectedOutcome, AllocationStatus, SimulationResult,
    SimulationState, DeploymentAlternative,
)


class HistoricalReplayEngine:
    def replay_and_compare(
        self,
        deployments: List[ModelDeployment],
        gpus: List[GPUInventory],
        tenant_policies: Dict[str, TenantPolicy],
        profiles: Dict[str, PerformanceProfile],
        demand_trace: Dict[str, List[WorkloadDemand]],
        baseline_alternatives: List[DeploymentAlternative],
        candidate_alternatives: List[DeploymentAlternative],
        duration_hours: float = 1.0,
    ) -> SimulationResult:
        result = SimulationResult(
            scenario_name="historical_replay",
            seed=42,
            config={
                "duration_hours": duration_hours,
                "workload_count": len(deployments),
                "gpu_count": len(gpus),
            },
        )
        result.state = SimulationState.RUNNING
        result.started_at = datetime.utcnow()

        # Track baseline and candidate separately
        baseline_metrics = self._simulate_allocation(
            deployments, gpus, tenant_policies, profiles,
            demand_trace, baseline_alternatives,
        )
        candidate_metrics = self._simulate_allocation(
            deployments, gpus, tenant_policies, profiles,
            demand_trace, candidate_alternatives,
        )

        result.total_requests = baseline_metrics["total_requests"]
        result.completed_requests = candidate_metrics["completed_requests"]
        result.slo_violations = candidate_metrics["slo_violations"]
        result.total_gpu_hours = candidate_metrics["gpu_hours"]
        result.total_cost = candidate_metrics["total_cost"]
        result.total_revenue = candidate_metrics["total_revenue"]
        result.total_contribution = candidate_metrics["total_revenue"] - candidate_metrics["total_cost"]
        result.scaling_events = candidate_metrics["scaling_events"]
        result.state = SimulationState.COMPLETED
        result.completed_at = datetime.utcnow()

        result.events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "replay_comparison",
            "description": "Baseline vs Candidate comparison completed",
            "details": {
                "baseline_cost": str(baseline_metrics["total_cost"]),
                "candidate_cost": str(candidate_metrics["total_cost"]),
                "baseline_revenue": str(baseline_metrics["total_revenue"]),
                "candidate_revenue": str(candidate_metrics["total_revenue"]),
                "baseline_contribution": str(baseline_metrics["total_revenue"] - baseline_metrics["total_cost"]),
                "candidate_contribution": str(candidate_metrics["total_revenue"] - candidate_metrics["total_cost"]),
                "savings": str(baseline_metrics["total_cost"] - candidate_metrics["total_cost"]),
                "requests_served": candidate_metrics["completed_requests"],
                "slo_violations": candidate_metrics["slo_violations"],
            },
        })

        return result

    def _simulate_allocation(
        self,
        deployments: List[ModelDeployment],
        gpus: List[GPUInventory],
        tenant_policies: Dict[str, TenantPolicy],
        profiles: Dict[str, PerformanceProfile],
        demand_trace: Dict[str, List[WorkloadDemand]],
        alternatives: List[DeploymentAlternative],
    ) -> Dict[str, Any]:
        gpu_allocation: Dict[str, Optional[str]] = {g.gpu_id: None for g in gpus}
        gpu_hours = 0.0

        alt_map = {a.deployment_id: a for a in alternatives if a.gpu_pool_id != "__baseline__"}

        completed = 0
        slo_violations = 0
        scaling_events = 0
        total_cost = Decimal("0")
        total_revenue = Decimal("0")
        total_requests = 0

        for dep in deployments:
            alt = alt_map.get(dep.deployment_id)
            if alt is None:
                continue

            gpu_for_alt = next((g for g in gpus if g.gpu_id == alt.gpu_pool_id), None)
            if gpu_for_alt:
                gpu_allocation[gpu_for_alt.gpu_id] = dep.deployment_id
                gpu_hours += 1.0
                total_cost += gpu_for_alt.resource_cost_per_hour

            trace = demand_trace.get(dep.deployment_id, [])
            total_requests_in_trace = sum(d.requests_per_second for d in trace)
            total_requests += int(total_requests_in_trace)

            profile = profiles.get(dep.deployment_id)
            if profile and profile.throughput_rps:
                capacity = profile.throughput_rps * alt.replica_count
                served = min(int(total_requests_in_trace), int(capacity))
                completed += served
                if served < total_requests_in_trace:
                    slo_violations += int(total_requests_in_trace - served)
            else:
                completed += int(total_requests_in_trace * 0.95)
                slo_violations += int(total_requests_in_trace * 0.05)

            if dep.observed_demand and dep.observed_demand.billing_model:
                from fusion.models.types import BillingModel
                bm = dep.observed_demand.billing_model
                if bm == BillingModel.PER_TOKEN:
                    total_revenue += Decimal(str(served * 1000 * 0.0001))
                elif bm == BillingModel.PER_REQUEST:
                    total_revenue += Decimal(str(served * 0.01))
                elif bm in (BillingModel.PER_HOUR_RESERVED, BillingModel.PER_HOUR_ON_DEMAND):
                    total_revenue += Decimal("10.00")

        return {
            "total_requests": total_requests,
            "completed_requests": completed,
            "slo_violations": slo_violations,
            "gpu_hours": gpu_hours,
            "total_cost": total_cost,
            "total_revenue": total_revenue,
            "scaling_events": scaling_events,
        }

    def generate_audit_report(
        self,
        baseline: FusionRecommendation,
        candidate: FusionRecommendation,
        sim_result: SimulationResult,
    ) -> str:
        lines = []
        lines.append("# Tesseract Fusion — Historical Replay Audit Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.utcnow().isoformat()}")
        lines.append("")
        lines.append("## Baseline (Current Allocation)")
        lines.append("")
        lines.append(f"- Revenue: ${baseline.expected_economics.expected_revenue}")
        lines.append(f"- Cost: ${baseline.expected_economics.expected_operating_cost}")
        bm = baseline.expected_economics.expected_contribution_margin
        lines.append(f"- Contribution: ${bm if bm is not None else 'N/A'}")
        lines.append("")
        lines.append("## Recommended (Proposed Allocation)")
        lines.append("")
        lines.append(f"- Revenue: ${candidate.expected_economics.expected_revenue}")
        lines.append(f"- Cost: ${candidate.expected_economics.expected_operating_cost}")
        cm = candidate.expected_economics.expected_contribution_margin
        lines.append(f"- Contribution: ${cm if cm is not None else 'N/A'}")
        lines.append("")
        lines.append("## Replay Verification")
        lines.append("")
        lines.append(f"- Requests Served: {sim_result.completed_requests}")
        lines.append(f"- SLO Violations: {sim_result.slo_violations}")
        lines.append(f"- GPU Hours: {sim_result.total_gpu_hours:.1f}")
        lines.append(f"- Cost (replay): ${sim_result.total_cost}")
        lines.append(f"- Revenue (replay): ${sim_result.total_revenue}")
        lines.append(f"- Contribution (replay): ${sim_result.total_contribution or 'N/A'}")
        lines.append("")
        lines.append("## Rejected Alternatives")
        lines.append("")
        for alt, reason in candidate.rejected_reasons.items():
            lines.append(f"- {alt}: {reason}")
        lines.append("")
        lines.append("## Constraint Validation Results")
        lines.append("")
        for name, result in candidate.constraint_checks.items():
            lines.append(f"- {name}: {result}")
        lines.append("")
        lines.append("## Status")
        lines.append("")
        lines.append(f"- Confidence: {candidate.confidence:.1%}")
        lines.append(f"- Approval: {candidate.approval_status.value}")
        lines.append("---")
        lines.append("*All savings and projections are estimates based on historical data replay.*")
        lines.append("*Results are labeled as simulated until verified on real GPU hardware.*")
        return "\n".join(lines)
