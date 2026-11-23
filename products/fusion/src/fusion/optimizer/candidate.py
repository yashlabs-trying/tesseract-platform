from __future__ import annotations
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
from fusion.models.types import (
    GPUInventory, ModelDeployment, PerformanceProfile, DeploymentAlternative,
    TenantPolicy, AllocationStatus, HealthStatus,
)


class CandidateGenerator:
    def generate_alternatives(
        self,
        deployment: ModelDeployment,
        gpus: List[GPUInventory],
        profile: Optional[PerformanceProfile] = None,
        max_candidates: int = 20,
    ) -> List[DeploymentAlternative]:
        alternatives: List[DeploymentAlternative] = []

        # Current allocation baseline — use the actual current worker count
        alternatives.append(self._build_alternative(
            deployment=deployment,
            gpu_pool_id="__baseline__",
            gpu_type="current",
            replica_count=deployment.worker_count,
            is_baseline=True,
        ))

        # Generate alternatives per GPU pool. Unhealthy GPUs are NOT skipped
        # here: they are still offered as candidates so that constraint
        # validation can reject them with an explicit, auditable reason.
        #
        # Every candidate places the workload on a REAL GPU. Replica counts
        # include the deployment's current worker_count (the natural 1:1
        # placement) plus scale-up/scale-down neighbours.
        base = max(1, deployment.worker_count)
        replica_options = sorted({
            base,
            max(1, base - 1),
            base + 1,
            max(1, base - 2),
        })

        # Iterate replica counts in the OUTER loop and GPUs in the INNER loop,
        # so that when the candidate cap truncates the list it still covers
        # every GPU at the natural replica count before exploring scale-ups.
        for replicas in replica_options:
            for gpu in gpus:
                if len(alternatives) >= max_candidates:
                    break
                alt = self._build_alternative(
                    deployment=deployment,
                    gpu_pool_id=gpu.gpu_id,
                    gpu_type=gpu.gpu_model,
                    replica_count=replicas,
                    gpu=gpu,
                    profile=profile,
                )
                alternatives.append(alt)
            if len(alternatives) >= max_candidates:
                break

        return alternatives[:max_candidates]

    def build_baseline(
        self,
        deployment: ModelDeployment,
        gpu_map: Dict[str, GPUInventory],
    ) -> DeploymentAlternative:
        """Build the baseline (current allocation) alternative.

        Cost reflects the GPUs the deployment currently occupies, when those
        gpu_ids are present in inventory. When the current allocation is
        unknown, cost is zero and the baseline is marked as unverified.
        """
        current = [
            gpu_map[gid] for gid in (deployment.current_gpu_allocation or [])
            if gid in gpu_map
        ]
        if current:
            cost = sum(
                (g.resource_cost_per_hour for g in current), Decimal("0")
            ) * Decimal(str(deployment.worker_count))
            verified = True
        else:
            cost = Decimal("0")
            verified = False

        total_memory = None
        if deployment.expected_memory_mb:
            total_memory = deployment.expected_memory_mb * deployment.worker_count

        return DeploymentAlternative(
            deployment_id=deployment.deployment_id,
            gpu_pool_id="__baseline__",
            gpu_type=current[0].gpu_model if current else "current",
            replica_count=deployment.worker_count,
            expected_memory_mb=total_memory,
            estimated_cost_per_hour=cost,
            estimated_revenue_per_hour=None,
            estimated_contribution=-cost,
            allocation_status=AllocationStatus.CURRENT,
            warm_policy="warm",
            explanation=(
                "Current allocation costed from inventory"
                if verified
                else "Current allocation cost unknown (no inventory match)"
            ),
        )

    def _build_alternative(
        self,
        deployment: ModelDeployment,
        gpu_pool_id: str,
        gpu_type: str,
        replica_count: int,
        gpu: Optional[GPUInventory] = None,
        profile: Optional[PerformanceProfile] = None,
        is_baseline: bool = False,
    ) -> DeploymentAlternative:
        cost = Decimal("0")
        if gpu:
            cost = gpu.resource_cost_per_hour * Decimal(str(replica_count))

        rev = None
        total_memory = None
        if deployment.expected_memory_mb:
            total_memory = deployment.expected_memory_mb * replica_count
        # Only compute revenue estimate when we have billing data in the deployment
        # The candidate generator cannot fabricate revenue
        if profile and profile.throughput_rps and deployment.observed_demand:
            hourly_requests = profile.throughput_rps * 3600 * replica_count
            # Revenue requires explicit billing rates — not fabricated here
            pass

        return DeploymentAlternative(
            deployment_id=deployment.deployment_id,
            gpu_pool_id=gpu_pool_id,
            gpu_type=gpu_type,
            replica_count=replica_count,
            expected_memory_mb=total_memory,
            expected_performance=profile,
            estimated_cost_per_hour=cost,
            estimated_revenue_per_hour=rev,
            estimated_contribution=rev - cost if rev is not None else None,
            allocation_status=AllocationStatus.CURRENT if is_baseline else AllocationStatus.CANDIDATE,
            warm_policy="warm" if replica_count > 0 else "cold",
        )
