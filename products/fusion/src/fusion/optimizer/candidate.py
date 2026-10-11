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

        # Generate alternatives per GPU pool
        for gpu in gpus:
            if gpu.health_status not in (HealthStatus.HEALTHY, HealthStatus.UNKNOWN):
                continue
            if len(alternatives) >= max_candidates:
                break

            for replicas in [
                deployment.worker_count,
                max(1, deployment.worker_count - 1),
                deployment.worker_count + 1,
                max(1, deployment.worker_count - 2),
            ]:
                if replicas != deployment.worker_count:
                    alt = self._build_alternative(
                        deployment=deployment,
                        gpu_pool_id=gpu.gpu_id,
                        gpu_type=gpu.gpu_model,
                        replica_count=replicas,
                        gpu=gpu,
                        profile=profile,
                    )
                    alternatives.append(alt)

        return alternatives[:max_candidates]

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
