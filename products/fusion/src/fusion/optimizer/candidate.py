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

        # Current allocation baseline
        if deployment.current_gpu_allocation:
            alternatives.append(self._build_alternative(
                deployment=deployment,
                gpu_pool_id="current",
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

            for replicas in [deployment.worker_count, max(1, deployment.worker_count - 1), deployment.worker_count + 1]:
                alt = self._build_alternative(
                    deployment=deployment,
                    gpu_pool_id=gpu.cluster_id,
                    gpu_type=gpu.gpu_model,
                    replica_count=replicas,
                    gpu=gpu,
                    profile=profile,
                )
                alternatives.append(alt)

        # Ensure we have at least one alternative
        if not alternatives:
            alternatives.append(self._build_alternative(
                deployment=deployment,
                gpu_pool_id="fallback",
                gpu_type="unknown",
                replica_count=1,
            ))

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
        if profile and profile.throughput_rps:
            rev = Decimal(str(profile.throughput_rps * replica_count * 3600 * 0.0001))

        return DeploymentAlternative(
            deployment_id=deployment.deployment_id,
            gpu_pool_id=gpu_pool_id,
            gpu_type=gpu_type,
            replica_count=replica_count,
            expected_memory_mb=deployment.expected_memory_mb,
            expected_performance=profile,
            estimated_cost_per_hour=cost,
            estimated_revenue_per_hour=rev,
            estimated_contribution=rev - cost if rev else -cost,
            allocation_status=AllocationStatus.CURRENT if is_baseline else AllocationStatus.CANDIDATE,
            warm_policy="warm" if replica_count > 0 else "cold",
        )
