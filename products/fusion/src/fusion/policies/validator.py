from __future__ import annotations
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
from fusion.models.types import (
    GPUInventory, TenantPolicy, ModelDeployment, PerformanceProfile,
    DeploymentAlternative, ConstraintResult, ConstraintCheck,
    IsolationMode, HealthStatus, AllocationStatus,
)


class ConstraintValidator:
    def validate_all(
        self,
        alternative: DeploymentAlternative,
        gpu: GPUInventory,
        deployment: ModelDeployment,
        tenant_policy: TenantPolicy,
        performance_profile: Optional[PerformanceProfile] = None,
    ) -> List[ConstraintCheck]:
        checks: List[ConstraintCheck] = []
        checks.append(self._check_gpu_count(gpu, alternative))
        checks.append(self._check_memory_capacity(gpu, deployment))
        checks.append(self._check_exclusive_allocation(gpu, tenant_policy))
        checks.append(self._check_gpu_compatibility(gpu, deployment))
        checks.append(self._check_tenant_quota(tenant_policy, deployment))
        checks.append(self._check_resource_quotas(tenant_policy, alternative))
        checks.append(self._check_health(gpu))
        checks.append(self._check_engine_compatibility(deployment, performance_profile))
        checks.append(self._check_sharing_mode(gpu, tenant_policy))
        checks.append(self._check_latency(performance_profile, tenant_policy))
        return checks

    def all_pass(self, checks: List[ConstraintCheck]) -> bool:
        return all(c.result == ConstraintResult.PASS for c in checks)

    def _check_gpu_count(self, gpu: GPUInventory, alt: DeploymentAlternative) -> ConstraintCheck:
        if alt.replica_count <= gpu.available_resource_slots:
            return ConstraintCheck(name="gpu_count", result=ConstraintResult.PASS)
        return ConstraintCheck(
            name="gpu_count", result=ConstraintResult.FAIL,
            details=f"Requested {alt.replica_count} replicas, GPU has {gpu.available_resource_slots} slots"
        )

    def _check_memory_capacity(self, gpu: GPUInventory, deployment: ModelDeployment) -> ConstraintCheck:
        if deployment.expected_memory_mb is None:
            return ConstraintCheck(name="memory_capacity", result=ConstraintResult.UNKNOWN,
                                   details="Deployment has no expected memory requirement")
        # Use available_memory_mb if provided (more accurate), fall back to capacity_mb
        available = gpu.available_memory_mb or gpu.memory_capacity_mb
        # Alternative's expected_memory_mb already includes replica count scaling
        required = deployment.expected_memory_mb
        if required <= available:
            return ConstraintCheck(name="memory_capacity", result=ConstraintResult.PASS)
        return ConstraintCheck(
            name="memory_capacity", result=ConstraintResult.FAIL,
            details=f"Deployment needs {required}MB, GPU {gpu.gpu_model} has {available}MB available"
        )

    def _check_exclusive_allocation(self, gpu: GPUInventory, policy: TenantPolicy) -> ConstraintCheck:
        if policy.isolation_requirements == IsolationMode.EXCLUSIVE and gpu.sharing_policy != "exclusive_only":
            return ConstraintCheck(
                name="exclusive_allocation", result=ConstraintResult.FAIL,
                details=f"Tenant requires exclusive, GPU policy is {gpu.sharing_policy}"
            )
        return ConstraintCheck(name="exclusive_allocation", result=ConstraintResult.PASS)

    def _check_gpu_compatibility(self, gpu: GPUInventory, deployment: ModelDeployment) -> ConstraintCheck:
        policy_classes = deployment.current_gpu_allocation if deployment.current_gpu_allocation else []
        if policy_classes and gpu.gpu_model not in policy_classes:
            return ConstraintCheck(
                name="gpu_compatibility", result=ConstraintResult.FAIL,
                details=f"Deployment restricted to {policy_classes}, GPU is {gpu.gpu_model}"
            )
        return ConstraintCheck(name="gpu_compatibility", result=ConstraintResult.PASS)

    def _check_tenant_quota(self, policy: TenantPolicy, deployment: ModelDeployment) -> ConstraintCheck:
        # Check against the deployment's expected worker count from the scenario
        # The per-alternative replica check is done in the optimizer's candidate generator
        if policy.max_concurrent_workers is not None:
            if deployment.worker_count > policy.max_concurrent_workers:
                return ConstraintCheck(
                    name="tenant_quota", result=ConstraintResult.FAIL,
                    details=f"Tenant max workers {policy.max_concurrent_workers}, deployment has {deployment.worker_count}"
                )
        return ConstraintCheck(name="tenant_quota", result=ConstraintResult.PASS)

    def _check_resource_quotas(self, policy: TenantPolicy, alt: DeploymentAlternative) -> ConstraintCheck:
        """Enforce the tenant's declared resource_quotas (e.g. {"gpu": N})."""
        gpu_quota = policy.resource_quotas.get("gpu") if policy.resource_quotas else None
        if gpu_quota is not None and alt.replica_count > gpu_quota:
            return ConstraintCheck(
                name="resource_quota", result=ConstraintResult.FAIL,
                details=f"Tenant GPU quota {gpu_quota}, requested {alt.replica_count}",
            )
        return ConstraintCheck(name="resource_quota", result=ConstraintResult.PASS)

    def _check_health(self, gpu: GPUInventory) -> ConstraintCheck:
        if gpu.health_status == HealthStatus.HEALTHY:
            return ConstraintCheck(name="gpu_health", result=ConstraintResult.PASS)
        if gpu.health_status == HealthStatus.UNKNOWN:
            return ConstraintCheck(name="gpu_health", result=ConstraintResult.UNKNOWN,
                                   details="GPU health is unknown")
        return ConstraintCheck(name="gpu_health", result=ConstraintResult.FAIL,
                               details=f"GPU health is {gpu.health_status}")

    def _check_engine_compatibility(self, deployment: ModelDeployment, profile: Optional[PerformanceProfile]) -> ConstraintCheck:
        if profile is None:
            return ConstraintCheck(name="engine_compatibility", result=ConstraintResult.UNKNOWN,
                                   details="No performance profile available")
        if profile.engine_version and deployment.serving_engine not in profile.engine_version:
            return ConstraintCheck(
                name="engine_compatibility", result=ConstraintResult.FAIL,
                details=f"Deployment engine {deployment.serving_engine} incompatible with profile {profile.engine_version}"
            )
        return ConstraintCheck(name="engine_compatibility", result=ConstraintResult.PASS)

    def _check_sharing_mode(self, gpu: GPUInventory, policy: TenantPolicy) -> ConstraintCheck:
        if policy.isolation_requirements == IsolationMode.EXCLUSIVE:
            if IsolationMode.EXCLUSIVE in gpu.supported_isolation_modes:
                return ConstraintCheck(name="sharing_mode", result=ConstraintResult.PASS)
            return ConstraintCheck(name="sharing_mode", result=ConstraintResult.FAIL,
                                   details="Exclusive isolation requested but not supported by GPU")
        return ConstraintCheck(name="sharing_mode", result=ConstraintResult.PASS)

    def _check_latency(self, profile: Optional[PerformanceProfile], policy: TenantPolicy) -> ConstraintCheck:
        if policy.latency_p99_ms is None or profile is None or profile.ttft_p99_ms is None:
            return ConstraintCheck(name="latency_slo", result=ConstraintResult.UNKNOWN,
                                   details="Insufficient data for latency check")
        if profile.ttft_p99_ms <= policy.latency_p99_ms:
            return ConstraintCheck(name="latency_slo", result=ConstraintResult.PASS)
        return ConstraintCheck(name="latency_slo", result=ConstraintResult.FAIL,
                               details=f"Profile TTFT p99 {profile.ttft_p99_ms}ms > policy {policy.latency_p99_ms}ms")


class PolicyValidator:
    def __init__(self):
        self._constraint_validator = ConstraintValidator()

    def validate_deployment(
        self,
        alternative: DeploymentAlternative,
        gpu: GPUInventory,
        deployment: ModelDeployment,
        tenant_policy: TenantPolicy,
        profile: Optional[PerformanceProfile] = None,
    ) -> Tuple[bool, List[ConstraintCheck]]:
        checks = self._constraint_validator.validate_all(alternative, gpu, deployment, tenant_policy, profile)
        passed = all(c.result == ConstraintResult.PASS for c in checks)
        return passed, checks
