"""Demo script showing Tesseract Fusion capabilities."""
from pathlib import Path
from decimal import Decimal
from fusion.simulator.scenario import ScenarioLoader
from fusion.simulator.engine import FleetSimulator
from fusion.optimizer.engine import OptimizationEngine
from fusion.models.types import (
    GPUInventory, ModelDeployment, TenantPolicy, PerformanceProfile,
    WorkloadDemand, OptimizationScenario, EvidenceType,
)
from fusion.economics.accounting import CostAccountingEngine, RevenueCalculator, MarginAnalyzer
from fusion.policies.validator import ConstraintValidator, PolicyValidator

DEMO_FLEET = {
    "name": "commercial-demo",
    "description": "Multi-tenant, heterogeneous GPU fleet demonstration",
    "seed": 42,
    "duration_hours": 24.0,
    "gpu_pools": [
        {"gpu_model": "H100", "count": 4, "memory_capacity_mb": 81920, "cost_per_hour": 4.50},
        {"gpu_model": "L40S", "count": 4, "memory_capacity_mb": 49152, "cost_per_hour": 2.25},
        {"gpu_model": "A100", "count": 8, "memory_capacity_mb": 81920, "cost_per_hour": 3.00},
    ],
    "workloads": [
        {"name": "prod-llm", "model_id": "llama-70b", "tenant_id": "acme-corp", "priority": 10,
         "base_rps": 50.0, "burst_multiplier": 3.0, "burst_probability": 0.05,
         "expected_memory_mb": 65536, "min_replicas": 2, "max_replicas": 8,
         "strict_latency_target": True, "isolation_requirement": "exclusive",
         "warm_workers": 2, "scale_to_zero_eligible": False},
        {"name": "bursty-demo", "model_id": "mistral-7b", "tenant_id": "startup-xyz", "priority": 5,
         "base_rps": 5.0, "burst_multiplier": 10.0, "burst_probability": 0.02,
         "expected_memory_mb": 16384, "min_replicas": 1, "max_replicas": 4,
         "strict_latency_target": False, "isolation_requirement": "exclusive",
         "warm_workers": 0, "scale_to_zero_eligible": True},
        {"name": "stable-serve", "model_id": "llama-13b", "tenant_id": "enterprise-bigco", "priority": 7,
         "base_rps": 100.0, "burst_multiplier": 1.5, "burst_probability": 0.01,
         "expected_memory_mb": 32768, "min_replicas": 3, "max_replicas": 6,
         "strict_latency_target": False, "isolation_requirement": "exclusive",
         "warm_workers": 2, "scale_to_zero_eligible": False},
        {"name": "dedicated-model", "model_id": "gpt2", "tenant_id": "secure-tenant", "priority": 8,
         "base_rps": 20.0, "burst_multiplier": 2.0, "burst_probability": 0.1,
         "expected_memory_mb": 8192, "min_replicas": 1, "max_replicas": 2,
         "strict_latency_target": False, "isolation_requirement": "exclusive",
         "warm_workers": 1, "scale_to_zero_eligible": False},
    ],
}


def run_demo():
    print("=" * 70)
    print("TESSERACT FUSION — Commercial Demo")
    print("GPU Fleet Economics and Optimization")
    print("=" * 70)

    loader = ScenarioLoader()
    scenario = loader.load_from_dict(DEMO_FLEET)
    inventory = loader.to_inventory(scenario)
    deployments = loader.to_deployments(scenario)
    policies = loader.to_tenant_policies(scenario)

    # 1. Fleet Overview
    print(f"\n[1] Fleet Overview: {scenario.name}")
    print(f"    Total GPUs: {len(inventory)}")
    print(f"    Total Workloads: {len(scenario.workloads)}")
    print(f"    Duration: {scenario.duration_hours}h")

    model_counts = {}
    model_costs = {}
    for gpu in inventory:
        model_counts[gpu.gpu_model] = model_counts.get(gpu.gpu_model, 0) + 1
        model_costs[gpu.gpu_model] = model_costs.get(gpu.gpu_model, Decimal("0")) + gpu.resource_cost_per_hour
    for model, count in model_counts.items():
        print(f"    {model}: {count}x @ ${model_costs[model]/count:.2f}/GPU/hr (${model_costs[model]:.2f}/hr total)")

    # 2. Simulate Baseline
    print(f"\n[2] Baseline Simulation (Fixed Allocation)")
    sim = FleetSimulator(seed=scenario.seed)
    baseline_result = sim.run(scenario)
    print(f"    Completed Requests: {baseline_result.completed_requests}")
    print(f"    GPU Hours: {baseline_result.total_gpu_hours:.1f}")
    print(f"    Total Cost: ${baseline_result.total_cost:.2f}")
    print(f"    Total Revenue: ${baseline_result.total_revenue:.2f}")
    print(f"    Contribution: ${baseline_result.total_contribution or 0:.2f}")
    print(f"    Scaling Events: {baseline_result.scaling_events}")
    print(f"    SLO Violations: {baseline_result.slo_violations}")

    # 3. Optimize
    print(f"\n[3] Optimization")
    engine = OptimizationEngine()
    recommendation = engine.optimize(
        deployments=deployments,
        gpus=inventory,
        tenant_policies=policies,
        profiles={},
        demands={},
    )
    print(f"    Recommendation ID: {recommendation.recommendation_id}")
    print(f"    Confidence: {recommendation.confidence:.1%}")
    print(f"    Status: {recommendation.approval_status.value}")
    print(f"    Baseline Cost: ${recommendation.baseline_economics.expected_operating_cost:.2f}")
    print(f"    Recommended Cost: ${recommendation.expected_economics.expected_operating_cost:.2f}")
    print(f"    Baseline Revenue: ${recommendation.baseline_economics.expected_revenue:.2f}")
    print(f"    Recommended Revenue: ${recommendation.expected_economics.expected_revenue:.2f}")

    # 4. Economics Comparison
    print(f"\n[4] Economics Comparison")
    analyzer = MarginAnalyzer()
    comparison = analyzer.compare_alternatives(
        recommendation.baseline_economics,
        recommendation.expected_economics,
    )
    for key, val in comparison.items():
        print(f"    {key}: {val}")

    # 5. Constraint Validation
    print(f"\n[5] Constraint Validation")
    validator = ConstraintValidator()
    for dep in deployments:
        policy = policies.get(dep.tenant_id, TenantPolicy(tenant_id=dep.tenant_id))
        for gpu in inventory[:2]:
            from fusion.models.types import DeploymentAlternative
            alt = DeploymentAlternative(
                deployment_id=dep.deployment_id,
                gpu_pool_id=gpu.cluster_id,
                gpu_type=gpu.gpu_model,
                replica_count=dep.worker_count,
            )
            checks = validator.validate_all(alt, gpu, dep, policy)
            failures = [c for c in checks if c.result == "fail"]
            if failures:
                for f in failures:
                    print(f"    FAIL [{dep.deployment_id} on {gpu.gpu_model}]: {f.details}")

    # 6. Cost Analysis
    print(f"\n[6] Detailed Cost Analysis")
    cost_engine = CostAccountingEngine()
    for gpu in inventory[:3]:
        day_cost = cost_engine.compute_gpu_cost(gpu, hours=24)
        week_cost = cost_engine.compute_gpu_cost(gpu, hours=168)
        print(f"    {gpu.gpu_id} ({gpu.gpu_model}): ${gpu.resource_cost_per_hour}/hr, "
              f"${day_cost}/day, ${week_cost}/week")

    print(f"\n{'=' * 70}")
    print("Demo Complete — All simulated results are labeled as simulated.")
    print("No real GPU hardware was used. Costs and revenue are illustrative.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
