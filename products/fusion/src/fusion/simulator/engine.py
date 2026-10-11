from __future__ import annotations
from datetime import datetime, timedelta
from decimal import Decimal
from random import Random
from typing import Dict, List, Optional, Tuple
from fusion.models.types import (
    SimulationResult, SimulationEvent, SimulationState,
    GPUInventory, ModelDeployment, TenantPolicy, WorkloadDemand,
    DeploymentAlternative, AllocationStatus,
)
from fusion.simulator.scenario import ScenarioDefinition, SimulatedWorkload
from fusion.economics.accounting import CostAccountingEngine
from fusion.policies.validator import ConstraintValidator


class FleetSimulator:
    def __init__(self, seed: int = 42):
        self._rng = Random(seed)
        self._seed = seed
        self._cost_engine = CostAccountingEngine()
        self._constraint_validator = ConstraintValidator()

    def run(
        self,
        scenario: ScenarioDefinition,
        alternative: Optional[List[DeploymentAlternative]] = None,
    ) -> SimulationResult:
        result = SimulationResult(
            simulation_id=f"sim-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self._seed}",
            scenario_name=scenario.name,
            seed=self._seed,
            config=scenario.model_dump(),
        )
        result.state = SimulationState.RUNNING
        result.started_at = datetime.utcnow()

        gpus = self._build_gpu_pool(scenario)
        gpu_allocations: Dict[str, Optional[str]] = {g.gpu_id: None for g in gpus}

        total_time_steps = int(scenario.duration_hours * 3600 / scenario.dt_seconds)
        result.events.append(SimulationEvent(
            timestamp=datetime.utcnow(),
            event_type="simulation_start",
            description=f"Starting {scenario.name} with {len(gpus)} GPUs, {len(scenario.workloads)} workloads",
        ))

        for step in range(total_time_steps):
            t = step * scenario.dt_seconds
            current_rps: Dict[str, float] = {}

            for w in scenario.workloads:
                base = w.base_rps
                if self._rng.random() < w.burst_probability * (scenario.dt_seconds / 3600):
                    base *= w.burst_multiplier
                    result.scaling_events += 1

                if base > 0:
                    jitter = 1.0 + (self._rng.random() - 0.5) * 0.2
                    current_rps[w.name] = base * jitter
                    result.total_requests += int(base * jitter * scenario.dt_seconds)

            # Simple allocation: first-fit based on availability
            for w in scenario.workloads:
                rps = current_rps.get(w.name, 0)
                if rps == 0:
                    continue
                needed_gpus = w.min_replicas
                assigned = sum(1 for g, o in gpu_allocations.items() if o == w.name)
                if assigned < needed_gpus:
                    for gpu in gpus:
                        if gpu_allocations[gpu.gpu_id] is None:
                            gpu_allocations[gpu.gpu_id] = w.name
                            assigned += 1
                            result.scaling_events += 1
                            if assigned >= needed_gpus:
                                break

            # Track SLO violations (simulated)
            if self._rng.random() < 0.01:
                result.slo_violations += 1

            result.completed_requests += int(sum(current_rps.values()) * scenario.dt_seconds * 0.98)

        # Track GPU hours
        total_gpu_hours = 0.0
        for gpu in gpus:
            if gpu_allocations[gpu.gpu_id] is not None:
                total_gpu_hours += scenario.duration_hours

        result.total_gpu_hours = total_gpu_hours

        # Compute costs
        for gpu in gpus:
            result.total_cost += gpu.resource_cost_per_hour * Decimal(str(scenario.duration_hours))

        # Compute simple revenue (simulated)
        for w in scenario.workloads:
            result.total_revenue += Decimal(str(w.base_rps * scenario.duration_hours * 3600 * 0.00001))

        result.total_contribution = result.total_revenue - result.total_cost
        result.state = SimulationState.COMPLETED
        result.completed_at = datetime.utcnow()

        result.events.append(SimulationEvent(
            timestamp=datetime.utcnow(),
            event_type="simulation_complete",
            description=f"Completed: {result.completed_requests} requests, {result.total_gpu_hours:.1f} GPU-hours",
            details={
                "total_requests": result.total_requests,
                "completed": result.completed_requests,
                "gpu_hours": result.total_gpu_hours,
                "cost": str(result.total_cost),
                "revenue": str(result.total_revenue),
            },
        ))

        return result

    def compare_policies(
        self,
        scenario: ScenarioDefinition,
        alternatives: List[Tuple[str, Optional[List[DeploymentAlternative]]]],
    ) -> Dict[str, SimulationResult]:
        results = {}
        for name, alt in alternatives:
            self._rng = Random(self._seed)
            results[name] = self.run(scenario, alt)
        return results

    def _build_gpu_pool(self, scenario: ScenarioDefinition) -> List[GPUInventory]:
        from fusion.simulator.scenario import ScenarioLoader
        loader = ScenarioLoader()
        return loader.to_inventory(scenario)
