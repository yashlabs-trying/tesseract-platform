# ruff: noqa: UP006, UP035, UP045, I001
"""Independent exhaustive oracle for optimizer correctness verification.

This is a DELIBERATELY SEPARATE implementation from Fusion's optimizer.
It brute-forces all feasible allocations and picks the true optimum by
contribution margin. Used to measure Fusion's optimality gap.
"""
from __future__ import annotations
from decimal import Decimal
from itertools import product
from typing import Dict, List, Optional, Tuple


def oracle_optimal_contribution(
    deployments: List,      # ModelDeployment
    gpus: List,             # GPUInventory
    choices_per_deployment: List[List[Tuple[str, int, Decimal]]],
) -> Tuple[Optional[Decimal], Optional[List[Tuple[str, int]]], int]:
    """Brute-force optimal allocation.

    choices_per_deployment[i] is a list of (gpu_id_or_None, replica_count, cost) options
    for deployment i. gpu_id None means baseline (no GPU, cost 0).

    Returns (best_total_cost_as_negative_contribution, chosen_options, n_combinations).
    Because Fusion maximizes contribution = revenue - cost and revenue is
    identical across plans in the no-billing case, maximizing contribution is
    equivalent to minimizing total cost. We return the minimum total cost.
    """
    gpu_slots = {g.gpu_id: g.available_resource_slots for g in gpus}
    gpu_mem = {g.gpu_id: (g.available_memory_mb or g.memory_capacity_mb) for g in gpus}

    best_cost: Optional[Decimal] = None
    best_choice: Optional[List[Tuple[str, int]]] = None
    n = 0

    for combo in product(*choices_per_deployment):
        n += 1
        usage: Dict[str, int] = {}
        mem_usage: Dict[str, int] = {}
        total_cost = Decimal("0")
        feasible = True
        choice = []

        for (gpu_id, replicas, cost, mem_per_replica) in combo:
            if gpu_id is None:
                choice.append((gpu_id, replicas))
                continue
            usage[gpu_id] = usage.get(gpu_id, 0) + replicas
            if usage[gpu_id] > gpu_slots.get(gpu_id, 0):
                feasible = False
                break
            mem_usage[gpu_id] = mem_usage.get(gpu_id, 0) + replicas * mem_per_replica
            if mem_usage[gpu_id] > gpu_mem.get(gpu_id, 0):
                feasible = False
                break
            total_cost += cost
            choice.append((gpu_id, replicas))

        if feasible and (best_cost is None or total_cost < best_cost):
            best_cost = total_cost
            best_choice = choice

    return best_cost, best_choice, n
