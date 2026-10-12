"""Synthetic fleet scenarios used by the Tesseract Fusion audit suite.

Every scenario produced here is *simulated*: no real GPU hardware, telemetry or
production traffic is involved.  Scenarios are plain ``ScenarioDefinition``-compatible
dictionaries (see :mod:`fusion.simulator.scenario`).

Workload classes A-O cover the operating envelope the Fusion optimizer is expected
to survive.  The mapping from an abstract class to the concrete ``SimulatedWorkload``
fields is necessarily approximate, because ``SimulatedWorkload`` does not carry
token-level attributes.  Token shape (prefill-heavy, decode-heavy, long-context)
is therefore encoded through ``expected_memory_mb``, ``base_rps`` and replica
bounds, and explained in each scenario ``description``.
"""
from __future__ import annotations

from typing import Any

#: Seeds exercised by the audit suite.
SEEDS: list[int] = [42, 1337, 2024]

#: Fleet sizes (GPU counts) exercised by the audit suite.
FLEET_SIZES: list[int] = [1, 2, 4, 8, 16, 32, 64]

DEFAULT_DURATION_HOURS = 2.0


def _gpu_pool(
    model: str = "H100",
    count: int = 8,
    memory_capacity_mb: int = 81920,
    cost_per_hour: float = 4.5,
) -> dict[str, Any]:
    return {
        "gpu_model": model,
        "count": count,
        "memory_capacity_mb": memory_capacity_mb,
        "cost_per_hour": cost_per_hour,
    }


def _workload(
    name: str,
    model_id: str,
    tenant_id: str,
    base_rps: float = 10.0,
    burst_multiplier: float = 2.0,
    burst_probability: float = 0.0,
    expected_memory_mb: int = 40960,
    min_replicas: int = 1,
    max_replicas: int = 4,
    priority: int = 0,
    strict_latency_target: bool = False,
    warm_workers: int = 1,
    scale_to_zero_eligible: bool = False,
    isolation_requirement: str = "exclusive",
    duration_hours: float = DEFAULT_DURATION_HOURS,
) -> dict[str, Any]:
    return {
        "name": name,
        "model_id": model_id,
        "tenant_id": tenant_id,
        "base_rps": base_rps,
        "burst_multiplier": burst_multiplier,
        "burst_probability": burst_probability,
        "expected_memory_mb": expected_memory_mb,
        "min_replicas": min_replicas,
        "max_replicas": max_replicas,
        "priority": priority,
        "strict_latency_target": strict_latency_target,
        "warm_workers": warm_workers,
        "scale_to_zero_eligible": scale_to_zero_eligible,
        "isolation_requirement": isolation_requirement,
        "duration_hours": duration_hours,
    }


def _scenario(
    name: str,
    description: str,
    gpu_pools: list[dict[str, Any]],
    workloads: list[dict[str, Any]],
    seed: int = 42,
    duration_hours: float = DEFAULT_DURATION_HOURS,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "seed": seed,
        "duration_hours": duration_hours,
        "gpu_pools": gpu_pools,
        "workloads": workloads,
    }


def class_a_steady(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """A: steady, predictable traffic with no bursts."""
    return _scenario(
        "class-a-steady",
        "Steady predictable traffic (no bursts).",
        [_gpu_pool("H100", gpu_count)],
        [
            _workload("a-steady-1", "llama-8b", "tenant-a", base_rps=25.0, burst_probability=0.0),
            _workload("a-steady-2", "mistral-7b", "tenant-a", base_rps=15.0, burst_probability=0.0),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_b_bursty(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """B: highly bursty traffic."""
    return _scenario(
        "class-b-bursty",
        "Highly bursty traffic with frequent large spikes.",
        [_gpu_pool("H100", gpu_count)],
        [
            _workload(
                "b-bursty-1",
                "llama-8b",
                "tenant-b",
                base_rps=10.0,
                burst_multiplier=12.0,
                burst_probability=0.35,
            ),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_c_idle(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """C: mostly idle models."""
    return _scenario(
        "class-c-idle",
        "Mostly idle models with near-zero steady traffic.",
        [_gpu_pool("A100", gpu_count, cost_per_hour=3.0)],
        [
            _workload(
                "c-idle-1",
                "tiny-1b",
                "tenant-c",
                base_rps=0.5,
                burst_probability=0.0,
                scale_to_zero_eligible=True,
                warm_workers=0,
            ),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_d_prefill(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """D: prefill-heavy (long prompts, few output tokens)."""
    return _scenario(
        "class-d-prefill-heavy",
        "Prefill-heavy: long prompts, high KV pressure, low request rate.",
        [_gpu_pool("H100", gpu_count)],
        [
            _workload(
                "d-prefill-1",
                "long-prompt-13b",
                "tenant-d",
                base_rps=6.0,
                expected_memory_mb=61440,
                max_replicas=3,
            ),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_e_decode(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """E: decode-heavy (short prompts, long generations)."""
    return _scenario(
        "class-e-decode-heavy",
        "Decode-heavy: short prompts, long generations, higher concurrency.",
        [_gpu_pool("H100", gpu_count)],
        [
            _workload(
                "e-decode-1",
                "chat-8b",
                "tenant-e",
                base_rps=40.0,
                expected_memory_mb=24576,
                max_replicas=6,
            ),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_f_mixed(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """F: mixed prefill/decode traffic."""
    return _scenario(
        "class-f-mixed",
        "Mixed prefill/decode traffic across two models.",
        [_gpu_pool("H100", gpu_count)],
        [
            _workload(
                "f-prefill", "long-prompt-13b", "tenant-f",
                base_rps=8.0, expected_memory_mb=61440,
            ),
            _workload(
                "f-decode", "chat-8b", "tenant-f",
                base_rps=35.0, expected_memory_mb=24576, max_replicas=6,
            ),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_g_footprints(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """G: multiple models with very different memory footprints."""
    return _scenario(
        "class-g-mixed-footprints",
        "Several models with different memory footprints sharing the fleet.",
        [_gpu_pool("H100", gpu_count)],
        [
            _workload("g-small", "tiny-1b", "tenant-g", base_rps=20.0, expected_memory_mb=8192),
            _workload("g-medium", "llama-8b", "tenant-g", base_rps=20.0, expected_memory_mb=24576),
            _workload("g-large", "llama-70b", "tenant-g", base_rps=12.0, expected_memory_mb=71680),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_h_long_context(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """H: long-context workload (KV cache pressure)."""
    return _scenario(
        "class-h-long-context",
        "Long-context workload: very large KV cache, few replicas per GPU.",
        [_gpu_pool("H100", gpu_count, memory_capacity_mb=81920)],
        [
            _workload(
                "h-long-context-1",
                "long-context-70b",
                "tenant-h",
                base_rps=4.0,
                expected_memory_mb=76800,
                max_replicas=2,
            ),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_i_multi_tenant_slo(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """I: multiple tenants with different SLO targets."""
    return _scenario(
        "class-i-multi-tenant-slo",
        "Multi-tenant fleet with divergent SLO targets and priorities.",
        [_gpu_pool("H100", gpu_count)],
        [
            _workload(
                "i-gold",
                "llama-70b",
                "tenant-gold",
                base_rps=20.0,
                priority=10,
                strict_latency_target=True,
                expected_memory_mb=71680,
            ),
            _workload("i-silver", "llama-8b", "tenant-silver", base_rps=30.0, priority=5),
            _workload("i-bronze", "mistral-7b", "tenant-bronze", base_rps=50.0, priority=1),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_j_spike(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """J: sudden demand spike."""
    return _scenario(
        "class-j-demand-spike",
        "Sudden demand spike: high burst probability and multiplier.",
        [_gpu_pool("H100", gpu_count)],
        [
            _workload(
                "j-spike-1",
                "llama-8b",
                "tenant-j",
                base_rps=12.0,
                burst_multiplier=25.0,
                burst_probability=0.5,
                max_replicas=8,
            ),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_k_collapse(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """K: sudden demand collapse (traffic falls to zero)."""
    return _scenario(
        "class-k-demand-collapse",
        "Sudden demand collapse: scale-to-zero eligible workload with no warm floor.",
        [_gpu_pool("A100", gpu_count, cost_per_hour=3.0)],
        [
            _workload(
                "k-collapse-1",
                "seasonal-8b",
                "tenant-k",
                base_rps=0.2,
                burst_probability=0.0,
                scale_to_zero_eligible=True,
                warm_workers=0,
            ),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_l_ood(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """L: out-of-distribution workload."""
    return _scenario(
        "class-l-out-of-distribution",
        "Out-of-distribution workload: unknown model shape and extreme request rate.",
        [_gpu_pool("L40S", gpu_count, memory_capacity_mb=49152, cost_per_hour=2.25)],
        [
            _workload(
                "l-ood-1",
                "experimental-moe-unknown",
                "tenant-l",
                base_rps=200.0,
                burst_multiplier=3.0,
                burst_probability=0.2,
                expected_memory_mb=45000,
                min_replicas=2,
                max_replicas=8,
            ),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_m_cold_starts(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """M: simultaneous cold starts across all workloads."""
    return _scenario(
        "class-m-cold-starts",
        "Simultaneous cold starts: every workload is scale-to-zero eligible with no warm floor.",
        [_gpu_pool("H100", gpu_count)],
        [
            _workload(
                f"m-cold-{i}",
                f"model-{i}",
                f"tenant-m{i}",
                base_rps=10.0 + 5.0 * i,
                scale_to_zero_eligible=True,
                warm_workers=0,
            )
            for i in range(3)
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_n_degraded(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """N: degraded GPU availability (half the fleet is unavailable)."""
    available = max(1, gpu_count // 2)
    return _scenario(
        "class-n-degraded-gpu",
        f"Degraded GPU availability: only {available} of {gpu_count} GPUs are schedulable.",
        [_gpu_pool("H100", available)],
        [
            _workload("n-degraded-1", "llama-8b", "tenant-n", base_rps=40.0, max_replicas=4),
            _workload(
                "n-degraded-2", "llama-70b", "tenant-n",
                base_rps=20.0, expected_memory_mb=71680,
            ),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


def class_o_heterogeneous(
    seed: int = 42, gpu_count: int = 8, duration_hours: float = DEFAULT_DURATION_HOURS
) -> dict[str, Any]:
    """O: heterogeneous GPU hardware and pricing."""
    h100 = max(1, gpu_count // 2)
    a100 = max(1, gpu_count // 4)
    l40s = max(1, gpu_count - h100 - a100)
    return _scenario(
        "class-o-heterogeneous",
        "Heterogeneous fleet: H100 / A100 / L40S with different prices and memory.",
        [
            _gpu_pool("H100", h100, memory_capacity_mb=81920, cost_per_hour=4.5),
            _gpu_pool("A100", a100, memory_capacity_mb=81920, cost_per_hour=3.0),
            _gpu_pool("L40S", l40s, memory_capacity_mb=49152, cost_per_hour=2.25),
        ],
        [
            _workload("o-large", "llama-70b", "tenant-o", base_rps=15.0, expected_memory_mb=71680),
            _workload("o-small", "llama-8b", "tenant-o", base_rps=30.0, expected_memory_mb=24576),
        ],
        seed=seed,
        duration_hours=duration_hours,
    )


#: Ordered registry of the 15 workload classes A-O.
WORKLOAD_CLASSES = [
    ("A", class_a_steady),
    ("B", class_b_bursty),
    ("C", class_c_idle),
    ("D", class_d_prefill),
    ("E", class_e_decode),
    ("F", class_f_mixed),
    ("G", class_g_footprints),
    ("H", class_h_long_context),
    ("I", class_i_multi_tenant_slo),
    ("J", class_j_spike),
    ("K", class_k_collapse),
    ("L", class_l_ood),
    ("M", class_m_cold_starts),
    ("N", class_n_degraded),
    ("O", class_o_heterogeneous),
]


def all_scenarios(
    seed: int = 42, gpu_count: int = 8
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(class_name, scenario_dict)`` for each of the 15 workload classes."""
    return [
        (label, builder(seed=seed, gpu_count=gpu_count))
        for label, builder in WORKLOAD_CLASSES
    ]


def _mixed_fleet_scenario(gpu_count: int, seed: int = 42) -> dict[str, Any]:
    h100 = gpu_count // 2
    a100 = gpu_count // 4
    l40s = gpu_count - h100 - a100
    pools: list[dict[str, Any]] = []
    if h100:
        pools.append(_gpu_pool("H100", h100, memory_capacity_mb=81920, cost_per_hour=4.5))
    if a100:
        pools.append(_gpu_pool("A100", a100, memory_capacity_mb=81920, cost_per_hour=3.0))
    if l40s:
        pools.append(_gpu_pool("L40S", l40s, memory_capacity_mb=49152, cost_per_hour=2.25))

    return _scenario(
        f"fleet-{gpu_count}gpu",
        f"Mixed workload fleet sized at {gpu_count} GPUs.",
        pools,
        [
            _workload(
                "mixed-steady", "llama-8b", "tenant-1",
                base_rps=20.0, burst_probability=0.0,
            ),
            _workload(
                "mixed-bursty", "mistral-7b", "tenant-2",
                base_rps=8.0, burst_multiplier=8.0, burst_probability=0.2,
            ),
            _workload(
                "mixed-large", "llama-70b", "tenant-3",
                base_rps=10.0, expected_memory_mb=71680, max_replicas=2,
            ),
            _workload(
                "mixed-idle", "tiny-1b", "tenant-4",
                base_rps=0.5, scale_to_zero_eligible=True, warm_workers=0,
            ),
        ],
        seed=seed,
        duration_hours=1.0,
    )


def fleet_configs(seed: int = 42) -> list[tuple[int, dict[str, Any]]]:
    """Return ``(gpu_count, scenario_dict)`` for each fleet size in :data:`FLEET_SIZES`."""
    return [(n, _mixed_fleet_scenario(n, seed=seed)) for n in FLEET_SIZES]
