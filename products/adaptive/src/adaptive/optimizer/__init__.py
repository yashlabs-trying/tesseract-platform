"""Precision optimization (Milestone 2).

Planned public interface::

    optimize(architecture, sensitivity, constraints) -> OptimizationResult

Implements greedy layer upgrades and a beam-search alternative that minimize
estimated/measured inference cost subject to quality, latency, memory,
throughput and compatibility constraints. Tracks the Pareto frontier and
compares against uniform 4-bit and 8-bit baselines.
"""

from __future__ import annotations
