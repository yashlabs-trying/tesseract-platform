"""Pareto frontier utilities (Milestone 2)."""

from __future__ import annotations

from collections.abc import Sequence

Point = tuple[str, float, float]
"""A candidate ``(id, cost, loss)`` where both cost and loss are minimized."""


def dominates(a: Point, b: Point) -> bool:
    """True when ``a`` dominates ``b`` (no worse on both, strictly better on one)."""
    _, cost_a, loss_a = a
    _, cost_b, loss_b = b
    no_worse = cost_a <= cost_b and loss_a <= loss_b
    strictly_better = cost_a < cost_b or loss_a < loss_b
    return no_worse and strictly_better


def pareto_frontier(points: Sequence[Point]) -> list[str]:
    """Return ids of non-dominated candidates, ordered by ascending cost."""
    frontier: list[Point] = []
    for candidate in points:
        if any(dominates(other, candidate) for other in points if other is not candidate):
            continue
        frontier.append(candidate)
    frontier.sort(key=lambda point: (point[1], point[2], point[0]))
    return [point[0] for point in frontier]


def is_antichain(points: Sequence[Point]) -> bool:
    """True when no candidate in ``points`` dominates another."""
    return not any(
        dominates(a, b)
        for i, a in enumerate(points)
        for j, b in enumerate(points)
        if i != j
    )
