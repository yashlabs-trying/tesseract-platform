"""Layer sensitivity analysis (Milestone 2).

Cheapest-first methods: weight reconstruction error, activation-weighted error,
diagonal Hessian-inspired error, gradient-based Fisher importance, and the
ground-truth output divergence. Attention and MLP families are attributed
separately. No Hessian or Fisher accuracy is claimed beyond what the
implemented approximation measures and what the tests validate.
"""

from __future__ import annotations

from adaptive.models.sensitivity import (
    LayerSensitivity,
    SensitivityMethod,
    SensitivityReport,
)
from adaptive.sensitivity.analyzer import (
    DEFAULT_METHODS,
    analyze_sensitivity,
    collect_activation_stats,
    collect_fisher_stats,
)

__all__ = [
    "DEFAULT_METHODS",
    "LayerSensitivity",
    "SensitivityMethod",
    "SensitivityReport",
    "analyze_sensitivity",
    "collect_activation_stats",
    "collect_fisher_stats",
]
