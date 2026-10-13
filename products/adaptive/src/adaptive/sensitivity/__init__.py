"""Layer sensitivity analysis (Milestone 2).

Planned public interface::

    analyze_sensitivity(architecture, calibration_batches, methods) -> SensitivityReport

Planned methods, cheapest first: activation-weighted reconstruction error,
calibration-based perturbation, diagonal Hessian-inspired approximations, and
optional Fisher-style importance. Attention and MLP families are attributed
separately. No Hessian or Fisher accuracy is claimed until the actual
approximation is implemented and validated against reference divergence.
"""

from __future__ import annotations
