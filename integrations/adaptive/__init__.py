"""Adaptive (optional) integration package.

Adaptive is a future product. The platform integrates through the reserved
``AdaptivePrecisionProvider`` contract and MUST function when it is absent.
"""

from integrations.adaptive.adapter import (
    DisabledAdaptiveProvider,
    HttpAdaptiveProvider,
    UnavailableAdaptiveProvider,
    build_adaptive_provider,
)
from integrations.adaptive.contracts import AdaptivePrecisionProvider, PrecisionConstraint

__all__ = [
    "AdaptivePrecisionProvider",
    "PrecisionConstraint",
    "DisabledAdaptiveProvider",
    "UnavailableAdaptiveProvider",
    "HttpAdaptiveProvider",
    "build_adaptive_provider",
]
