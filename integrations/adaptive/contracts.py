"""Reserved Adaptive contracts.

This module is the *only* place the platform learns about Adaptive. It is
versioned and intentionally narrow so the real Adaptive repository can be
connected later without touching the coordinator.
"""

from __future__ import annotations

from pydantic import BaseModel

from integrations.base import AdaptivePrecisionProvider
from tesseract_platform.models import PrecisionProfileRef

#: Versioned contract identifiers.
PRECISION_PROFILE_SCHEMA = "tesseract.platform.precision-profile.v1"
TRANSITION_PLAN_SCHEMA = "tesseract.platform.precision-transition.v1"


class PrecisionConstraint(BaseModel):
    """A constraint reported by Adaptive for a workload (e.g. unsupported runtime)."""

    name: str
    satisfied: bool
    detail: str = ""


__all__ = [
    "AdaptivePrecisionProvider",
    "PrecisionConstraint",
    "PrecisionProfileRef",
    "PRECISION_PROFILE_SCHEMA",
    "TRANSITION_PLAN_SCHEMA",
]
