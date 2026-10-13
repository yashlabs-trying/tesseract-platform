"""Typed exception hierarchy for Tesseract Adaptive.

All errors raised by the framework derive from :class:`AdaptiveError` so that
callers (controllers, integration adapters, CLI) can distinguish framework
failures from programming errors.
"""

from __future__ import annotations


class AdaptiveError(Exception):
    """Base class for all Tesseract Adaptive errors."""


class ConfigurationError(AdaptiveError):
    """Raised when configuration is structurally invalid or contradictory."""


class UnsupportedModelError(AdaptiveError):
    """Raised when a model architecture cannot be analyzed by any adapter."""


class UnsupportedQuantizationError(AdaptiveError):
    """Raised when a quantization configuration is not supported by a runtime."""


class CompatibilityError(AdaptiveError):
    """Raised when a profile is not executable on the declared runtime."""


class IntegrityError(AdaptiveError):
    """Raised when an artifact fails revision or content verification."""


class QualityGateError(AdaptiveError):
    """Raised when a candidate violates a configured quality constraint."""


class ConstraintViolationError(AdaptiveError):
    """Raised when an optimization candidate violates a hard constraint."""
