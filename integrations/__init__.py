"""Integration adapters for Tesseract Foundry, Fusion, Ignite and Adaptive.

Adapters translate between the platform's versioned contracts and each
product's native interface. They are intentionally thin: pure mapping functions
are separated from HTTP/Kubernetes I/O so they can be tested without a network.
"""

from integrations.base import (
    AdaptivePrecisionProvider,
    DeploymentHandle,
    DeploymentProvider,
    IntegrationError,
    IntegrationUnavailable,
    PlacementProvider,
    PlacementResult,
    ProfileProvider,
)

__all__ = [
    "AdaptivePrecisionProvider",
    "DeploymentHandle",
    "DeploymentProvider",
    "IntegrationError",
    "IntegrationUnavailable",
    "PlacementProvider",
    "PlacementResult",
    "ProfileProvider",
]
