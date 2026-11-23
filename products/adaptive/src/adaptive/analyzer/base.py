"""Architecture adapter protocol.

Adapters translate a Hugging Face ``config.json`` mapping into a normalized
:class:`~adaptive.models.architecture.ModelArchitecture`. They must never
download weights and must never assume a single architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from adaptive.models.architecture import ModelArchitecture


class ArchitectureAdapter(ABC):
    """Base class for model-family architecture adapters."""

    name: str = "base"
    model_types: tuple[str, ...] = ()

    def matches(self, config: Mapping[str, Any]) -> bool:
        """Return True when this adapter can analyze ``config``."""
        return str(config.get("model_type", "")).lower() in self.model_types

    @abstractmethod
    def analyze(
        self,
        config: Mapping[str, Any],
        *,
        model_id: str,
        revision: str | None = None,
    ) -> ModelArchitecture:
        """Build a normalized architecture description."""
        raise NotImplementedError
