"""Adapter registry and model analyzer entry point."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from adaptive.analyzer.adapters import (
    LlamaAdapter,
    MistralAdapter,
    Qwen2Adapter,
    Qwen3Adapter,
)
from adaptive.analyzer.base import ArchitectureAdapter
from adaptive.errors import ConfigurationError, UnsupportedModelError
from adaptive.logging import get_logger
from adaptive.models.architecture import ModelArchitecture

_log = get_logger("analyzer")


def default_adapters() -> tuple[ArchitectureAdapter, ...]:
    """Return the built-in adapter set."""
    return (LlamaAdapter(), Qwen2Adapter(), Qwen3Adapter(), MistralAdapter())


class AdapterRegistry:
    """Ordered registry that resolves a config to a single adapter."""

    def __init__(self, adapters: Iterable[ArchitectureAdapter] | None = None) -> None:
        self._adapters: list[ArchitectureAdapter] = list(adapters or default_adapters())

    def register(self, adapter: ArchitectureAdapter) -> None:
        self._adapters.insert(0, adapter)

    @property
    def adapters(self) -> tuple[ArchitectureAdapter, ...]:
        return tuple(self._adapters)

    def find(self, config: Mapping[str, Any]) -> ArchitectureAdapter:
        for adapter in self._adapters:
            if adapter.matches(config):
                return adapter
        model_type = config.get("model_type", "<missing>")
        raise UnsupportedModelError(
            f"no architecture adapter for model_type={model_type!r}; "
            f"registered={[a.name for a in self._adapters]}"
        )


class ModelAnalyzer:
    """High-level facade for turning configs into architectures."""

    def __init__(self, registry: AdapterRegistry | None = None) -> None:
        self.registry = registry or AdapterRegistry()

    def analyze(
        self,
        config: Mapping[str, Any] | Any,
        *,
        model_id: str,
        revision: str | None = None,
    ) -> ModelArchitecture:
        mapping = _as_mapping(config)
        adapter = self.registry.find(mapping)
        architecture = adapter.analyze(mapping, model_id=model_id, revision=revision)
        _log.info(
            "analyzed model",
            extra={
                "event": "model_analyzed",
                "model_id": model_id,
                "family": architecture.family.value,
                "adapter": adapter.name,
                "parameters": architecture.total_parameters,
                "layers": len(architecture.layers),
            },
        )
        return architecture

    def analyze_file(
        self,
        path: Path | str,
        *,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> ModelArchitecture:
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigurationError(f"config file not found: {config_path}")
        data = json.loads(config_path.read_text())
        if not isinstance(data, dict):
            raise ConfigurationError(f"config root must be an object: {config_path}")
        return self.analyze(
            data,
            model_id=model_id or config_path.parent.name,
            revision=revision,
        )


def _as_mapping(config: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise ConfigurationError("config must be a mapping or expose to_dict()")
