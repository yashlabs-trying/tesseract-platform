"""Sensitivity analysis domain models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from adaptive.models.enums import EvidenceType, LayerFamily


class SensitivityMethod(str, Enum):
    """Layer-sensitivity estimation methods, cheapest first."""

    weight_error = "weight_error"
    activation_weighted = "activation_weighted"
    diagonal_hessian = "diagonal_hessian"
    fisher = "fisher"
    output_divergence = "output_divergence"


class LayerSensitivity(BaseModel):
    """A single (layer, method) sensitivity measurement."""

    model_config = {"frozen": True}

    layer_name: str
    family: LayerFamily
    method: SensitivityMethod
    bits: int
    group_size: int | None = None
    raw_score: float
    normalized_score: float = 0.0
    rank: int = 0


class SensitivityReport(BaseModel):
    """Layer-wise and group-wise sensitivity rankings for one model + bit width."""

    model_config = {"frozen": True}

    model_id: str
    structural_fingerprint: str
    calibration_fingerprint: str
    methods: tuple[SensitivityMethod, ...]
    bits: int
    group_size: int | None = None
    scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    """layer name -> method -> normalized score in [0, 1] (1 = most sensitive)."""

    raw_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    rankings: dict[str, list[str]] = Field(default_factory=dict)
    """method -> layer names ordered from most to least sensitive."""

    family_attribution: dict[str, dict[str, float]] = Field(default_factory=dict)
    """method -> family -> mean normalized score."""

    evidence: EvidenceType = EvidenceType.estimated
    notes: tuple[str, ...] = Field(default_factory=tuple)

    def score(self, layer_name: str, method: SensitivityMethod | str) -> float:
        key = method.value if isinstance(method, SensitivityMethod) else method
        return self.scores.get(layer_name, {}).get(key, 0.0)

    def ranking(self, method: SensitivityMethod | str) -> list[str]:
        key = method.value if isinstance(method, SensitivityMethod) else method
        return list(self.rankings.get(key, []))

    def top_k(self, method: SensitivityMethod | str, k: int) -> list[str]:
        return self.ranking(method)[:k]
