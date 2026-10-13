"""Model architecture domain model and accounting helpers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from adaptive.models.enums import (
    DType,
    LayerFamily,
    LayerRole,
    ModelFamily,
    dtype_bytes,
)


class LayerSpec(BaseModel):
    """A single addressable tensor/parameter group in a model."""

    model_config = {"frozen": True}

    name: str
    role: LayerRole
    family: LayerFamily
    shape: tuple[int, ...]
    num_parameters: int
    bias: bool = False
    quantizable: bool = True
    block_index: int | None = None
    """Transformer block index for per-block layers; ``None`` for global layers."""

    @property
    def is_linear(self) -> bool:
        return len(self.shape) == 2 and self.role not in {
            LayerRole.token_embedding,
            LayerRole.lm_head,
        }

    @property
    def out_features(self) -> int | None:
        return self.shape[0] if self.shape else None

    @property
    def in_features(self) -> int | None:
        return self.shape[-1] if len(self.shape) >= 2 else None

    def storage_bytes(self, dtype: DType) -> int:
        """Storage required to serialize this layer at ``dtype``."""
        return _ceil_bytes(self.num_parameters * dtype_bytes(dtype))


class ModelArchitecture(BaseModel):
    """Normalized architecture description produced by an analyzer adapter."""

    model_config = {"frozen": True}

    model_id: str
    model_type: str
    family: ModelFamily
    revision: str | None = None
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    baseline_dtype: DType
    layers: tuple[LayerSpec, ...]
    quantization_compatible: bool = True
    notes: tuple[str, ...] = Field(default_factory=tuple)

    # ---- aggregate accounting -------------------------------------------

    @property
    def total_parameters(self) -> int:
        return sum(layer.num_parameters for layer in self.layers)

    @property
    def quantizable_parameters(self) -> int:
        return sum(layer.num_parameters for layer in self.layers if layer.quantizable)

    @property
    def baseline_weight_bytes(self) -> int:
        return sum(layer.storage_bytes(self.baseline_dtype) for layer in self.layers)

    def parameter_count_by_family(self) -> dict[LayerFamily, int]:
        totals: dict[LayerFamily, int] = {family: 0 for family in LayerFamily}
        for layer in self.layers:
            totals[layer.family] += layer.num_parameters
        return totals

    def layer(self, name: str) -> LayerSpec:
        for spec in self.layers:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def linear_layers(self) -> tuple[LayerSpec, ...]:
        return tuple(layer for layer in self.layers if layer.is_linear)

    def structural_fingerprint(self) -> str:
        """Deterministic hash of the structural fields (excludes revision)."""
        import hashlib
        import json

        payload = {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
            "baseline_dtype": self.baseline_dtype.value,
            "layers": [
                [layer.name, layer.role.value, list(layer.shape), layer.bias]
                for layer in self.layers
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _ceil_bytes(num_bytes: float) -> int:
    """Round a fractional byte count up to the nearest whole byte."""
    import math

    return int(math.ceil(num_bytes - 1e-9))
