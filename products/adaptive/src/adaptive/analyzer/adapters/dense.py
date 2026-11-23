"""Shared adapter for Llama-style dense decoder-only transformers.

Llama, Qwen2, Qwen3 and Mistral share the same block layout
(``self_attn.{q,k,v,o}_proj`` + ``mlp.{gate,up,down}_proj`` with RMSNorm),
so a single implementation covers them with small family-specific hooks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from adaptive.analyzer.base import ArchitectureAdapter
from adaptive.errors import ConfigurationError
from adaptive.models.architecture import LayerSpec, ModelArchitecture
from adaptive.models.enums import (
    DType,
    LayerFamily,
    LayerRole,
    ModelFamily,
    dtype_from_torch_name,
)

_REQUIRED_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "vocab_size",
)


class DenseTransformerAdapter(ArchitectureAdapter):
    """Adapter for dense decoder-only transformer families."""

    family: ModelFamily = ModelFamily.unknown
    default_attention_bias: bool = False
    has_qk_norm: bool = False
    default_tie_word_embeddings: bool = False

    def analyze(
        self,
        config: Mapping[str, Any],
        *,
        model_id: str,
        revision: str | None = None,
    ) -> ModelArchitecture:
        self._require(config)
        notes: list[str] = []

        hidden = int(config["hidden_size"])
        num_layers = int(config["num_hidden_layers"])
        num_heads = int(config["num_attention_heads"])
        num_kv_heads = int(config.get("num_key_value_heads", num_heads))
        vocab = int(config["vocab_size"])
        intermediate = int(
            config.get("intermediate_size", config.get("ffn_dim", 4 * hidden))
        )
        head_dim = int(config.get("head_dim", hidden // num_heads))
        max_pos = int(config.get("max_position_embeddings", 0))
        attention_bias = bool(config.get("attention_bias", self.default_attention_bias))
        tie = bool(config.get("tie_word_embeddings", self.default_tie_word_embeddings))
        baseline_dtype = dtype_from_torch_name(
            str(config.get("torch_dtype", config.get("dtype", "float16")))
        )

        if num_heads <= 0 or hidden % num_heads != 0:
            raise ConfigurationError(
                f"hidden_size {hidden} is not divisible by num_attention_heads {num_heads}"
            )
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ConfigurationError(
                f"num_attention_heads {num_heads} is not divisible by "
                f"num_key_value_heads {num_kv_heads}"
            )

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        layers: list[LayerSpec] = []
        layers.append(
            LayerSpec(
                name="model.embed_tokens",
                role=LayerRole.token_embedding,
                family=LayerFamily.embedding,
                shape=(vocab, hidden),
                num_parameters=vocab * hidden,
                quantizable=True,
            )
        )

        for block in range(num_layers):
            prefix = f"model.layers.{block}"
            layers.append(
                LayerSpec(
                    name=f"{prefix}.input_layernorm",
                    role=LayerRole.input_norm,
                    family=LayerFamily.norm,
                    shape=(hidden,),
                    num_parameters=hidden,
                    quantizable=False,
                    block_index=block,
                )
            )
            layers.extend(
                self._linear(
                    f"{prefix}.self_attn.q_proj",
                    (q_dim, hidden),
                    LayerRole.attention_q,
                    LayerFamily.attention,
                    block,
                    bias=attention_bias,
                )
            )
            layers.extend(
                self._linear(
                    f"{prefix}.self_attn.k_proj",
                    (kv_dim, hidden),
                    LayerRole.attention_k,
                    LayerFamily.attention,
                    block,
                    bias=attention_bias,
                )
            )
            layers.extend(
                self._linear(
                    f"{prefix}.self_attn.v_proj",
                    (kv_dim, hidden),
                    LayerRole.attention_v,
                    LayerFamily.attention,
                    block,
                    bias=attention_bias,
                )
            )
            layers.extend(
                self._linear(
                    f"{prefix}.self_attn.o_proj",
                    (hidden, q_dim),
                    LayerRole.attention_o,
                    LayerFamily.attention,
                    block,
                    bias=False,
                )
            )
            if self.has_qk_norm:
                layers.append(
                    LayerSpec(
                        name=f"{prefix}.self_attn.q_norm",
                        role=LayerRole.attention_q_norm,
                        family=LayerFamily.norm,
                        shape=(head_dim,),
                        num_parameters=head_dim,
                        quantizable=False,
                        block_index=block,
                    )
                )
                layers.append(
                    LayerSpec(
                        name=f"{prefix}.self_attn.k_norm",
                        role=LayerRole.attention_k_norm,
                        family=LayerFamily.norm,
                        shape=(head_dim,),
                        num_parameters=head_dim,
                        quantizable=False,
                        block_index=block,
                    )
                )
            layers.append(
                LayerSpec(
                    name=f"{prefix}.post_attention_layernorm",
                    role=LayerRole.post_attention_norm,
                    family=LayerFamily.norm,
                    shape=(hidden,),
                    num_parameters=hidden,
                    quantizable=False,
                    block_index=block,
                )
            )
            layers.extend(
                self._linear(
                    f"{prefix}.mlp.gate_proj",
                    (intermediate, hidden),
                    LayerRole.mlp_gate,
                    LayerFamily.mlp,
                    block,
                )
            )
            layers.extend(
                self._linear(
                    f"{prefix}.mlp.up_proj",
                    (intermediate, hidden),
                    LayerRole.mlp_up,
                    LayerFamily.mlp,
                    block,
                )
            )
            layers.extend(
                self._linear(
                    f"{prefix}.mlp.down_proj",
                    (hidden, intermediate),
                    LayerRole.mlp_down,
                    LayerFamily.mlp,
                    block,
                )
            )

        layers.append(
            LayerSpec(
                name="model.norm",
                role=LayerRole.final_norm,
                family=LayerFamily.norm,
                shape=(hidden,),
                num_parameters=hidden,
                quantizable=False,
            )
        )
        if not tie:
            layers.append(
                LayerSpec(
                    name="lm_head",
                    role=LayerRole.lm_head,
                    family=LayerFamily.head,
                    shape=(vocab, hidden),
                    num_parameters=vocab * hidden,
                    quantizable=True,
                )
            )
        else:
            notes.append("lm_head tied to token embeddings; stored once")

        if baseline_dtype not in (DType.fp16, DType.bf16, DType.fp32):
            notes.append(f"unusual baseline dtype {baseline_dtype.value}; assuming fp16 storage")

        return ModelArchitecture(
            model_id=model_id,
            model_type=str(config["model_type"]).lower(),
            family=self.family,
            revision=revision,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab,
            max_position_embeddings=max_pos,
            tie_word_embeddings=tie,
            attention_bias=attention_bias,
            baseline_dtype=baseline_dtype,
            layers=tuple(layers),
            notes=tuple(notes),
        )

    @staticmethod
    def _linear(
        name: str,
        shape: tuple[int, int],
        role: LayerRole,
        family: LayerFamily,
        block: int,
        *,
        bias: bool = False,
    ) -> list[LayerSpec]:
        out_features, in_features = shape
        params = out_features * in_features + (out_features if bias else 0)
        specs = [
            LayerSpec(
                name=name,
                role=role,
                family=family,
                shape=shape,
                num_parameters=params,
                bias=bias,
                quantizable=True,
                block_index=block,
            )
        ]
        if bias:
            specs.append(
                LayerSpec(
                    name=f"{name}.bias",
                    role=LayerRole.other,
                    family=family,
                    shape=(out_features,),
                    num_parameters=out_features,
                    quantizable=False,
                    block_index=block,
                )
            )
        return specs

    @staticmethod
    def _require(config: Mapping[str, Any]) -> None:
        missing = [key for key in _REQUIRED_KEYS if key not in config]
        if missing:
            raise ConfigurationError(
                f"config missing required keys: {', '.join(missing)}"
            )
