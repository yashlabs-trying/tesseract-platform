"""Synthetic reference transformer used for CPU/MPS sensitivity and quality work.

This is a real (if small) decoder-only transformer with Llama-style RMSNorm,
grouped-query attention, SwiGLU MLP and causal masking. It exists so the
sensitivity, quality and optimization pipelines can be exercised deterministically
on a laptop without downloading weights or requiring CUDA.

Layer names mirror Hugging Face conventions (``model.layers.0.self_attn.q_proj``
etc.) so sensitivity results map directly onto ``ModelArchitecture`` layer names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from adaptive.models.architecture import ModelArchitecture


@dataclass(frozen=True)
class TransformerConfig:
    """Structural configuration for a :class:`ReferenceTransformer`."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int = 128
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_architecture(cls, architecture: ModelArchitecture) -> TransformerConfig:
        return cls(
            vocab_size=architecture.vocab_size,
            hidden_size=architecture.hidden_size,
            intermediate_size=architecture.intermediate_size,
            num_layers=architecture.num_layers,
            num_attention_heads=architecture.num_attention_heads,
            num_key_value_heads=architecture.num_key_value_heads,
            head_dim=architecture.head_dim,
            max_position_embeddings=max(architecture.max_position_embeddings, 2),
            tie_word_embeddings=architecture.tie_word_embeddings,
            attention_bias=architecture.attention_bias,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, q_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.num_key_value_heads != self.num_heads:
            repeat = self.num_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        out: torch.Tensor = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        projected: torch.Tensor = self.o_proj(out)
        return projected


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated: torch.Tensor = functional.silu(self.gate_proj(x))
        gated: torch.Tensor = activated * self.up_proj(x)
        projected: torch.Tensor = self.down_proj(gated)
        return projected


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _TransformerBody(nn.Module):
    """Container whose attribute names mirror Hugging Face module paths."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class ReferenceTransformer(nn.Module):
    """A small deterministic decoder-only transformer."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _TransformerBody(config)
        if config.tie_word_embeddings:
            self.lm_head: nn.Linear | None = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reset_parameters()

    @classmethod
    def from_architecture(
        cls, architecture: ModelArchitecture, *, seed: int = 0
    ) -> ReferenceTransformer:
        model = cls(TransformerConfig.from_architecture(architecture))
        model.reset_parameters(seed)
        return model

    def reset_parameters(self, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(0.0, 0.02, generator=generator)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(0.0, 0.02, generator=generator)
            elif isinstance(module, RMSNorm):
                module.weight.data.fill_(1.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        hidden = self.model.norm(hidden)
        if self.lm_head is None:
            result: torch.Tensor = functional.linear(hidden, self.model.embed_tokens.weight)
            return result
        projected: torch.Tensor = self.lm_head(hidden)
        return projected

    def linear_layers(self) -> dict[str, nn.Linear]:
        """Return quantizable linear layers keyed by HF-style names."""
        return {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, nn.Linear)
        }

    def named_parameters_by_layer(self) -> dict[str, nn.Parameter]:
        return {name: param for name, param in self.named_parameters()}


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Next-token cross entropy for a ``(batch, seq, vocab)`` logits tensor."""
    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = input_ids[:, 1:].reshape(-1)
    result: torch.Tensor = functional.cross_entropy(shift_logits, shift_labels)
    return result


def make_token_batches(
    sequences: Sequence[Sequence[int]],
    *,
    batch_size: int,
    max_batches: int | None = None,
) -> list[torch.Tensor]:
    """Group token id sequences into padded ``(batch, seq)`` tensors."""
    batches: list[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        if not chunk:
            continue
        width = max(len(seq) for seq in chunk)
        padded = [list(seq) + [0] * (width - len(seq)) for seq in chunk]
        batches.append(torch.tensor(padded, dtype=torch.long))
        if max_batches is not None and len(batches) >= max_batches:
            break
    return batches
