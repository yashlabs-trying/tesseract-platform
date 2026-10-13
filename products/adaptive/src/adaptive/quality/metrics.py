"""Quality metrics (Milestone 2).

Every metric returns a :class:`MetricResult`. Reconstruction error is provided
for calibration diagnostics only and is never treated as proof of model quality.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence

import torch
from torch import nn

from adaptive.models.architecture import ModelArchitecture
from adaptive.models.enums import EvidenceType
from adaptive.models.quality import MetricResult, QualityMetric
from adaptive.models.quantization import PrecisionProfile
from adaptive.nets.reference_transformer import make_token_batches
from adaptive.quality.datasets import TokenDataset
from adaptive.quantize.reference import dequantize_tensor, quantize_tensor

_EPS = 1e-12
LogitsFn = Callable[[torch.Tensor], torch.Tensor]


def _batches(dataset: TokenDataset, batch_size: int, max_batches: int | None) -> list[torch.Tensor]:
    return make_token_batches(
        [list(seq) for seq in dataset.sequences], batch_size=batch_size, max_batches=max_batches
    )


def _total_nll(logits_fn: LogitsFn, batches: Sequence[torch.Tensor]) -> tuple[float, int]:
    total = 0.0
    tokens = 0
    with torch.no_grad():
        for batch in batches:
            logits = logits_fn(batch)
            shift_logits = logits[:, :-1, :].float()
            shift_labels = batch[:, 1:]
            flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
            flat_labels = shift_labels.reshape(-1)
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_labels, reduction="sum")
            total += float(loss.item())
            tokens += int(flat_labels.numel())
    return total, tokens


def perplexity(
    logits_fn: LogitsFn,
    dataset: TokenDataset,
    *,
    batch_size: int = 2,
    max_batches: int | None = 8,
    baseline_value: float | None = None,
) -> MetricResult:
    """Token-level perplexity of ``logits_fn`` on ``dataset``."""
    batches = _batches(dataset, batch_size, max_batches)
    if not batches:
        raise ValueError("dataset produced no batches")
    total, tokens = _total_nll(logits_fn, batches)
    value = math.exp(total / max(tokens, 1))
    return MetricResult(
        metric=QualityMetric.perplexity,
        value=value,
        num_samples=tokens,
        baseline_value=baseline_value,
    )


def long_context_perplexity(
    logits_fn: LogitsFn,
    dataset: TokenDataset,
    *,
    min_length: int = 32,
    batch_size: int = 1,
    max_batches: int | None = 4,
    baseline_value: float | None = None,
) -> MetricResult:
    """Perplexity restricted to sequences at least ``min_length`` tokens long."""
    long_sequences = [seq for seq in dataset.sequences if len(seq) >= min_length]
    if not long_sequences:
        raise ValueError(f"no sequences of length >= {min_length}")
    subset = TokenDataset(
        name=f"{dataset.name}-long",
        split=dataset.split,
        sequences=tuple(long_sequences),
        seed=dataset.seed,
    )
    result = perplexity(
        logits_fn, subset, batch_size=batch_size, max_batches=max_batches
    )
    return result.model_copy(
        update={"metric": QualityMetric.long_context, "baseline_value": baseline_value}
    )


def calibration_reconstruction(
    architecture: ModelArchitecture,
    profile: PrecisionProfile,
    model: nn.Module,
) -> MetricResult:
    """Mean relative weight reconstruction error over the profile's layers."""
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    total_error = 0.0
    counted = 0
    for spec in architecture.layers:
        if not spec.quantizable or spec.name not in modules:
            continue
        config = profile.config_for(spec)
        if config.is_float:
            continue
        weight = modules[spec.name].weight.detach().float()
        reconstructed = dequantize_tensor(quantize_tensor(weight, config))
        diff = (weight - reconstructed).pow(2).sum().sqrt()
        denom = weight.pow(2).sum().sqrt().clamp_min(_EPS)
        total_error += float((diff / denom).item())
        counted += 1
    value = total_error / counted if counted else 0.0
    return MetricResult(
        metric=QualityMetric.calibration_reconstruction,
        value=value,
        num_samples=counted,
        evidence=EvidenceType.estimated,
    )


def _kl(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    log_ref = torch.log_softmax(reference.float(), dim=-1)
    log_cand = torch.log_softmax(candidate.float(), dim=-1)
    value = float((log_ref.exp() * (log_ref - log_cand)).sum(dim=-1).mean().item())
    return max(value, 0.0)


def output_divergence(
    reference_fn: LogitsFn,
    candidate_fn: LogitsFn,
    dataset: TokenDataset,
    *,
    batch_size: int = 2,
    max_batches: int | None = 8,
) -> MetricResult:
    """Mean KL divergence of candidate logits from reference logits."""
    batches = _batches(dataset, batch_size, max_batches)
    total = 0.0
    tokens = 0
    with torch.no_grad():
        for batch in batches:
            reference = reference_fn(batch)
            candidate = candidate_fn(batch)
            total += _kl(reference, candidate) * batch.shape[0]
            tokens += batch.shape[0]
    value = total / max(tokens, 1)
    return MetricResult(
        metric=QualityMetric.output_divergence,
        value=value,
        num_samples=tokens,
        baseline_value=0.0,
    )


def top1_agreement(
    reference_fn: LogitsFn,
    candidate_fn: LogitsFn,
    dataset: TokenDataset,
    *,
    batch_size: int = 2,
    max_batches: int | None = 8,
) -> MetricResult:
    """Fraction of positions where candidate and reference share the top token."""
    batches = _batches(dataset, batch_size, max_batches)
    agree = 0
    total = 0
    with torch.no_grad():
        for batch in batches:
            ref = reference_fn(batch).argmax(dim=-1)
            cand = candidate_fn(batch).argmax(dim=-1)
            agree += int((ref == cand).sum().item())
            total += int(ref.numel())
    value = agree / max(total, 1)
    return MetricResult(
        metric=QualityMetric.task_accuracy,
        value=value,
        num_samples=total,
        baseline_value=1.0,
    )


def task_accuracy(
    predict_fn: Callable[[Sequence[int]], int],
    examples: Sequence[tuple[Sequence[int], int]],
    *,
    baseline_value: float | None = None,
) -> MetricResult:
    """Exact-match accuracy over ``(prompt_tokens, label)`` examples."""
    if not examples:
        raise ValueError("task dataset is empty")
    correct = sum(1 for prompt, label in examples if predict_fn(prompt) == label)
    value = correct / len(examples)
    return MetricResult(
        metric=QualityMetric.task_accuracy,
        value=value,
        num_samples=len(examples),
        baseline_value=baseline_value,
    )


def structured_output(
    generate_fn: Callable[[str], str],
    prompts: Sequence[str],
    validator: Callable[[str], bool] | None = None,
) -> MetricResult:
    """Fraction of generated outputs that are valid structured (JSON) documents."""
    if not prompts:
        raise ValueError("prompt set is empty")

    def is_valid(text: str) -> bool:
        if validator is not None:
            return validator(text)
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    valid = sum(1 for prompt in prompts if is_valid(generate_fn(prompt)))
    value = valid / len(prompts)
    return MetricResult(
        metric=QualityMetric.structured_output,
        value=value,
        num_samples=len(prompts),
    )
