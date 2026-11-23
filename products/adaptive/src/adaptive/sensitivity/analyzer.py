"""Layer sensitivity analysis (Milestone 2).

Implements, cheapest first:

1. ``weight_error`` - unweighted relative weight reconstruction error.
2. ``activation_weighted`` - reconstruction error weighted by mean |activation|.
3. ``diagonal_hessian`` - reconstruction error weighted by activation second
   moments (a diagonal, second-order approximation; *not* an exact Hessian).
4. ``fisher`` - reconstruction error weighted by empirical squared gradients
   from calibration (a real, gradient-based importance estimate).
5. ``output_divergence`` - ground truth: per-layer KL divergence of model
   logits versus the full-precision model.

Only ``output_divergence`` measures model behaviour directly; the cheaper
methods are proxies whose agreement with it is validated by tests.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from adaptive.errors import ConfigurationError
from adaptive.models.architecture import LayerSpec, ModelArchitecture
from adaptive.models.enums import EvidenceType, Granularity, QuantScheme
from adaptive.models.quantization import QuantConfig, int_config
from adaptive.models.sensitivity import SensitivityMethod, SensitivityReport
from adaptive.nets.reference_transformer import causal_lm_loss, make_token_batches
from adaptive.quantize.reference import dequantize_tensor, quantize_tensor

DEFAULT_METHODS: tuple[SensitivityMethod, ...] = (
    SensitivityMethod.weight_error,
    SensitivityMethod.activation_weighted,
    SensitivityMethod.diagonal_hessian,
)
_DATA_METHODS = {
    SensitivityMethod.activation_weighted,
    SensitivityMethod.diagonal_hessian,
    SensitivityMethod.fisher,
    SensitivityMethod.output_divergence,
}
_EPS = 1e-12


def _linear_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """Return the model's quantizable linear layers keyed by name."""
    provider = getattr(model, "linear_layers", None)
    if callable(provider):
        return cast(dict[str, nn.Linear], provider())
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }


@dataclass
class _ActivationStats:
    total: torch.Tensor | None = None
    total_sq: torch.Tensor | None = None
    count: int = 0

    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1]).float()
        if flat.numel() == 0:
            return
        s = flat.sum(dim=0)
        sq = (flat * flat).sum(dim=0)
        self.total = s if self.total is None else self.total + s
        self.total_sq = sq if self.total_sq is None else self.total_sq + sq
        self.count += flat.shape[0]

    def mean(self) -> torch.Tensor:
        if self.total is None or self.count == 0:
            raise ConfigurationError("activation statistics were never populated")
        return self.total / self.count

    def mean_sq(self) -> torch.Tensor:
        if self.total_sq is None or self.count == 0:
            raise ConfigurationError("activation statistics were never populated")
        return self.total_sq / self.count


@dataclass
class _FisherStats:
    total: torch.Tensor | None = None
    batches: int = 0

    def update(self, grad: torch.Tensor) -> None:
        sq = grad.detach().float().pow(2)
        self.total = sq if self.total is None else self.total + sq
        self.batches += 1

    def mean(self) -> torch.Tensor:
        if self.total is None or self.batches == 0:
            raise ConfigurationError("fisher statistics were never populated")
        return self.total / self.batches


def _target_layers(
    architecture: ModelArchitecture,
    model: nn.Module,
    max_layers: int | None,
) -> list[LayerSpec]:
    available = _linear_layers(model)
    specs = [
        spec
        for spec in architecture.layers
        if spec.quantizable
        and len(spec.shape) == 2
        and spec.name in available
    ]
    if max_layers is not None and len(specs) > max_layers:
        step = len(specs) / max_layers
        specs = [specs[int(i * step)] for i in range(max_layers)]
    return specs


@contextlib.contextmanager
def _quantized_weight(layer: nn.Linear, config: QuantConfig) -> Iterator[None]:
    original = layer.weight.data.clone()
    dequantized = dequantize_tensor(quantize_tensor(original.float(), config))
    layer.weight.data.copy_(dequantized.to(original.dtype))
    try:
        yield
    finally:
        layer.weight.data.copy_(original)


def collect_activation_stats(
    model: nn.Module,
    layer_names: Sequence[str],
    batches: Sequence[torch.Tensor],
) -> dict[str, _ActivationStats]:
    modules = _linear_layers(model)
    stats = {name: _ActivationStats() for name in layer_names}
    handles = []

    def make_hook(name: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...], object], None]:
        def hook(
            _module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: object
        ) -> None:
            if inputs:
                stats[name].update(inputs[0].detach())

        return hook

    for name in layer_names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))
    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                model(batch)
    finally:
        for handle in handles:
            handle.remove()
    return stats


def collect_fisher_stats(
    model: nn.Module,
    layer_names: Sequence[str],
    batches: Sequence[torch.Tensor],
) -> dict[str, _FisherStats]:
    modules = _linear_layers(model)
    stats = {name: _FisherStats() for name in layer_names}
    model.train(False)
    for batch in batches:
        model.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = causal_lm_loss(logits, batch)
        loss.backward()  # type: ignore[no-untyped-call]
        for name in layer_names:
            grad = modules[name].weight.grad
            if grad is not None:
                stats[name].update(grad)
    model.zero_grad(set_to_none=True)
    return stats


def _relative_error(delta_sq: torch.Tensor, weight_sq: torch.Tensor) -> float:
    numerator = float(delta_sq.sum().item())
    denominator = float(weight_sq.sum().item())
    if denominator <= _EPS:
        return 0.0
    return math.sqrt(max(numerator, 0.0) / denominator)


def _weighted_error(
    delta_sq: torch.Tensor, weight_sq: torch.Tensor, weights: torch.Tensor
) -> float:
    numerator = float((delta_sq * weights).sum().item())
    denominator = float((weight_sq * weights).sum().item())
    if denominator <= _EPS:
        return 0.0
    return math.sqrt(max(numerator, 0.0) / denominator)


def _quadratic_form(delta_sq: torch.Tensor, weights: torch.Tensor, numel: int) -> float:
    return float((delta_sq * weights).sum().item()) / max(numel, 1)


def _kl_divergence(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    log_ref = torch.log_softmax(reference.float(), dim=-1)
    log_cand = torch.log_softmax(candidate.float(), dim=-1)
    ref = log_ref.exp()
    value = float((ref * (log_ref - log_cand)).sum(dim=-1).mean().item())
    return max(value, 0.0)


def _normalize(raw: dict[str, float]) -> dict[str, float]:
    if not raw:
        return {}
    values = list(raw.values())
    lo, hi = min(values), max(values)
    if hi - lo <= _EPS:
        return {name: 0.0 for name in raw}
    return {name: (value - lo) / (hi - lo) for name, value in raw.items()}


def analyze_sensitivity(
    architecture: ModelArchitecture,
    model: nn.Module,
    sequences: Sequence[Sequence[int]],
    *,
    methods: Sequence[SensitivityMethod] = DEFAULT_METHODS,
    bits: int = 4,
    group_size: int | None = 128,
    calibration_fingerprint: str = "unknown",
    batch_size: int = 2,
    max_batches: int = 8,
    max_layers: int | None = None,
) -> SensitivityReport:
    """Compute per-layer sensitivity rankings for ``model``."""
    method_list = tuple(methods)
    specs = _target_layers(architecture, model, max_layers)
    if not specs:
        raise ConfigurationError("no analyzable linear layers found in the model")
    layer_names = [spec.name for spec in specs]
    families = {spec.name: spec.family for spec in specs}

    needs_data = bool(set(method_list) & _DATA_METHODS)
    batches: list[torch.Tensor] = []
    if needs_data:
        if not sequences:
            raise ConfigurationError(
                "calibration sequences are required for the requested methods"
            )
        batches = make_token_batches(
            sequences, batch_size=batch_size, max_batches=max_batches
        )

    config = int_config(
        bits,
        scheme=QuantScheme.symmetric,
        granularity=(Granularity.per_group if group_size else Granularity.per_channel),
        group_size=group_size,
    )

    modules = _linear_layers(model)
    deltas: dict[str, torch.Tensor] = {}
    weights: dict[str, torch.Tensor] = {}
    for name in layer_names:
        weight = modules[name].weight.detach().float()
        reconstructed = dequantize_tensor(quantize_tensor(weight, config))
        deltas[name] = (weight - reconstructed).pow(2)
        weights[name] = weight.pow(2)

    raw_by_method: dict[SensitivityMethod, dict[str, float]] = {}

    if SensitivityMethod.weight_error in method_list:
        raw_by_method[SensitivityMethod.weight_error] = {
            name: _relative_error(deltas[name], weights[name]) for name in layer_names
        }

    if SensitivityMethod.activation_weighted in method_list or (
        SensitivityMethod.diagonal_hessian in method_list
    ):
        act_stats = collect_activation_stats(model, layer_names, batches)

    if SensitivityMethod.activation_weighted in method_list:
        raw_by_method[SensitivityMethod.activation_weighted] = {
            name: _weighted_error(deltas[name], weights[name], act_stats[name].mean().abs())
            for name in layer_names
        }

    if SensitivityMethod.diagonal_hessian in method_list:
        raw_by_method[SensitivityMethod.diagonal_hessian] = {
            name: _quadratic_form(
                deltas[name], act_stats[name].mean_sq(), modules[name].weight.numel()
            )
            for name in layer_names
        }

    if SensitivityMethod.fisher in method_list:
        fisher = collect_fisher_stats(model, layer_names, batches)
        raw_by_method[SensitivityMethod.fisher] = {
            name: _quadratic_form(deltas[name], fisher[name].mean(), modules[name].weight.numel())
            for name in layer_names
        }

    if SensitivityMethod.output_divergence in method_list:
        raw_by_method[SensitivityMethod.output_divergence] = _output_divergence(
            model, modules, layer_names, config, batches
        )

    scores: dict[str, dict[str, float]] = {name: {} for name in layer_names}
    raw_scores: dict[str, dict[str, float]] = {name: {} for name in layer_names}
    rankings: dict[str, list[str]] = {}
    family_attribution: dict[str, dict[str, float]] = {}

    for method in method_list:
        raw = raw_by_method.get(method, {})
        normalized = _normalize(raw)
        rankings[method.value] = sorted(raw, key=lambda n: (-raw[n], n))
        attribution: dict[str, list[float]] = {}
        for name in layer_names:
            scores[name][method.value] = normalized.get(name, 0.0)
            raw_scores[name][method.value] = raw.get(name, 0.0)
            attribution.setdefault(families[name].value, []).append(normalized.get(name, 0.0))
        family_attribution[method.value] = {
            family: sum(values) / len(values) for family, values in attribution.items()
        }

    notes = [
        "diagonal_hessian is a diagonal second-order approximation, not an exact Hessian",
        "cheap proxies are validated against output_divergence by the test suite",
    ]
    return SensitivityReport(
        model_id=architecture.model_id,
        structural_fingerprint=architecture.structural_fingerprint(),
        calibration_fingerprint=calibration_fingerprint,
        methods=method_list,
        bits=bits,
        group_size=group_size,
        scores=scores,
        raw_scores=raw_scores,
        rankings=rankings,
        family_attribution=family_attribution,
        evidence=EvidenceType.measured,
        notes=tuple(notes),
    )


def _output_divergence(
    model: nn.Module,
    modules: dict[str, nn.Linear],
    layer_names: Sequence[str],
    config: QuantConfig,
    batches: Sequence[torch.Tensor],
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        reference = [model(batch) for batch in batches]
        divergences: dict[str, float] = {}
        for name in layer_names:
            total = 0.0
            with _quantized_weight(modules[name], config):
                for batch, ref_logits in zip(batches, reference, strict=False):
                    candidate = model(batch)
                    total += _kl_divergence(ref_logits, candidate)
            divergences[name] = total / max(len(batches), 1)
    return divergences


__all__ = [
    "DEFAULT_METHODS",
    "SensitivityMethod",
    "analyze_sensitivity",
    "collect_activation_stats",
    "collect_fisher_stats",
]
