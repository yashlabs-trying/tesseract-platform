"""Apply a precision profile to a live ``nn.Module`` by simulating dequantized weights.

The reference backend has no integer kernels, so "executing" a profile means
replacing each quantized weight with its dequantized reconstruction. This is
adequate for measuring the *quality* impact of a profile and for building the
candidate models used by output-divergence evaluation. It is not a production
execution path.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from adaptive.models.architecture import ModelArchitecture
from adaptive.models.quantization import PrecisionProfile, QuantConfig
from adaptive.quantize.reference import dequantize_tensor, quantize_tensor


def dequantized_weight(weight: torch.Tensor, config: QuantConfig) -> torch.Tensor:
    """Return the dequantized reconstruction of ``weight`` under ``config``."""
    if config.is_float:
        return weight.detach().to(weight.dtype)
    qt = quantize_tensor(weight.detach().float(), config)
    return dequantize_tensor(qt).to(weight.dtype)


def quantize_model(
    architecture: ModelArchitecture,
    profile: PrecisionProfile,
    model: nn.Module,
    *,
    inplace: bool = False,
) -> nn.Module:
    """Return a model whose linear weights are the profile's dequantized weights."""
    target = model if inplace else copy.deepcopy(model)
    modules = dict(target.named_modules())
    specs = {layer.name: layer for layer in architecture.layers}
    for name, module in modules.items():
        if not isinstance(module, nn.Linear):
            continue
        spec = specs.get(name)
        if spec is None or not spec.quantizable:
            continue
        config = profile.config_for(spec)
        with torch.no_grad():
            module.weight.data.copy_(dequantized_weight(module.weight.data, config))
    return target
