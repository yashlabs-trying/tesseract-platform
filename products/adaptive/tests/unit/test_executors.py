"""M5: backend execution hooks (CPU-runnable paths only)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adaptive.backends import default_executor, fp8_executor
from adaptive.errors import BackendUnavailableError, UnsupportedQuantizationError
from adaptive.models.enums import QuantMethod, RuntimeBackend
from adaptive.quantize import uniform_profile


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(x))


def _profile(tiny_architecture, method: QuantMethod, bits: int):
    return uniform_profile(
        tiny_architecture, bits, runtime=RuntimeBackend.cuda, method=method
    )


def test_default_executor_registry(tiny_architecture) -> None:
    assert default_executor(QuantMethod.fp8) is fp8_executor
    assert default_executor(QuantMethod.bitsandbytes_int8) is not None
    assert default_executor(QuantMethod.bitsandbytes_nf4) is not None
    assert default_executor(QuantMethod.gptq) is not None
    assert default_executor(QuantMethod.awq) is not None
    assert default_executor(QuantMethod.reference) is None


def test_fp8_executor_converts_weights(tiny_architecture) -> None:
    model = _Tiny()
    profile = _profile(tiny_architecture, QuantMethod.fp8, 8)
    result = fp8_executor(model, profile)
    assert result is model
    assert model.proj.weight.dtype == torch.float8_e4m3fn


def test_fp8_executor_rejects_linearless_model(tiny_architecture) -> None:
    model = torch.nn.Embedding(8, 8)
    profile = _profile(tiny_architecture, QuantMethod.fp8, 8)
    with pytest.raises(UnsupportedQuantizationError):
        fp8_executor(model, profile)


def test_gptq_executor_is_delegated() -> None:
    executor = default_executor(QuantMethod.gptq)
    assert executor is not None
    with pytest.raises(BackendUnavailableError):
        executor(object(), None)  # type: ignore[arg-type]
