"""Enumerations shared across the Tesseract Adaptive domain model."""

from __future__ import annotations

from enum import Enum


class DType(str, Enum):
    """Weight/activation storage data types."""

    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"
    fp8_e4m3 = "fp8_e4m3"
    fp8_e5m2 = "fp8_e5m2"
    int8 = "int8"
    int4 = "int4"
    uint8 = "uint8"


class QuantScheme(str, Enum):
    """Affine quantization scheme."""

    symmetric = "symmetric"
    asymmetric = "asymmetric"


class Granularity(str, Enum):
    """Scale/zero-point granularity."""

    per_tensor = "per_tensor"
    per_channel = "per_channel"
    per_group = "per_group"


class QuantMethod(str, Enum):
    """Quantization method / backend implementation."""

    reference = "reference"
    gptq = "gptq"
    awq = "awq"
    bitsandbytes_int8 = "bitsandbytes_int8"
    bitsandbytes_nf4 = "bitsandbytes_nf4"
    fp8 = "fp8"
    gguf = "gguf"


class RuntimeBackend(str, Enum):
    """Execution runtime targeted by a precision profile."""

    torch_eager = "torch_eager"
    torch_compile = "torch_compile"
    mps = "mps"
    cuda = "cuda"
    vllm = "vllm"
    sglang = "sglang"
    tensorrt_llm = "tensorrt_llm"
    llama_cpp = "llama_cpp"


class EvidenceType(str, Enum):
    """Provenance of a measured/estimated quantity."""

    synthetic = "synthetic"
    estimated = "estimated"
    measured = "measured"
    validated = "validated"


class ModelFamily(str, Enum):
    """Supported transformer architecture families."""

    llama = "llama"
    qwen2 = "qwen2"
    qwen3 = "qwen3"
    mistral = "mistral"
    unknown = "unknown"


class LayerFamily(str, Enum):
    """Functional family a layer belongs to (used for sensitivity attribution)."""

    embedding = "embedding"
    attention = "attention"
    mlp = "mlp"
    norm = "norm"
    head = "head"
    other = "other"


class LayerRole(str, Enum):
    """Fine-grained structural role of a layer."""

    token_embedding = "token_embedding"
    attention_q = "attention_q"
    attention_k = "attention_k"
    attention_v = "attention_v"
    attention_o = "attention_o"
    attention_q_norm = "attention_q_norm"
    attention_k_norm = "attention_k_norm"
    mlp_gate = "mlp_gate"
    mlp_up = "mlp_up"
    mlp_down = "mlp_down"
    input_norm = "input_norm"
    post_attention_norm = "post_attention_norm"
    final_norm = "final_norm"
    lm_head = "lm_head"
    other = "other"


_DTYPE_BYTES: dict[DType, float] = {
    DType.fp32: 4.0,
    DType.fp16: 2.0,
    DType.bf16: 2.0,
    DType.fp8_e4m3: 1.0,
    DType.fp8_e5m2: 1.0,
    DType.int8: 1.0,
    DType.uint8: 1.0,
    DType.int4: 0.5,
}


def dtype_bytes(dtype: DType) -> float:
    """Return the storage size in bytes of a single element of ``dtype``."""
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown dtype: {dtype}") from exc


def dtype_from_torch_name(name: str) -> DType:
    """Map a Hugging Face ``torch_dtype`` string to a :class:`DType`."""
    normalized = name.lower().replace("torch.", "")
    mapping = {
        "float32": DType.fp32,
        "fp32": DType.fp32,
        "float16": DType.fp16,
        "fp16": DType.fp16,
        "half": DType.fp16,
        "bfloat16": DType.bf16,
        "bf16": DType.bf16,
    }
    return mapping.get(normalized, DType.fp16)
