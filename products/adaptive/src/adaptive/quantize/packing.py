"""Packing helpers for sub-byte integer payloads.

The reference quantizer keeps codes in an ``int8`` container for clarity.
Serialization packs 4-bit codes two-per-byte; these helpers exist so storage
accounting can be verified against a real packed representation.
"""

from __future__ import annotations

from adaptive.errors import UnsupportedQuantizationError

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise UnsupportedQuantizationError("packing requires PyTorch")


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 codes in ``[-8, 7]`` into a uint8 buffer.

    Codes are biased by +8 to ``[0, 15]`` and two consecutive codes are packed
    low-nibble-first. An odd element count is zero-padded.
    """
    _require_torch()
    flat = codes.reshape(-1)
    if flat.numel() and (flat.min().item() < -8 or flat.max().item() > 7):
        raise UnsupportedQuantizationError("int4 codes must lie in [-8, 7]")
    unsigned = (flat.to(torch.int16) + 8).to(torch.uint8)
    if unsigned.numel() % 2:
        unsigned = torch.cat([unsigned, torch.zeros(1, dtype=torch.uint8)])
    low = unsigned[0::2]
    high = unsigned[1::2]
    return (low | (high << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Inverse of :func:`pack_int4`; returns ``numel`` signed int4 codes."""
    _require_torch()
    low = (packed & 0x0F).to(torch.int16)
    high = ((packed >> 4) & 0x0F).to(torch.int16)
    interleaved = torch.stack([low, high], dim=1).reshape(-1)
    codes = (interleaved - 8).to(torch.int8)
    return codes[:numel]


def packed_int4_bytes(numel: int) -> int:
    """Serialized byte count of ``numel`` 4-bit codes."""
    return (numel + 1) // 2
