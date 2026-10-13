"""Backend availability detection (Milestone 5).

Detection is dependency-light: it never imports heavy CUDA libraries eagerly
and never attempts to run kernels. It only reports whether a backend *could*
run in the current environment.
"""

from __future__ import annotations

import importlib.util


def package_available(name: str) -> bool:
    """True when an importable package (or submodule) is present."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def cuda_available() -> bool:
    """True when PyTorch reports a usable CUDA device."""
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_device_name() -> str | None:
    """Return the first CUDA device name, or ``None`` when unavailable."""
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return None


__all__ = ["cuda_available", "cuda_device_name", "package_available"]
