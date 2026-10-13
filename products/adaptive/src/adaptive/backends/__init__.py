"""Production backend adapters (Milestone 5).

Adapters over existing quantization libraries (GPTQ, AWQ, bitsandbytes, FP8)
that produce deployment plans without a GPU and gate execution behind explicit
availability checks. Execution is delegated to the library on a validated GPU
host; no backend is claimed to run when it cannot, and no results are
fabricated.
"""

from __future__ import annotations

from adaptive.backends.adapters import (
    AwqBackend,
    BitsAndBytesInt8Backend,
    BitsAndBytesNf4Backend,
    Fp8Backend,
    GptqBackend,
    available_backends,
    backends_for_runtime,
    default_backends,
    get_backend,
)
from adaptive.backends.availability import cuda_available, cuda_device_name, package_available
from adaptive.backends.base import (
    DEPLOYMENT_PLAN_SCHEMA,
    BackendAdapter,
    BackendRequirements,
    DeploymentPlan,
    plan_document,
)
from adaptive.backends.executors import (
    bitsandbytes_executor,
    default_executor,
    fp8_executor,
)

__all__ = [
    "DEPLOYMENT_PLAN_SCHEMA",
    "AwqBackend",
    "BackendAdapter",
    "BackendRequirements",
    "BitsAndBytesInt8Backend",
    "BitsAndBytesNf4Backend",
    "DeploymentPlan",
    "Fp8Backend",
    "GptqBackend",
    "available_backends",
    "backends_for_runtime",
    "bitsandbytes_executor",
    "cuda_available",
    "cuda_device_name",
    "default_backends",
    "default_executor",
    "fp8_executor",
    "get_backend",
    "package_available",
    "plan_document",
]
