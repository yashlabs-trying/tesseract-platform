"""Production backend adapter framework (Milestone 5).

Backends are adapters over existing quantization libraries (GPTQ, AWQ,
bitsandbytes, FP8). This framework produces *deployment plans* that work
without a GPU and gates actual execution behind availability checks, so no
backend is ever claimed to run when it cannot. Execution itself is delegated to
the library on a validated GPU host; nothing here fabricates results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from adaptive.errors import BackendUnavailableError, UnsupportedQuantizationError
from adaptive.models.architecture import ModelArchitecture
from adaptive.models.enums import EvidenceType, QuantMethod, RuntimeBackend
from adaptive.models.quantization import PrecisionProfile
from adaptive.quantize.storage import compute_storage

Executor = Callable[[Any, PrecisionProfile], Any]

DEPLOYMENT_PLAN_SCHEMA = "tesseract.adaptive.deployment-plan.v1"


class BackendRequirements(BaseModel):
    """Environmental requirements for a backend."""

    model_config = {"frozen": True}

    requires_cuda: bool = True
    requires_packages: tuple[str, ...] = Field(default_factory=tuple)
    requires_any_packages: tuple[str, ...] = Field(default_factory=tuple)
    min_compute_capability: str | None = None
    notes: tuple[str, ...] = Field(default_factory=tuple)

    def summary(self) -> str:
        parts = []
        if self.requires_cuda:
            parts.append("CUDA")
        if self.requires_packages:
            parts.append("packages: " + ", ".join(self.requires_packages))
        if self.requires_any_packages:
            parts.append("one of: " + ", ".join(self.requires_any_packages))
        return "; ".join(parts) if parts else "no special requirements"


class DeploymentPlan(BaseModel):
    """A backend-specific, non-executing deployment plan for a profile."""

    model_config = {"frozen": True}

    backend: RuntimeBackend
    method: QuantMethod
    profile_id: str
    artifacts: tuple[str, ...] = Field(default_factory=tuple)
    calibration_required: bool = False
    calibration_samples: int = 0
    group_size: int | None = None
    expected_weight_bytes: int = 0
    estimated_prepare_time_s: float = 0.0
    supported_bits: tuple[int, ...] = Field(default_factory=tuple)
    executable_here: bool = False
    notes: tuple[str, ...] = Field(default_factory=tuple)
    evidence: EvidenceType = EvidenceType.estimated


def plan_document(plan: DeploymentPlan) -> dict[str, object]:
    """Serialize a deployment plan with a versioned schema id."""
    return {"schema_version": DEPLOYMENT_PLAN_SCHEMA, **plan.model_dump(mode="json")}


class BackendAdapter(ABC):
    """Base class for production quantization backends."""

    name: str = "backend"
    method: QuantMethod = QuantMethod.reference
    runtime: RuntimeBackend = RuntimeBackend.cuda
    supported_bits: tuple[int, ...] = (4, 8)
    requires_calibration: bool = False
    default_calibration_samples: int = 0
    artifact_suffix: str = "safetensors"

    def __init__(self, executor: Executor | None = None) -> None:
        self._executor = executor

    @abstractmethod
    def requirements(self) -> BackendRequirements:
        """Return the environment this backend needs."""

    def available(self) -> bool:
        """True when this backend can execute in the current environment."""
        from adaptive.backends.availability import cuda_available, package_available

        req = self.requirements()
        if req.requires_cuda and not cuda_available():
            return False
        if not all(package_available(pkg) for pkg in req.requires_packages):
            return False
        if req.requires_any_packages and not any(
            package_available(pkg) for pkg in req.requires_any_packages
        ):
            return False
        return True

    def supports(self, profile: PrecisionProfile) -> bool:
        """True when the profile's bit widths are supported by this backend."""
        return all(bits in self.supported_bits for bits in profile.distinct_bits() if bits < 16)

    def plan(self, architecture: ModelArchitecture, profile: PrecisionProfile) -> DeploymentPlan:
        """Produce a deployment plan without executing anything."""
        storage = compute_storage(architecture, profile)
        safe_model = architecture.model_id.replace("/", "-").replace(".", "-")
        artifact = f"{safe_model}-{self.method.value}-{self.artifact_suffix}"
        notes: list[str] = []
        if not self.supports(profile):
            notes.append("profile bit width not supported by this backend")
        if not self.available():
            notes.append(
                f"backend not executable here ({self.requirements().summary()})"
            )
        return DeploymentPlan(
            backend=self.runtime,
            method=self.method,
            profile_id=profile.profile_id,
            artifacts=(artifact,),
            calibration_required=self.requires_calibration,
            calibration_samples=self.default_calibration_samples,
            group_size=profile.default_config.group_size,
            expected_weight_bytes=storage.total_bytes,
            estimated_prepare_time_s=0.0,
            supported_bits=self.supported_bits,
            executable_here=self.available() and self.supports(profile),
            notes=tuple(notes),
            evidence=EvidenceType.estimated,
        )

    def prepare(self, model: Any, profile: PrecisionProfile) -> Any:
        """Execute the backend; requires availability and a configured executor."""
        if not self.supports(profile):
            raise UnsupportedQuantizationError(
                f"{self.name} does not support profile {profile.profile_id}"
            )
        if not self.available():
            raise BackendUnavailableError(
                f"{self.name} is unavailable: {self.requirements().summary()}"
            )
        if self._executor is None:
            raise BackendUnavailableError(
                f"{self.name} has no execution hook configured; run on a validated GPU host"
            )
        return self._executor(model, profile)


__all__ = [
    "BackendAdapter",
    "BackendRequirements",
    "DeploymentPlan",
    "Executor",
]
