"""Platform runtime assembly.

Builds the control plane from configuration and wires the product adapters.
Providers can be injected (used by tests and by embedding the platform), and any
provider may be ``None`` when its product is not configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from integrations.adaptive import build_adaptive_provider
from integrations.foundry import FoundryAdapter
from integrations.fusion import FusionAdapter
from integrations.ignite import IgniteAdapter
from tesseract_platform.config import Settings, get_settings
from tesseract_platform.coordinator import Reconciler, WorkloadCoordinator
from tesseract_platform.observability import OptimizationController, TelemetryCollector
from tesseract_platform.policies import PlatformPolicy
from tesseract_platform.state import CapacityLedger, StateStore


@dataclass
class Platform:
    settings: Settings
    store: StateStore
    policy: PlatformPolicy
    telemetry: TelemetryCollector
    coordinator: WorkloadCoordinator
    controller: OptimizationController
    reconciler: Reconciler
    precision_provider: object

    def close(self) -> None:
        self.store.close()


def _default_profile_provider(settings: Settings):
    if settings.foundry_url:
        return FoundryAdapter(base_url=settings.foundry_url, api_key=settings.foundry_api_key)
    return None


def _default_placement_provider(settings: Settings):
    if settings.fusion_url:
        return FusionAdapter(base_url=settings.fusion_url, api_key=settings.fusion_api_key)
    return None


def _default_deployment_provider(settings: Settings):
    if settings.ignite_kubeconfig:
        return IgniteAdapter(
            namespace=settings.ignite_namespace, kubeconfig=settings.ignite_kubeconfig
        )
    return None


def build_platform(
    settings: Optional[Settings] = None,
    *,
    profile_provider=None,
    placement_provider=None,
    deployment_provider=None,
    precision_provider=None,
) -> Platform:
    settings = settings or get_settings()
    store = StateStore(settings.state_dsn)
    policy = PlatformPolicy(
        require_measured_evidence=settings.require_measured_evidence,
        allow_conservative_defaults=settings.allow_conservative_defaults,
        reservation_ttl_seconds=settings.reservation_ttl_seconds,
        readiness_timeout_seconds=settings.readiness_timeout_seconds,
        default_gpu_model=settings.default_gpu_model,
    )
    telemetry = TelemetryCollector(store)
    precision = precision_provider if precision_provider is not None else build_adaptive_provider(settings)
    capacity_ledger = (
        CapacityLedger(settings.total_usable_capacity)
        if settings.total_usable_capacity > 0
        else None
    )

    coordinator = WorkloadCoordinator(
        store=store,
        policy=policy,
        profile_provider=profile_provider
        if profile_provider is not None
        else _default_profile_provider(settings),
        placement_provider=placement_provider
        if placement_provider is not None
        else _default_placement_provider(settings),
        deployment_provider=deployment_provider
        if deployment_provider is not None
        else _default_deployment_provider(settings),
        precision_provider=precision,
        telemetry=telemetry,
        capacity_ledger=capacity_ledger,
    )
    controller = OptimizationController(
        store,
        telemetry,
        mode=settings.controller_mode,
        cooldown_seconds=settings.controller_cooldown_seconds,
        action_budget=settings.controller_action_budget,
        ttft_hysteresis_pct=settings.controller_ttft_hysteresis_pct,
    )
    reconciler = Reconciler(
        coordinator, store, readiness_timeout_seconds=settings.readiness_timeout_seconds
    )
    return Platform(
        settings=settings,
        store=store,
        policy=policy,
        telemetry=telemetry,
        coordinator=coordinator,
        controller=controller,
        reconciler=reconciler,
        precision_provider=precision,
    )
