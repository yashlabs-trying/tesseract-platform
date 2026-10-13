"""Platform configuration.

All settings are environment driven and have safe defaults so the platform can
start with no external dependencies (useful for tests and local development).
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class ControllerMode(str, Enum):
    """Optimization controller operating mode.

    The platform always starts in ``OBSERVE_ONLY``. Automatic reconfiguration
    must not be enabled until safety and rollback tests pass.
    """

    OBSERVE_ONLY = "observe_only"
    RECOMMENDATION = "recommendation"
    CONTROLLED_ACTION = "controlled_action"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TESSERACT_", env_file=".env", extra="ignore")

    # Control plane
    state_dsn: str = "sqlite:///./var/platform.db"
    api_host: str = "0.0.0.0"
    api_port: int = 8088
    log_level: str = "INFO"

    # Foundry
    foundry_url: Optional[str] = None
    foundry_api_key: Optional[str] = None

    # Fusion
    fusion_url: Optional[str] = None
    fusion_api_key: Optional[str] = None

    # Ignite
    ignite_proxy_url: Optional[str] = None
    ignite_namespace: str = "tesseract"
    ignite_kubeconfig: Optional[str] = None

    # Optimization controller
    controller_mode: ControllerMode = ControllerMode.OBSERVE_ONLY
    controller_cooldown_seconds: int = 120
    controller_action_budget: int = 0
    controller_ttft_hysteresis_pct: float = 15.0

    # Adaptive (optional)
    adaptive_enabled: bool = False
    adaptive_url: Optional[str] = None

    # Orchestration policy
    require_measured_evidence: bool = False
    default_gpu_model: str = "unknown"
    reservation_ttl_seconds: int = 900
    readiness_timeout_seconds: int = 300
    allow_conservative_defaults: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
