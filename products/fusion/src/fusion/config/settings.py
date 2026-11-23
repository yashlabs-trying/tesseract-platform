from pydantic_settings import BaseSettings
from pydantic import SecretStr
from typing import Optional


class Settings(BaseSettings):
    model_config = {"env_prefix": "FUSION_", "case_sensitive": False}

    # Service
    host: str = "0.0.0.0"
    port: int = 8100
    log_level: str = "INFO"
    environment: str = "development"

    # Database
    database_url: str = "sqlite:///./fusion.db"
    database_echo: bool = False

    # Adapters
    kubernetes_config_file: Optional[str] = None
    kubernetes_in_cluster: bool = False
    prometheus_url: Optional[str] = None
    opencost_url: Optional[str] = None
    dynamo_url: Optional[str] = None
    dynamo_planner_url: Optional[str] = None
    kai_scheduler_url: Optional[str] = None

    # Inventory
    inventory_refresh_seconds: int = 300
    telemetry_refresh_seconds: int = 60
    stale_data_threshold_seconds: int = 600

    # Optimization
    optimizer_timeout_seconds: int = 60
    optimizer_seed: int = 42
    max_candidates_per_workload: int = 20
    hysteresis_threshold: float = 0.1
    min_decision_interval_minutes: int = 15

    # Security
    api_key: Optional[SecretStr] = None
    enable_auth: bool = False
    advisory_mode: bool = True

    # Default currency
    default_currency: str = "USD"


def get_settings() -> Settings:
    return Settings()
