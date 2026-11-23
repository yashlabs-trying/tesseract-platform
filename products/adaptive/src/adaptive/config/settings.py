"""Configuration settings for Tesseract Adaptive."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AdaptiveSettings(BaseSettings):
    """Runtime configuration.

    Values may be overridden via environment variables prefixed with
    ``ADAPTIVE_`` (e.g. ``ADAPTIVE_QUALITY_TOLERANCE=0.02``).
    """

    model_config = SettingsConfigDict(env_prefix="ADAPTIVE_", env_file=".env", extra="ignore")

    quality_tolerance: float = Field(
        default=0.01,
        description="Maximum acceptable relative quality degradation (e.g. perplexity).",
    )
    p95_latency_ms: float = Field(default=2000.0, description="P95 latency SLO in milliseconds.")
    memory_budget_gb: float = Field(
        default=24.0, description="Available GPU/unified memory budget."
    )
    min_throughput_tps: float = Field(
        default=0.0, description="Minimum required output tokens per second."
    )
    controller_min_dwell_s: float = Field(
        default=300.0, description="Minimum time before the controller may switch profiles again."
    )
    controller_cooldown_s: float = Field(
        default=120.0, description="Cooldown after a failed switch or rollback."
    )
    controller_hysteresis: float = Field(
        default=0.05, description="Relative margin required to justify a switch (anti-oscillation)."
    )
    controller_rate_limit_per_hour: int = Field(
        default=4, description="Maximum precision switches permitted per hour."
    )
    recommendation_only: bool = Field(
        default=True, description="When true the controller never mutates a live deployment."
    )


def load_settings(**overrides: object) -> AdaptiveSettings:
    """Load settings with optional explicit overrides."""
    return AdaptiveSettings(**overrides)  # type: ignore[arg-type]
