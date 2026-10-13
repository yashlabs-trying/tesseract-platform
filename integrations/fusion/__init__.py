"""Fusion integration package."""

from integrations.fusion.adapter import (
    FusionAdapter,
    map_evidence,
    map_recommendation,
)

__all__ = ["FusionAdapter", "map_evidence", "map_recommendation"]
