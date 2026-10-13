"""Durable platform state."""

from tesseract_platform.state.store import (
    DuplicateOperation,
    StaleRevision,
    StateStore,
)

__all__ = ["StateStore", "DuplicateOperation", "StaleRevision"]
