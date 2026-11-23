"""Timezone helpers.

Fusion stores timestamps as naive UTC. ``datetime.utcnow()`` is deprecated
since Python 3.12, so this returns the same value without the deprecation.
"""
from datetime import datetime, timezone


def utcnow_naive() -> datetime:
    """Current UTC time as a naive datetime (same semantics as datetime.utcnow())."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
