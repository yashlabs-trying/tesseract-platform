"""Runtime telemetry ingestion (Milestone 3).

Planned public interface::

    TelemetrySnapshot: memory_pressure, queue_depth, ttft_ms, itl_ms,
                       tokens_per_second, model_demand

Telemetry is either measured (from a real runtime) or simulated (mock GPU
telemetry for Mac M5 development). The two sources are always labelled with an
:class:`~adaptive.models.enums.EvidenceType` and never conflated.
"""

from __future__ import annotations
