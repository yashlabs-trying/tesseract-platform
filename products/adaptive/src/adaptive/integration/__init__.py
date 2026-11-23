"""Integration adapters (Milestone 4).

Versioned, network-free contract translations for sibling Tesseract products:

* Foundry - import benchmark measurements, export profile comparisons with
  confidence and provenance.
* Fusion  - report profile memory requirements, recommend precision under
  placement constraints, project capacity improvements.
* Ignite  - load prevalidated profiles, plan safe replica transitions, expose
  readiness and rollback signals.

Each product remains independently usable; adapters are optional and contain no
hidden coupling to the other products.
"""

from __future__ import annotations

from adaptive.integration.foundry import (
    FOUNDRY_PERFORMANCE_SCHEMA,
    PROFILE_COMPARISON_SCHEMA,
    BenchmarkMeasurement,
    Provenance,
    export_profile_comparison,
    import_performance_profile,
    measurement_to_cost,
)
from adaptive.integration.fusion import (
    PLACEMENT_RECOMMENDATION_SCHEMA,
    MemoryRequirement,
    PlacementCandidate,
    PlacementConstraints,
    PlacementRequest,
    PlacementResponse,
    candidates_from_options,
    placement_document,
    project_capacity,
    recommend_precision_for_placement,
    report_memory_requirements,
)
from adaptive.integration.ignite import (
    IGNITE_PROFILE_LOAD_SCHEMA,
    IgniteLoadRequest,
    IgniteLoadResponse,
    IgniteProvider,
    IgniteReplicaTransition,
    InProcessIgnite,
    load_document,
    plan_replica_transition,
)

__all__ = [
    "FOUNDRY_PERFORMANCE_SCHEMA",
    "IGNITE_PROFILE_LOAD_SCHEMA",
    "PLACEMENT_RECOMMENDATION_SCHEMA",
    "PROFILE_COMPARISON_SCHEMA",
    "BenchmarkMeasurement",
    "IgniteLoadRequest",
    "IgniteLoadResponse",
    "IgniteProvider",
    "IgniteReplicaTransition",
    "InProcessIgnite",
    "MemoryRequirement",
    "PlacementCandidate",
    "PlacementConstraints",
    "PlacementRequest",
    "PlacementResponse",
    "Provenance",
    "candidates_from_options",
    "export_profile_comparison",
    "import_performance_profile",
    "load_document",
    "measurement_to_cost",
    "placement_document",
    "plan_replica_transition",
    "project_capacity",
    "recommend_precision_for_placement",
    "report_memory_requirements",
]
