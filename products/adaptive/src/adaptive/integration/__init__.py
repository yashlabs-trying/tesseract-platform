"""Integration adapters (Milestone 4).

Versioned interfaces for sibling Tesseract products:

* Foundry - import benchmark measurements, export profile comparisons,
  report confidence and provenance.
* Fusion  - report profile memory requirements, recommend precision under
  placement constraints, project capacity improvements.
* Ignite  - load prevalidated profiles, perform safe replica transitions,
  expose readiness and rollback signals.

Each product remains independently usable; adapters are optional and
network-isolated behind explicit contracts.
"""

from __future__ import annotations
