"""Contract tests for M3 versioned JSON schemas."""

from __future__ import annotations

import pytest

jsonschema = pytest.importorskip("jsonschema")

from adaptive.controller import ControllerPolicy, ProfileOption, run_simulation
from adaptive.schemas import (
    decision_document,
    load_schema,
    simulation_document,
    telemetry_document,
)
from adaptive.telemetry import memory_pressure_scenario


def _validate(document: dict, schema_name: str) -> None:
    jsonschema.validate(instance=document, schema=dict(load_schema(schema_name)))


def test_telemetry_document_contract() -> None:
    snapshot = memory_pressure_scenario()[0]
    _validate(telemetry_document(snapshot), "telemetry-snapshot-v1.json")


def test_decision_and_simulation_documents_contract() -> None:
    options = [
        ProfileOption(
            profile_id="p4", precision_bits=4, estimated_cost=1.0, weight_bytes=4_000_000_000
        ),
        ProfileOption(
            profile_id="p16", precision_bits=16, estimated_cost=4.0, weight_bytes=16_000_000_000
        ),
    ]
    policy = ControllerPolicy(min_dwell_s=0.0, cooldown_s=0.0)
    result = run_simulation(policy, options, memory_pressure_scenario(), automatic=True)
    assert result.decisions
    _validate(decision_document(result.decisions[0]), "control-decision-v1.json")
    _validate(simulation_document(result), "controller-simulation-v1.json")


def test_m3_schema_ids() -> None:
    for name, expected in (
        ("telemetry-snapshot-v1.json", "tesseract.adaptive.telemetry-snapshot.v1"),
        ("control-decision-v1.json", "tesseract.adaptive.control-decision.v1"),
        ("controller-simulation-v1.json", "tesseract.adaptive.controller-simulation.v1"),
    ):
        assert load_schema(name)["$id"] == expected
