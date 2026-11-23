"""M5 end-to-end: deployment planning and measured-cost wiring (no GPU required)."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive.backends import get_backend
from adaptive.integration import import_performance_profile
from adaptive.models.enums import EvidenceType, QuantMethod, RuntimeBackend
from adaptive.optimizer import MeasuredCostModel, measured_costs_from_measurements
from adaptive.quantize import uniform_profile

ROOT = Path(__file__).resolve().parents[2]


def test_deployment_plan_for_gpu_profile(tiny_architecture) -> None:
    backend = get_backend(QuantMethod.gptq)
    profile = uniform_profile(
        tiny_architecture, 4, runtime=RuntimeBackend.cuda, method=QuantMethod.gptq, group_size=128
    )
    plan = backend.plan(tiny_architecture, profile)
    assert plan.expected_weight_bytes > 0
    assert plan.calibration_required
    # no GPU here, so the plan must not claim executability
    assert plan.executable_here is False
    assert any("not executable" in note for note in plan.notes)


def test_measured_cost_model_uses_measurements(tiny_architecture) -> None:
    measurement = import_performance_profile(
        json.loads((ROOT / "fixtures" / "foundry" / "performance-profile-example.json").read_text())
    )
    profile = uniform_profile(
        tiny_architecture, 4, runtime=RuntimeBackend.cuda, method=QuantMethod.gptq
    ).model_copy(update={"profile_id": measurement.profile_id})

    model = MeasuredCostModel(measured_costs_from_measurements([measurement]))
    cost = model.estimate(tiny_architecture, profile)
    assert cost.evidence is EvidenceType.measured
    assert cost.estimated_latency_ms == 71.0
    assert cost.estimated_throughput_tps == 1850.0


def test_measured_cost_model_falls_back_to_analytic(tiny_architecture) -> None:
    measurement = import_performance_profile(
        json.loads((ROOT / "fixtures" / "foundry" / "performance-profile-example.json").read_text())
    )
    profile = uniform_profile(
        tiny_architecture, 8, runtime=RuntimeBackend.cuda, method=QuantMethod.gptq
    )
    model = MeasuredCostModel(measured_costs_from_measurements([measurement]))
    cost = model.estimate(tiny_architecture, profile)
    assert cost.evidence is EvidenceType.estimated
