# Milestone 3 & 4 Report - Adaptation and Integration

Date: 2026-10-13
Host: Ubuntu (WSL2) x86_64, CPU-only PyTorch 2.14.1+cpu, Python 3.12.14.
No CUDA used. Primary target remains Apple Silicon M5.

## Files created

Source (`src/adaptive/`):
- `models/controller.py` - telemetry snapshot, quality certificate, profile
  option, controller policy, switch/rollback plans, control decision,
  simulation result.
- `telemetry/sources.py`, `telemetry/__init__.py` - telemetry sources
  (constant, scripted) and deterministic scenarios (steady, memory pressure,
  traffic burst).
- `controller/controller.py`, `controller/simulation.py`,
  `controller/__init__.py` - adaptive controller and timeline simulation.
- `integration/foundry.py` - import `tesseract.performance-profile.v1`
  measurements, export profile comparisons with confidence/provenance.
- `integration/fusion.py` - memory requirements, placement recommendation,
  capacity projection.
- `integration/ignite.py` - prevalidated profile loading, safe replica
  transitions, readiness/rollback signals.
- `integration/__init__.py`.
- `schemas.py` - telemetry/decision/simulation serialization.
- `cli/main.py` - `controller` command.

Schemas: `schemas/tesseract/{telemetry-snapshot,control-decision,controller-simulation,profile-comparison,placement-recommendation,ignite-profile-load}-v1.json`

Fixtures: `fixtures/foundry/performance-profile-example.json`

Tests: `tests/unit/{test_controller,test_integration}.py`,
`tests/property/test_controller_properties.py`,
`tests/contract/{test_m3_schema_contract,test_m4_schema_contract}.py`,
`tests/integration/test_end_to_end_mac.py`, `tests/unit/test_cli.py` (extended).

## Features implemented

**M3 - Adaptation**

- **Telemetry** with measured/simulated provenance and three deterministic
  scenarios.
- **Controller** consuming memory pressure, queue depth, TTFT, ITL, throughput
  and demand; emitting downshift/upshift/rollback/hold with projected memory,
  cost and quality impact, a replica-based switch plan and a rollback plan.
- **Anti-oscillation**: hysteresis dead-band, minimum dwell time, cooldown,
  per-hour rate limit, and emergency bypass only for critical pressure.
- **Rollback** for a recent failing *upgrade*; downshifts under pressure are
  never rolled back (they are already the safer direction).
- **Recommendation-only by default**; automatic switching requires
  `enable_automatic(validated=True)` and never performs in-place weight
  replacement.
- **Simulation + fault injection** with switch-failure injection.

**M4 - Integration**

- **Foundry**: import measurements, convert to `InferenceCost`, export
  comparisons with per-profile confidence and provenance.
- **Fusion**: `report_memory_requirements`, `project_capacity`,
  `recommend_precision_for_placement` (highest precision that fits and meets
  SLOs) with capacity-improvement projection.
- **Ignite**: `InProcessIgnite` refusing unvalidated profiles, certificate
  acceptance, safe rolling replica transitions (`in_place=False`,
  `max_unavailable=0`), readiness/rollback signals.

## Tests

Command: `pytest tests/ -q` plus `ruff check src tests`, `mypy src`.

- Passed: **161** (was 119 at M2)
- Failed: **0**; Skipped: **0**
- Ruff: clean. Mypy (strict): clean, 56 source files.
- Coverage: ~93%.

M3/M4 acceptance areas covered: controller response to memory pressure and
traffic bursts, hysteresis and oscillation prevention, safe rollback, profile
switching simulation, fault injection and recovery, Foundry measurement
import/export, Fusion integration contracts, Ignite integration contracts, and
an end-to-end Mac pipeline test spanning M1-M4.

## Actual controller behaviour (tiny-llama, synthetic scenarios)

`adaptive controller fixtures/models/tiny-llama.json --scenario <s>`:

| scenario | switches | rollbacks | oscillated |
| --- | ---: | ---: | --- |
| steady | 0 | 0 | False |
| pressure | 2 | 0 | False |
| burst | 2 | 0 | False |

The pressure scenario downshifts to 4-bit under a spike, holds while pressure
persists, then upshifts back to 16-bit when headroom returns - no oscillation.
Fault injection with a failed switch produces exactly one rollback.

## Integration behaviour (contract examples)

- Foundry import of the example profile: measured, TTFT p95 71 ms, throughput
  1850 tps, provenance `foundry`.
- Fusion placement: 24 GB GPU, 10% headroom, 50 ms SLO -> 8-bit recommended
  (16-bit violates SLO), 21 replicas/GPU, capacity improvement > 1x.
- Ignite: unvalidated profile rejected; prevalidated/certified profile accepted
  with a rolling transition and readiness/rollback signals.

## Limitations

- Telemetry is simulated; no real GPU/runtime telemetry was collected.
- Controller decisions use estimated costs/quality (from M2), not measured
  production data.
- Integration adapters are in-process contract translations; no live HTTP
  clients to Foundry/Fusion/Ignite are enabled.
- Automatic switching is exercised only in simulation, never against a live
  deployment.
- `capacity_improvement` uses the largest candidate as the baseline.

## Bugs found and fixed during M3/M4

1. Rollback originally triggered on any recent switch while pressure remained
   high, causing downshifts under pressure to immediately revert. Fixed to
   only roll back failing *upgrades*.
2. `evaluate_rollback` needed a precision comparison to distinguish upgrades
   from downshifts; added.
3. Property test assumed all high pressure bypasses dwell; emergency threshold
   is `high + 2*hysteresis`, corrected the test to remove dwell instead.

## Remaining GPU-only validation

Unchanged from `docs/GPU_VALIDATION_PLAN.md`: production kernels, real
TTFT/ITL/throughput/VRAM, multi-GPU placement, and real-model quality on GPU
runtimes. M3 telemetry and M4 integration must be exercised against real
Foundry/Fusion/Ignite services and a live runtime before automatic switching is
enabled in production.

## Next recommended engineering action

Begin **Milestone 5**: implement production backend adapters (GPTQ/AWQ/
bitsandbytes/FP8) behind the existing profile interface, add GPU-specific test
manifests and benchmark scripts, and wire Foundry measured-cost import into the
optimizer and controller. Keep GPU claims unvalidated until measured.
