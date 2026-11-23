# Tesseract

**A control plane for running LLM inference on GPUs at the lowest cost that
still meets quality and latency targets.** It takes a model and a workload,
measures it, decides where it should run and at what precision, deploys it,
scale-to-zero included, and keeps the books on what each decision costs.

```
customer request
   │
   ▼
PLATFORM ── profile ──► FOUNDRY   measure the model on candidate GPUs, price it
   │      ── place ────► FUSION    pick the cheapest feasible GPU placement, reserve it
   │      ── precision ► ADAPTIVE  choose a quantization profile that passes quality gates (optional)
   │      ── deploy ───► IGNITE    deploy on Kubernetes, scale to zero, wake on request
   ▼
model server ── telemetry ──► FOUNDRY ──► controller: observe → recommend → act / roll back
```

| Component | What it does | Where | Tests |
|---|---|---|---|
| **Platform** | Workload lifecycle: durable reservation state machine, coordinator, compensating rollback, telemetry loop, REST API | [`tesseract_platform/`](tesseract_platform/) | 81 |
| **Foundry** | Turns a model + hardware + workload into candidate deployments, benchmarks or simulates them, computes unit economics and a launch recommendation | [`products/foundry/`](products/foundry/) | 47 |
| **Fusion** | GPU fleet economics: inventory, demand forecast, constraint-checked placement optimizer, simulation | [`products/fusion/`](products/fusion/) | 183 |
| **Adaptive** | Quality-constrained mixed-precision search: per-layer sensitivity, Pareto optimizer over 4/8-bit profiles, safe runtime switching with rollback | [`products/adaptive/`](products/adaptive/) | 254 |
| **Ignite** | Kubernetes scale-to-zero for model servers with a buffering proxy | [kubbernetd-saver](https://github.com/yashlabs-trying/kubbernetd-saver) | 77 + 9 kind E2E |

All suites run on CPU in CI (no GPU needed). The control plane talks to every
product only through the versioned JSON-Schema contracts in
[`contracts/`](contracts/), so each product can be replaced or run alone.

## Status

- **Done:** contracts, coordinator, reservation lifecycle, rollback, telemetry,
  and the four products, all covered by unit, contract, property-based and
  in-process integration tests; Ignite's scale-to-zero path is exercised on a
  real kind cluster in CI.
- **Not yet done:** a measured end-to-end run on real GPUs. Foundry's numbers
  in fixture mode are synthetic and labelled as such: its recommender refuses
  to launch on unverified evidence. The plan for that run is in
  [`docs/GPU_VALIDATION.md`](docs/GPU_VALIDATION.md).

## Quick start

```bash
pip install -e ".[test]"
pip install -e products/foundry -e products/fusion -e products/adaptive
pytest                                   # control plane

foundry economics products/foundry/fixtures/scenario-1-single-gpu.yaml
foundry recommend products/foundry/fixtures/scenario-1-single-gpu.yaml
fusion optimize products/fusion/fixtures/fleet.yaml
adaptive analyze products/adaptive/fixtures/models/llama-3.2-1b.json

tesseract-platform serve                 # control-plane API
```

## Layout

```
tesseract_platform/   control plane: coordinator, state machine, store, policies, API
integrations/         adapters for foundry, fusion, ignite, adaptive
contracts/            JSON-Schema contracts shared across products
products/             foundry, fusion, adaptive (each its own package and tests)
deployments/          docker-compose, Kubernetes, Helm, kind
benchmarks/           control-plane benchmark scenarios and baselines
validation/           CPU validation campaign scripts
docs/                 architecture, contracts, operations, validation plans
reports/              validation reports and raw logs
```

The control-plane package is `tesseract_platform/`, not `platform/`, because a
top-level package named `platform` would shadow Python's standard library.

## Design principles

- **Contracts, not imports.** Products are developed together here, but the
  control plane depends only on contracts, never on product internals.
- **Durable state machine** spanning Fusion and Kubernetes instead of one
  database transaction; every failure has an explicit compensating action.
- **Evidence before claims.** Synthetic or estimated numbers are labelled and
  cannot drive a launch decision.
- **Adaptive is optional.** Contract tests prove the platform works without it.

## License

MIT © Yashvardhan Singh Shekhawat
