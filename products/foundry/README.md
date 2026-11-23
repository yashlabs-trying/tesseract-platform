# Tesseract Foundry

Deployment intelligence for GPU inference providers. Given a model, the
available hardware and a workload, Foundry generates candidate deployments
(GPU type, count, parallelism, precision, batching), profiles or simulates
them, computes unit economics, and issues a launch recommendation that refuses
to rely on unverified evidence.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q

foundry validate  fixtures/scenario-1-single-gpu.yaml
foundry economics fixtures/scenario-1-single-gpu.yaml
foundry recommend fixtures/scenario-1-single-gpu.yaml
foundry report <recommendation-id> --format markdown
foundry manifest <recommendation-id> --output deployment.yaml
foundry serve                     # REST API
```

Example `economics` output for the single-GPU scenario (fixture data):

```
--- candidate-... ---
  GPU Cost/Month:   $374.40
  Total Cost/Month: $409.50
  Revenue/Month:    $2,985.98
  Margin:           $2,576.48 (86.3%)
  Cost/1M Input:    $0.0617
  Cost/1M Output:   $0.1234
```

In fixture mode every measurement is synthetic, so `recommend` returns
`unverified_only` and advises against launching: a recommendation needs
evidence measured on real hardware.

## Fixtures

- `fixtures/models/`: Qwen3-0.6B, Llama-3.1-8B, Mixtral-8x22B
- `fixtures/hardware/`: L4, H100, 4×A100
- `fixtures/workloads/`: steady batch, chat, bursty demand
- `fixtures/scenario-*.yaml`: end-to-end scenarios (single GPU, two GPU types,
  bursty demand with negative margin)

Foundry emits the `tesseract.performance-profile.v1` contract consumed by the
Tesseract control plane and by Adaptive.

## License

MIT
