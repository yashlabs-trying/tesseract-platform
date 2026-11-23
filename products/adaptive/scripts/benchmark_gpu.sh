#!/usr/bin/env bash
# GPU benchmark template (Milestone 5).
#
# This script documents the intended GPU measurement workflow. It does NOT
# fabricate results: it requires a real GPU, a real serving engine, and real
# measurements, then imports them through the Foundry contract to build a
# benchmark report.
#
# Usage: scripts/benchmark_gpu.sh gpu/manifests/qwen2.5-0.5b-int4-gptq.yaml
set -euo pipefail

MANIFEST="${1:-}"
if [[ -z "$MANIFEST" ]]; then
  echo "usage: $0 <gpu/manifests/*.yaml>" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "no NVIDIA GPU detected; this script must run on a validated GPU host" >&2
  exit 3
fi

echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> Manifest: $MANIFEST"
python - "$MANIFEST" <<'PY'
import sys
from adaptive.benchmark import load_manifest

manifest = load_manifest(sys.argv[1])
print(f"model:      {manifest.model_id} @ {manifest.model_revision}")
print(f"profile:    {manifest.profile.bits}-bit {manifest.profile.method}")
print(f"hardware:   {manifest.hardware.gpu_model} x{manifest.hardware.gpu_count}")
print(f"workload:   c={manifest.workload.concurrency} "
      f"in={manifest.workload.input_tokens} out={manifest.workload.output_tokens}")
print(f"thresholds: ttft_p95<={manifest.thresholds.ttft_p95_ms}ms "
      f"itl_p95<={manifest.thresholds.itl_p95_ms}ms "
      f"tps>={manifest.thresholds.min_throughput_tps}")
PY

cat <<'NOTE'

==> Next steps (run manually on this GPU host):
  1. Quantize/prepare the model with the manifest's backend (e.g. vLLM + GPTQ,
     or bitsandbytes). Use the pinned revision and calibration dataset.
  2. Serve it and benchmark the pinned workload (concurrency, token lengths).
  3. Emit a tesseract.performance-profile.v1 JSON with the measured metrics and
     evidence_type=measured.
  4. Build the report:
       adaptive benchmark measured.json --out reports/<model>-<profile>
  5. Import the report into Foundry for cross-product comparison.

Do not promote any profile to production until its report is measured/validated
and a rollback has been exercised (see docs/GPU_VALIDATION_PLAN.md).
NOTE
