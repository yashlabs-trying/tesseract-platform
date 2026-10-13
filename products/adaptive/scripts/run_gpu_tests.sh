#!/usr/bin/env bash
# Run GPU-only validation tests on an NVIDIA host.
#
# These tests are skipped by default on Mac/CPU development machines. Run this
# script only on a validated GPU host with the production backends installed.
set -euo pipefail

echo "==> GPU host: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
echo "==> Running GPU-marked tests"
pytest tests/gpu -m gpu -v --tb=short

echo "==> Done. Import any produced measurements into Foundry before making claims."
