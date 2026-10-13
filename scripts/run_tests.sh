#!/usr/bin/env bash
# Run the platform test suite. Kubernetes-marked tests are skipped unless a
# cluster is available; GPU-marked tests are always skipped unless -m gpu.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"

echo "==> Contract validation"
$PY -m tesseract_platform.cli validate-contracts

echo "==> Lint"
if $PY -m ruff --version >/dev/null 2>&1; then
  $PY -m ruff check .
fi

echo "==> Unit + contract + integration tests"
$PY -m pytest -m "not gpu and not kubernetes" -o addopts='' -q

echo "==> Kubernetes tests (skipped if no cluster)"
if kubectl cluster-info >/dev/null 2>&1; then
  $PY -m pytest -m kubernetes -o addopts='' -q
else
  echo "no cluster reachable; skipping -m kubernetes"
fi
