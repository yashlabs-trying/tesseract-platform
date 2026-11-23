#!/usr/bin/env bash
# Reproducible Mac M5 (or any CPU host) test script for Tesseract Adaptive.
# Requires Python 3.11 or 3.12. Uses CPU PyTorch; no CUDA required.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

echo "==> Creating virtualenv with $PYTHON_BIN"
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing CPU PyTorch and package"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

echo "==> Ruff"
ruff check src tests

echo "==> Mypy"
mypy src

echo "==> Pytest"
pytest tests/ -v --tb=short

echo "==> Example"
python examples/demo_analyze.py

echo "All checks passed."
