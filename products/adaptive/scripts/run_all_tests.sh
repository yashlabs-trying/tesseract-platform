#!/usr/bin/env bash
# Unified test/validation runner for Tesseract Adaptive.
#
# Runs lint, type checks, the full test suite, and optional security audits.
# GPU-marked tests are skipped automatically without CUDA.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

if [[ ! -d .venv ]]; then
  echo "==> Creating virtualenv with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -e ".[dev]"
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> Ruff"
ruff check src tests

echo "==> Mypy (strict)"
mypy src

echo "==> Pytest (full, coverage)"
pytest tests/ -q --cov=adaptive --cov-report=term

echo "==> Security audits (optional)"
if python -c "import bandit" 2>/dev/null; then
  bandit -q -r src -x tests || true
else
  echo "bandit not installed; skipping"
fi
if command -v pip-audit >/dev/null 2>&1; then
  pip-audit || true
else
  echo "pip-audit not installed; skipping"
fi

echo "All checks completed."
