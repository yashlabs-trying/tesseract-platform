#!/usr/bin/env bash
# Set up an Apple Silicon (M5) Mac for local Tesseract validation.
# Installs ARM64 tooling via Homebrew. Idempotent.
set -euo pipefail

echo "==> Tesseract Platform: Mac M5 setup"
echo "arch: $(uname -m)  os: $(sw_vers -productVersion 2>/dev/null || uname -s)"

if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew is required (https://brew.sh)." >&2
  exit 1
fi

echo "==> Installing tooling (kind, kubectl, helm, docker)"
brew list kind >/dev/null 2>&1 || brew install kind
brew list kubectl >/dev/null 2>&1 || brew install kubectl
brew list helm >/dev/null 2>&1 || brew install helm
if ! command -v docker >/dev/null 2>&1; then
  echo "WARN: Docker not found. Install Docker Desktop (Apple Silicon) or colima." >&2
fi

echo "==> Creating a Python virtualenv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"

echo "==> Validating contracts"
python -m tesseract_platform.cli validate-contracts

echo "==> Done. Next: scripts/create_cluster.sh"
