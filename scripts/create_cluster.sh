#!/usr/bin/env bash
# Create a local kind cluster for Mac M5 validation (ARM64).
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-tesseract}"
CONFIG="$(dirname "$0")/../deployments/kind/cluster.yaml"

echo "==> Creating kind cluster '$CLUSTER_NAME'"
if kind get clusters | grep -qx "$CLUSTER_NAME"; then
  echo "cluster already exists"
else
  kind create cluster --name "$CLUSTER_NAME" --config "$CONFIG"
fi
kubectl cluster-info --context "kind-$CLUSTER_NAME"
