#!/usr/bin/env bash
# Tear down the local Tesseract deployment and (optionally) the cluster.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAMESPACE="${NAMESPACE:-tesseract}"
CLUSTER_NAME="${CLUSTER_NAME:-tesseract}"
DELETE_CLUSTER=0
for arg in "$@"; do [ "$arg" = "--cluster" ] && DELETE_CLUSTER=1; done

echo "==> Deleting platform resources in namespace '$NAMESPACE'"
kubectl delete -f "$ROOT/deployments/kubernetes/platform.yaml" --ignore-not-found || true
kubectl delete namespace "$NAMESPACE" --ignore-not-found || true

if [ "$DELETE_CLUSTER" = "1" ]; then
  echo "==> Deleting kind cluster '$CLUSTER_NAME'"
  kind delete cluster --name "$CLUSTER_NAME" || true
fi

echo "==> Cleanup complete"
