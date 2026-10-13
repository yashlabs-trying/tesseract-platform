#!/usr/bin/env bash
# Deploy the Tesseract stack locally.
#   --gpu : also apply GPU device-plugin manifests (requires NVIDIA hardware)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAMESPACE="${NAMESPACE:-tesseract}"
GPU=0
for arg in "$@"; do [ "$arg" = "--gpu" ] && GPU=1; done

echo "==> Applying namespace and platform manifests"
kubectl apply -f "$ROOT/deployments/kubernetes/namespace.yaml"
kubectl apply -f "$ROOT/deployments/kubernetes/rbac.yaml"
kubectl apply -f "$ROOT/deployments/kubernetes/platform.yaml"

echo "==> Applying product CRDs (Ignite ReplicaGroup)"
if [ -d "$ROOT/src-checkouts/kubbernetd-saver/deploy" ]; then
  kubectl apply -f "$ROOT/src-checkouts/kubbernetd-saver/deploy/kubbernetd/crd.yaml"
else
  echo "WARN: Ignite checkout not present; skipping CRD. Run scripts/pin_checkouts.sh." >&2
fi

if [ "$GPU" = "1" ]; then
  echo "==> Applying GPU device plugin (requires NVIDIA hardware)"
  kubectl apply -f "$ROOT/deployments/kubernetes/gpu-device-plugin.yaml"
fi

kubectl -n "$NAMESPACE" get pods
echo "==> Deployed. Port-forward with: kubectl -n $NAMESPACE port-forward svc/tesseract-platform 8088:8088"
