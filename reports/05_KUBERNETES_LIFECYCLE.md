# 05 — Kubernetes Lifecycle

**Status: NOT TESTED.** No Kubernetes cluster is available on the validation
host, and the Mac target has no container runtime (see
`docs/MAC_SETUP_COMMANDS.md`). No bypass of branch checks was attempted.

## Why not tested

* WSL2 host: no Docker/kind/kubectl/helm; `sudo` requires a password.
* Mac `192.168.1.4`: no Homebrew, no Docker/colima, no kubectl/kind/helm; `sudo`
  requires a password.
* Per campaign decision, no risky sudo workarounds or silent alternative
  runtimes were used.

## Ignite PR #1

`yashlabs-trying/kubbernetd-saver` PR #1 (`fix/code-issues` → `main`) is **still
open**. `main` is protected by a required **`k8s-e2e`** status check, which needs
a real cluster. The three validated commits
(`ddb01e8`, `f6d131c`, `df22730`) remain safe on `origin/fix/code-issues`. The
check must not be bypassed; it will be satisfied when Phase E runs on the Mac.

## What was prepared

* `deployments/kind/cluster.yaml` (multi-node, ARM64-ready).
* `deployments/kubernetes/{namespace,rbac,platform}.yaml` — least-privilege RBAC
  (ReplicaGroups + Deployments in-namespace only).
* `deployments/kubernetes/gpu-device-plugin.yaml` (GPU-only, clearly labelled).
* `deployments/helm/tesseract-platform/` (chart + values + template).
* `scripts/{setup_mac,create_cluster,deploy_stack,cleanup}.sh`.
* `scripts/run_full_stack_validation.sh` runs `-m kubernetes` automatically when
  a cluster is reachable.
* The platform's Ignite adapter creates real `ReplicaGroup` + `Deployment`
  objects and polls `status.phase`/`readyReplicas` (readiness is not inferred
  from object creation).

## Required test list (to run on the Mac)

1. Fresh installation. 2. Repeated installation. 3. Config updates.
4. Controller startup. 5. CRD registration. 6. RBAC permissions.
7. Service discovery. 8. Pod readiness. 9. Deployment scaling.
10. Scale-to-zero. 11. Cold wake. 12. Proxy routing. 13. Multi-model
deployments. 14. Rolling restart. 15. Complete uninstall/cleanup.

## Exact remaining execution steps

```bash
# On the Mac, after docs/MAC_SETUP_COMMANDS.md steps 1–4:
git clone https://github.com/yashlabs-trying/tesseract-platform
cd tesseract-platform
scripts/setup_mac.sh
scripts/create_cluster.sh
scripts/deploy_stack.sh
scripts/run_full_stack_validation.sh --keep-cluster
# copy reports/SCORECARD.md and reports/raw/k8s_pytest.log back
```

## Verdict

**NOT TESTED.** A prepared harness is not a completed hardware/cluster test.
