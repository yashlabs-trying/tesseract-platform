# Mac M5 — Exact Setup & Verification Commands

The Mac at `192.168.1.4` (arm64, macOS 26.6.2) is reachable over SSH but has
**no Homebrew, no container runtime, no kubectl/kind/helm, and `sudo` requires a
password**. These commands must be run **by the operator** on the Mac (they need
the sudo password; do not share it).

## 1. Homebrew (one-time)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Add brew to PATH for Apple Silicon (zsh):
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew --version
```

## 2. Container runtime — choose ONE

**Option A — Docker Desktop (GUI, easiest):**

```bash
brew install --cask docker
open -a Docker          # launch and complete first-run prompts
docker version          # must show a running daemon
```

**Option B — Colima (CLI, lighter, no Docker Desktop license):**

```bash
brew install colima docker
colima start --cpu 4 --memory 8 --arch aarch64
docker version
```

## 3. Kubernetes tooling

```bash
brew install kind kubectl helm
kind version && kubectl version --client && helm version
```

## 4. Verify the environment

```bash
uname -m                 # expect: arm64
docker info | grep -E 'Architecture|CPUs|Total Memory'
kubectl config current-context 2>/dev/null || echo "no cluster yet"
```

## 5. Create the cluster and validate

```bash
# From the platform repo root on the Mac:
scripts/setup_mac.sh                 # venv + deps + contract validation
scripts/create_cluster.sh            # kind cluster (ARM64)
scripts/deploy_stack.sh              # apply namespace/rbac/platform + Ignite CRD
kubectl -n tesseract get pods -w     # wait for platform ready
scripts/run_full_stack_validation.sh --keep-cluster
```

## 6. Expected results

* `kubernetes lifecycle` moves from **NOT TESTED → PASS/FAIL**.
* `GPU validation` remains **NOT TESTED** (no NVIDIA hardware).

## Notes

* Do not run `sudo` workarounds or alternative runtimes silently; if Docker
  Desktop/Colima cannot be installed, Phase E stays NOT TESTED.
* After a successful run, capture `reports/SCORECARD.md` and
  `reports/raw/k8s_pytest.log` and copy them back for the final report.
