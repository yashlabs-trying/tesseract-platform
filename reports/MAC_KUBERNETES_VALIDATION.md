# Mac M5 Kubernetes Validation

**Status: NOT TESTED.** The Mac at `192.168.1.4` (arm64, macOS 26.6.2) is
reachable over SSH but remains unprovisioned: no Homebrew, no Docker/colima, no
kubectl/kind/helm, and `sudo` requires a password. Per policy, no sudo
workarounds were attempted.

## Verified environment (this run)
```
arch: arm64
os:   26.6.2
brew: none
docker/kind/kubectl/helm: none
python3: 3.9.6
mem: 16 GB
disk: 343 GB free
sudo: requires password
```

## Exact operator commands
See `docs/MAC_SETUP_COMMANDS.md`. Summary:

```bash
# 1. Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile && eval "$(/opt/homebrew/bin/brew shellenv)"

# 2. Container runtime (choose one)
brew install --cask docker && open -a Docker          # Docker Desktop
# or: brew install colima docker && colima start --cpu 4 --memory 8 --arch aarch64

# 3. Kubernetes tooling
brew install kind kubectl helm

# 4. Validate
git clone https://github.com/yashlabs-trying/tesseract-platform
cd tesseract-platform
scripts/setup_mac.sh
scripts/create_cluster.sh
scripts/deploy_stack.sh
scripts/run_full_stack_validation.sh --keep-cluster
```

## What is ready for the Mac
* `deployments/kind/cluster.yaml` (ARM64), `deployments/kubernetes/*`,
  `deployments/helm/tesseract-platform/*`.
* `scripts/{setup_mac,create_cluster,deploy_stack,run_tests,collect_reports,cleanup}.sh`.
* `scripts/run_full_stack_validation.sh` runs `-m kubernetes` automatically when a
  cluster is reachable.

## Gate
`MAC M5 KUBERNETES VALIDATION = NOT TESTED`. A prepared harness is not a
completed cluster test. No result is inferred from the WSL2 CPU runs.
