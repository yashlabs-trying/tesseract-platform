#!/usr/bin/env bash
# Foundry, Fusion and Adaptive are in-tree under products/. Only Ignite
# (kubbernetd-saver) is a separate repository; clone it at the commit pinned in
# platform.manifest.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/src-checkouts"
mkdir -p "$DEST"
SHA="$(python3 -c 'import yaml;print(yaml.safe_load(open("'"$ROOT"'/platform.manifest.yaml"))["products"]["ignite"]["pinned_commit"])')"

if [ ! -d "$DEST/kubbernetd-saver/.git" ]; then
  git clone --filter=blob:none https://github.com/yashlabs-trying/kubbernetd-saver.git "$DEST/kubbernetd-saver"
fi
git -C "$DEST/kubbernetd-saver" fetch --all --tags
git -C "$DEST/kubbernetd-saver" checkout "$SHA"
echo "==> kubbernetd-saver @ $(git -C "$DEST/kubbernetd-saver" rev-parse HEAD)"
