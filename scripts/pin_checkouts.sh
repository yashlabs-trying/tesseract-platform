#!/usr/bin/env bash
# Pin product source checkouts at the revisions recorded in platform.manifest.yaml.
# Read-only clones; the platform never modifies product repositories.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/src-checkouts"
mkdir -p "$DEST"

clone_pin() {
  local name="$1" url="$2" sha="$3"
  if [ -d "$DEST/$name/.git" ]; then
    echo "==> $name already present"
  else
    echo "==> Cloning $name"
    git clone --filter=blob:none "$url" "$DEST/$name"
  fi
  git -C "$DEST/$name" fetch --all --tags
  git -C "$DEST/$name" checkout "$sha"
  echo "    $name @ $(git -C "$DEST/$name" rev-parse HEAD)"
}

clone_pin tesseract-foundry https://github.com/yashlabs-trying/tesseract-foundry.git b8a70306d2c5d40b73156593987b3c8fab10cd29
clone_pin tesseract-fusion  https://github.com/yashlabs-trying/tesseract-fusion.git  c0f4eb9838c6203e185449989a932475a37ee827
clone_pin kubbernetd-saver  https://github.com/yashlabs-trying/kubbernetd-saver.git  df227307a916a1ec3d3fadab9a28cb2cdb14e911
clone_pin tesseract-adaptive https://github.com/yashlabs-trying/tesseract-adaptive.git 06e2bf37861323f6a52fe20671992c45599590ff

echo "==> Pinned checkouts in $DEST"
