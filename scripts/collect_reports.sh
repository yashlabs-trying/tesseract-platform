#!/usr/bin/env bash
# Run the local benchmark harness and collect JSON + Markdown reports.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
SCENARIO="${1:-benchmarks/scenarios/local-lifecycle.json}"

echo "==> Benchmark: $SCENARIO"
$PY benchmarks/runner.py --scenario "$SCENARIO" --out benchmarks/reports

echo "==> Reports:"
ls -1 benchmarks/reports
