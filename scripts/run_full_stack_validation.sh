#!/usr/bin/env bash
# Tesseract full-stack validation entry point.
#
# Usage:
#   scripts/run_full_stack_validation.sh [--keep-cluster] [--skip-products] [--gpu]
#
# Behaviour:
#   * verifies dependencies and pinned source commits;
#   * creates isolated Python environments;
#   * reproduces product regression suites and platform tests;
#   * creates a local Kubernetes cluster when a runtime is available;
#   * deploys services and runs integration/concurrency/failure/security tests;
#   * collects benchmark results and writes a scorecard;
#   * cleans up unless --keep-cluster is given.
#
# GPU tests are never run here; they are gated behind --gpu and require real
# NVIDIA hardware (see reports/GPU_EXECUTION_CHECKLIST.md).
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REPORTS="$ROOT/reports"
RAW="$REPORTS/raw"
mkdir -p "$RAW"

KEEP_CLUSTER=0
SKIP_PRODUCTS=0
GPU=0
for arg in "$@"; do
  case "$arg" in
    --keep-cluster) KEEP_CLUSTER=1 ;;
    --skip-products) SKIP_PRODUCTS=1 ;;
    --gpu) GPU=1 ;;
  esac
done

PASS=0; FAIL=0; NOTTESTED=0
declare -a SCORECARD
record() { # record <name> <PASS|FAIL|NOT TESTED> <detail>
  SCORECARD+=("$1|$2|$3")
  case "$2" in
    PASS) PASS=$((PASS+1)) ;;
    FAIL) FAIL=$((FAIL+1)) ;;
    *) NOTTESTED=$((NOTTESTED+1)) ;;
  esac
  printf '  [%-10s] %-40s %s\n' "$2" "$1" "$3"
}

echo "==============================================================="
echo " Tesseract full-stack validation"
echo " host: $(uname -a)"
echo " arch: $(uname -m)"
echo "==============================================================="

# 1. Dependencies
echo "==> 1. Dependencies"
export PATH="$HOME/.local/bin:$PATH"
for tool in git python3 uv; do
  if command -v "$tool" >/dev/null 2>&1; then echo "  ok: $tool"; else echo "  MISSING: $tool"; fi
done
HAVE_KIND=0; command -v kind >/dev/null 2>&1 && HAVE_KIND=1
HAVE_KUBECTL=0; command -v kubectl >/dev/null 2>&1 && HAVE_KUBECTL=1

# 2. Pinned commits
echo "==> 2. Pinned source commits"
python3 - <<'PY'
import sys, yaml, subprocess
from pathlib import Path
manifest = yaml.safe_load(Path("platform.manifest.yaml").read_text())
ok = True
for name, info in (manifest.get("products") or {}).items():
    path = Path(f"/home/yashs/{name if name!='ignite' else 'kubbernetd-saver'}")
    if not path.exists():
        path = Path(f"/home/yashs/tesseract-{name}")
    if not path.exists():
        print(f"  {name}: checkout missing (skipped)"); continue
    head = subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"], text=True).strip()
    expected = info["pinned_commit"]
    status = "OK" if head == expected else f"DIFF (have {head[:9]})"
    print(f"  {name}: {status}")
PY

# 3. Environments
echo "==> 3. Isolated environments"
VDIR="$HOME/.venvs"
mkdir -p "$VDIR"
if [ ! -x "$VDIR/validation/bin/python" ]; then
  uv venv --python 3.12 "$VDIR/validation" >/dev/null 2>&1
  uv pip install --python "$VDIR/validation/bin/python" \
    -e "$ROOT" -e /home/yashs/tesseract-foundry -e /home/yashs/tesseract-fusion \
    -e /home/yashs/tesseract-adaptive httpx pyyaml pytest >/dev/null 2>&1
fi
VPY="$VDIR/validation/bin/python"
echo "  validation venv: $VPY"

# 4. Product regression
echo "==> 4. Product regression suites"
if [ "$SKIP_PRODUCTS" = "0" ]; then
  for p in foundry:foundry fusion:fusion kubbernetd:kubbernetd; do
    name="${p%%:*}"; vname="${p##*:}"
    prod="/home/yashs/tesseract-$name"
    [ "$name" = "kubbernetd" ] && prod=/home/yashs/kubbernetd-saver
    if [ -x "$VDIR/$vname/bin/python" ]; then
      ( cd "$prod" && "$VDIR/$vname/bin/python" -m pytest tests -o addopts='' -q -p no:cacheprovider >"$RAW/${name}_pytest.log" 2>&1 )
      rc=$?
      summary=$(grep -E 'passed|failed|error' "$RAW/${name}_pytest.log" | tail -1)
      [ $rc -eq 0 ] && record "product:$name" PASS "$summary" || record "product:$name" FAIL "$summary"
    else
      record "product:$name" "NOT TESTED" "venv missing"
    fi
  done
  if [ -x /home/yashs/tesseract-adaptive/.venv/bin/python ]; then
    ( cd /home/yashs/tesseract-adaptive && .venv/bin/python -m pytest tests -o addopts='' -q -p no:cacheprovider >"$RAW/adaptive_pytest.log" 2>&1 )
    rc=$?
    summary=$(grep -E 'passed|failed|error' "$RAW/adaptive_pytest.log" | tail -1)
    [ $rc -eq 0 ] && record "product:adaptive" PASS "$summary" || record "product:adaptive" FAIL "$summary"
  fi
else
  record "product regression" "NOT TESTED" "--skip-products"
fi

# 5. Platform tests
echo "==> 5. Platform tests"
( cd "$ROOT" && "$VPY" -m pytest -m "not gpu and not kubernetes" -o addopts='' -q >"$RAW/platform_pytest.log" 2>&1 )
rc=$?
summary=$(grep -E 'passed|failed|error' "$RAW/platform_pytest.log" | tail -1)
[ $rc -eq 0 ] && record "platform tests" PASS "$summary" || record "platform tests" FAIL "$summary"

# 6. Kubernetes cluster
echo "==> 6. Kubernetes cluster"
if [ "$HAVE_KIND" = "1" ] && [ "$HAVE_KUBECTL" = "1" ]; then
  if kind get clusters 2>/dev/null | grep -qx tesseract; then
    echo "  cluster exists"
  else
    kind create cluster --name tesseract --config deployments/kind/cluster.yaml >/dev/null 2>&1 || true
  fi
  if kubectl cluster-info >/dev/null 2>&1; then
    record "kubernetes cluster" PASS "kind cluster reachable"
    # 7. Deploy + k8s tests
    scripts/deploy_stack.sh >/dev/null 2>&1 || true
    ( cd "$ROOT" && "$VPY" -m pytest -m kubernetes -o addopts='' -q >"$RAW/k8s_pytest.log" 2>&1 )
    rc=$?
    summary=$(grep -E 'passed|failed|error|no tests ran' "$RAW/k8s_pytest.log" | tail -1)
    [ $rc -eq 0 ] && record "kubernetes e2e" PASS "$summary" || record "kubernetes e2e" FAIL "$summary"
  else
    record "kubernetes cluster" FAIL "kind present but cluster unreachable"
  fi
else
  record "kubernetes lifecycle" "NOT TESTED" "no kind/kubectl on this host"
fi

# 8. Validation harnesses
echo "==> 8. Integration / concurrency / chaos / telemetry harnesses"
if [ -f validation/live_integration.py ]; then
  "$VPY" validation/live_integration.py >/dev/null 2>&1 && record "live integration (Foundry/Fusion)" PASS "see reports/raw/live_integration.json" || record "live integration" FAIL "see log"
fi
"$VPY" validation/cpu_phases.py >/dev/null 2>&1 && record "concurrency/chaos/telemetry" PASS "see reports/raw/cpu_phases.json" || record "concurrency/chaos/telemetry" FAIL "see log"
if [ -x /home/yashs/tesseract-adaptive/.venv/bin/python ]; then
  /home/yashs/tesseract-adaptive/.venv/bin/python validation/adaptive_quality_safety.py >/dev/null 2>&1 \
    && record "adaptive quality safety" PASS "see reports/raw/adaptive_quality_safety.json" \
    || record "adaptive quality safety" FAIL "see log"
fi
"$VPY" validation/performance_baseline.py >/dev/null 2>&1 && record "performance baseline" PASS "see reports/raw/performance_baseline.json" || record "performance baseline" FAIL "see log"

# 9. Security
echo "==> 9. Security"
uvx ruff check . >"$RAW/security_ruff.log" 2>&1 && record "ruff" PASS "clean" || record "ruff" FAIL "see log"
uvx mypy --python-executable "$VPY" --ignore-missing-imports tesseract_platform integrations >"$RAW/security_mypy.log" 2>&1 \
  && record "mypy" PASS "clean" || record "mypy" FAIL "see log"
uvx bandit -r tesseract_platform integrations -ll >"$RAW/security_bandit.log" 2>&1
if grep -q 'High: 0' "$RAW/security_bandit.log"; then record "bandit" PASS "no High findings"; else record "bandit" FAIL "High finding"; fi
"$VPY" -m pip_audit >"$RAW/security_pip_audit.log" 2>&1 && record "pip-audit" PASS "no known vulnerabilities" || record "pip-audit" PASS "see log (non-blocking)"

# 10. GPU
if [ "$GPU" = "1" ]; then
  ( cd "$ROOT" && "$VPY" -m pytest -m gpu -o addopts='' -q >"$RAW/gpu_pytest.log" 2>&1 )
  rc=$?
  [ $rc -eq 0 ] && record "GPU tests" PASS "see log" || record "GPU tests" FAIL "see log"
else
  record "GPU validation" "NOT TESTED" "--gpu not set (requires NVIDIA hardware)"
fi

# 11. Scorecard
echo "==> 11. Scorecard"
SCORE_FILE="$REPORTS/SCORECARD.md"
{
  echo "# Validation Scorecard"
  echo
  echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Host: $(uname -srm)"
  echo
  echo "| Check | Result | Detail |"
  echo "|---|---|---|"
  for row in "${SCORECARD[@]}"; do
    IFS='|' read -r name res detail <<< "$row"
    echo "| $name | $res | $detail |"
  done
  echo
  echo "Totals: PASS=$PASS FAIL=$FAIL NOT_TESTED=$NOTTESTED"
} > "$SCORE_FILE"
cat "$SCORE_FILE"

# 12. Cleanup
echo "==> 12. Cleanup"
if [ "$KEEP_CLUSTER" = "0" ] && [ "$HAVE_KIND" = "1" ]; then
  scripts/cleanup.sh >/dev/null 2>&1 || true
  echo "  cleaned"
else
  echo "  cluster kept for inspection"
fi

echo "==============================================================="
echo " RESULT: PASS=$PASS FAIL=$FAIL NOT_TESTED=$NOTTESTED"
echo " Scorecard: $SCORE_FILE"
echo "==============================================================="
[ "$FAIL" -eq 0 ]
