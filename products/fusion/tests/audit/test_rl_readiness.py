"""RL-readiness audit for Tesseract Fusion.

EXPECTED FINDING: Fusion does **not** implement reinforcement learning.

There is no RL environment, no policy checkpoint, and no training loop.  The
optimizer is a deterministic, transparent candidate evaluator
(``docs/architecture.md``: "Deterministic optimization - Transparent,
reproducible candidate evaluation").  These tests assert the *absence* of RL
and the *presence* of a deterministic, advisory-only fallback so that any future
RL work starts from a documented baseline.

The scan deliberately looks for RL *framework imports* and *training* idioms
rather than the bare words "policy", "reward", "env" or "agent", which appear
throughout a legitimate policy/optimizer codebase.
"""
from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

from fusion.models.types import GPUInventory, HealthStatus, ModelDeployment, TenantPolicy
from fusion.optimizer.candidate import CandidateGenerator
from fusion.optimizer.engine import OptimizationEngine
from fusion.optimizer.global_allocator import GlobalAllocator


def make_gpu(gpu_id: str, slots: int = 4) -> GPUInventory:
    return GPUInventory(
        gpu_id=gpu_id,
        node_id=f"node-{gpu_id}",
        gpu_model="H100",
        memory_capacity_mb=81920,
        available_memory_mb=81920,
        available_resource_slots=slots,
        health_status=HealthStatus.HEALTHY,
        resource_cost_per_hour=Decimal("2.0"),
    )


def make_deployment(deployment_id: str, tenant_id: str) -> ModelDeployment:
    return ModelDeployment(
        deployment_id=deployment_id,
        tenant_id=tenant_id,
        model_id="m1",
        worker_count=1,
        expected_memory_mb=40960,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

#: Import statements that would indicate an RL framework is present.
RL_IMPORT_PATTERNS = (
    r"^\s*import\s+gym\b",
    r"^\s*import\s+gymnasium\b",
    r"^\s*from\s+gym\b",
    r"^\s*from\s+gymnasium\b",
    r"^\s*(?:import|from)\s+stable_baselines",
    r"^\s*(?:import|from)\s+torch\b",
    r"^\s*(?:import|from)\s+tensorflow\b",
    r"^\s*(?:import|from)\s+ray\.rllib",
    r"^\s*(?:import|from)\s+pettingzoo",
    r"^\s*(?:import|from)\s+mlagents",
)

#: Idioms that would indicate a training loop.
TRAINING_PATTERNS = (
    r"\.backward\s*\(",
    r"torch\.optim",
    r"optimizer\.step\s*\(",
    r"\.learn\s*\(",
    r"training_step",
    r"train_loop",
    r"experience_replay",
    r"replay_buffer",
    r"q_network",
    r"policy_network",
    r"actor_critic",
)

#: File suffixes that would indicate serialized policy weights.
CHECKPOINT_SUFFIXES = (
    ".pt", ".pth", ".pkl", ".ckpt", ".h5", ".safetensors", ".onnx",
)

_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        yield path


def _find_pattern(root: Path, patterns: tuple[str, ...]):
    """Return ``(relative_path, line_no, line)`` for every matching line."""
    matches = []
    this_file = Path(__file__).resolve()
    for path in _iter_python_files(root):
        if path.resolve() == this_file:
            # Skip this file: it contains the patterns themselves as literals.
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if any(re.search(pattern, line) for pattern in patterns):
                matches.append((str(path.relative_to(REPO_ROOT)), line_no, line.strip()))
    return matches


# --------------------------------------------------------------------------- #
# 1. No RL environment
# --------------------------------------------------------------------------- #
def test_no_rl_environment_present():
    """No gym / RL framework is imported anywhere in the codebase."""
    hits = _find_pattern(REPO_ROOT, RL_IMPORT_PATTERNS)
    assert hits == [], f"RL framework imports found: {hits}"

    # Also assert no ``gym.Env`` style environment subclass exists.
    env_defs = []
    env_re = re.compile(r"class\s+\w+\s*\(\s*(?:gym|gymnasium)\.Env\s*\)")
    for path in _iter_python_files(REPO_ROOT):
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover
            continue
        if env_re.search(text):
            env_defs.append(str(path.relative_to(REPO_ROOT)))
    assert env_defs == [], f"RL environment classes found: {env_defs}"


# --------------------------------------------------------------------------- #
# 2. No policy checkpoints
# --------------------------------------------------------------------------- #
def test_no_policy_checkpoints():
    """No serialized policy/model weight files exist in the repository."""
    checkpoints = []
    for path in REPO_ROOT.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES:
            checkpoints.append(str(path.relative_to(REPO_ROOT)))
    assert checkpoints == [], f"policy checkpoint files found: {checkpoints}"


# --------------------------------------------------------------------------- #
# 3. No training code
# --------------------------------------------------------------------------- #
def test_no_training_code():
    """No training loops / gradient updates exist in the codebase."""
    hits = _find_pattern(REPO_ROOT, TRAINING_PATTERNS)
    assert hits == [], f"training code found: {hits}"


# --------------------------------------------------------------------------- #
# 4. Deterministic baseline exists
# --------------------------------------------------------------------------- #
def test_deterministic_baseline_exists():
    """A deterministic optimizer exists as the RL fallback."""
    engine = OptimizationEngine()
    deployments = [
        make_deployment("d1", "t1"),
        make_deployment("d2", "t2"),
    ]
    gpus = [make_gpu("g0", slots=4), make_gpu("g1", slots=4)]
    policies = {
        "t1": TenantPolicy(tenant_id="t1"),
        "t2": TenantPolicy(tenant_id="t2"),
    }

    first = engine.optimize(deployments, gpus, policies, {}, {})
    second = engine.optimize(deployments, gpus, policies, {}, {})

    assert first.provenance.get("engine") == "deterministic_candidate_fixed"
    # Same input -> same plan (reproducible).
    assert [a.gpu_pool_id for a in first.recommended_plan] == [
        a.gpu_pool_id for a in second.recommended_plan
    ]
    # The deterministic optimizer is advisory-only.
    assert first.approval_status.value == "advisory_only"
    assert first.execution_eligibility is False


# --------------------------------------------------------------------------- #
# 5. Safe fallback available
# --------------------------------------------------------------------------- #
def test_safe_fallback_available(monkeypatch):
    """The GlobalAllocator greedy fallback works when or-tools is absent."""
    import fusion.optimizer.global_allocator as global_allocator_mod

    alloc = GlobalAllocator()
    gpus = [make_gpu("g0", slots=4), make_gpu("g1", slots=4)]
    gpu_map = {g.gpu_id: g for g in gpus}
    deployments = [make_deployment(f"d{i}", f"t{i}") for i in range(12)]
    policies = {f"t{i}": TenantPolicy(tenant_id=f"t{i}") for i in range(12)}
    all_alts = {
        d.deployment_id: CandidateGenerator().generate_alternatives(d, gpus)
        for d in deployments
    }

    rec = alloc._greedy_allocate(
        deployments, gpus, gpu_map, policies, {}, all_alts, []
    )
    assert rec.provenance.get("engine") == "greedy_global"
    assert rec.approval_status.value == "advisory_only"
    assert rec.execution_eligibility is False

    # Force the no-or-tools path through the public entrypoint as well.
    monkeypatch.setattr(global_allocator_mod, "_has_ortools", lambda: False)
    fallback = alloc.allocate(deployments, gpus, policies, {}, {})
    assert fallback.provenance.get("engine") == "greedy_global"
