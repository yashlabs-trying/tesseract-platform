# Fusion — RL Readiness Assessment

## 1. Headline finding

**Reinforcement Learning is NOT implemented in Fusion.**

There is no RL environment, no policy, no training loop, no reward function, and no checkpoints.
Fusion uses a **deterministic, safe fallback** (GlobalAllocator with exhaustive / CP-SAT / greedy
solver paths). This document records what was searched, the current deterministic behavior, a
recommendation on whether RL-family methods are worth implementing, and the prerequisites.

---

## 2. What was searched for

| Artifact | Searched | Found? |
|----------|----------|--------|
| `gym` / `gymnasium` dependency | Yes | **No** |
| `stable_baselines` / `stable-baselines3` | Yes | **No** |
| `torch` / PyTorch | Yes | **No** |
| `tensorflow` | Yes | **No** |
| `Env` classes (RL environment) | Yes | **No** |
| Checkpoints | Yes | **No** |
| Reward function | Yes | **No** |
| Policy (learned) | Yes | **No** |
| Training loops | Yes | **No** |

**Conclusion:** None of the above exist. There is no RL code path in Fusion.

---

## 3. Current deterministic safe fallback

Fusion's allocation/optimization is performed by `GlobalAllocator` using deterministic solver paths:

| Path | When used | Property |
|------|-----------|----------|
| Exhaustive search | Small search spaces | Exact within bounds; **falls back** when space is too large |
| CP-SAT | Larger/constrained spaces | Constraint-satisfying |
| Greedy | Fallback / large or hard spaces | Fast, bounded |

Relevant fixed defects that hardened this path:

- Exhaustive search previously **silently under-explored large spaces**; now it **falls back to
  CP-SAT/greedy**.
- `GlobalAllocator` previously **always chose the zero-cost baseline** (revenue never populated) —
  **fixed**.
- `GlobalAllocator` previously **over-provisioned** (rewarded replicas) — **fixed**.
- `OptimizationEngine` could **double-book GPUs** — **fixed** via fleet accounting.

The result is a **deterministic, auditable, safe** allocator that does not learn online.

---

## 4. RL_READINESS checks (required checklist)

All RL-readiness checks are marked **NOT APPLICABLE**, because RL is not implemented.

| # | RL readiness check | Status |
|---|--------------------|--------|
| 1 | RL environment (`Env`) implemented | **NOT APPLICABLE** |
| 2 | Observation space defined | **NOT APPLICABLE** |
| 3 | Action space defined | **NOT APPLICABLE** |
| 4 | Reward function defined | **NOT APPLICABLE** |
| 5 | Policy network / policy class present | **NOT APPLICABLE** |
| 6 | Training loop present | **NOT APPLICABLE** |
| 7 | Checkpointing / model persistence | **NOT APPLICABLE** |
| 8 | Offline dataset / replay buffer | **NOT APPLICABLE** |
| 9 | Evaluation harness for learned policies | **NOT APPLICABLE** |
| 10 | Safety validator / action masking | **NOT APPLICABLE** |
| 11 | Sim-to-real validation of a learned policy | **NOT APPLICABLE** |
| 12 | RL dependency stack (gym/torch/tf/SB3) present | **NOT APPLICABLE** |

---

## 5. Recommendation

### 5.1 Should contextual bandits / MPC / constrained RL be implemented?

| Approach | Worth implementing now? | Rationale |
|----------|------------------------|-----------|
| Contextual bandits | **Not yet** | Lower risk than full RL, but still needs a reward signal and real telemetry that Fusion does not yet have. Premature without calibrated data. |
| MPC (model-predictive control) | **Not yet** | Attractive for a control problem, but requires a trustworthy dynamics model (calibrated simulator + real telemetry) that does not exist today. |
| Constrained RL | **Not yet** | Highest potential but highest risk. Requires a safety validator, a reward definition, and calibrated telemetry. Without these, it is unsafe to deploy. |

**Bottom line:** None of these should be implemented **now**. The current deterministic allocator is
safe and auditable. Any RL-family method should be considered **only after** the prerequisites in
§5.2 are met, and should begin with the lowest-risk option (contextual bandits) behind a safety
validator.

### 5.2 Prerequisites before any RL-family work

1. **Real telemetry** — measured utilization/latency/throughput from real fleets (not synthetic).
2. **Calibrated simulator** — `FleetSimulator` currently does not model latency percentiles,
   KV-cache dynamics, cold-start timing, power, or NUMA; it must be calibrated before it can serve
   as a training/simulation substrate.
3. **Reward definition** — a well-specified, non-gameable objective (cost, SLO adherence, fairness)
   with explicit constraints.
4. **Safety validator** — an independent guard that rejects unsafe/overcommitting actions, plus
   action masking and fleet-accounting invariants.

Only when (1)–(4) exist should RL-family methods be prototyped, and only in advisory mode initially.

---

## 6. Caveats

- Fusion is advisory-only (`advisory_mode=True`, `execution_eligibility=False`); it does not execute
  or mutate cluster state. Any learned policy would inherit the same advisory boundary.
- All current performance/cost numbers are simulated; RL evaluation would inherit the simulator's
  calibration gaps.
- No GPU hardware validation has been performed.
