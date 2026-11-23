"""Generate machine-readable benchmark data for the audit report."""
from __future__ import annotations
import json
import sys
from decimal import Decimal

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from tests.audit.test_baseline_competition import (
    BASELINES, _comparison_inputs, _fusion_recommendation, _fusion_cost,
    _improvement_pct,
)
from fusion.optimizer.global_allocator import GlobalAllocator


def main():
    deployments, gpus = _comparison_inputs()
    rec = _fusion_recommendation(deployments, gpus)
    fusion_cost = _fusion_cost(rec)

    results = {}
    for name, fn in BASELINES:
        baseline = fn(deployments, gpus)
        base_cost = baseline["total_cost"]
        improvement = _improvement_pct(base_cost, fusion_cost)
        results[name] = {
            "baseline_cost": str(base_cost),
            "fusion_cost": str(fusion_cost),
            "improvement_pct": str(improvement),
        }

    served = sorted(a.deployment_id for a in rec.recommended_plan)
    out = {
        "fusion_cost": str(fusion_cost),
        "workloads_served": len(served),
        "total_workloads": len(deployments),
        "engine": rec.provenance.get("engine"),
        "baselines": results,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
