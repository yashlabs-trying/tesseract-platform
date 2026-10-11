from __future__ import annotations
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from fusion.models.types import (
    GPUInventory, ServiceEconomics, DeploymentAlternative,
    BillingModel, Currency, ExpectedOutcome,
)


class CostAccountingEngine:
    def compute_gpu_cost(self, gpu: GPUInventory, hours: float = 1.0) -> Decimal:
        return gpu.resource_cost_per_hour * Decimal(str(hours))

    def compute_allocated_cost(
        self,
        deployment_id: str,
        gpus: List[GPUInventory],
        hours: float = 1.0,
    ) -> ServiceEconomics:
        total_gpu_cost = sum((g.resource_cost_per_hour * Decimal(str(hours)) for g in gpus), Decimal("0"))
        return ServiceEconomics(
            deployment_id=deployment_id,
            tenant_id="unknown",
            billing_model=BillingModel.PER_HOUR_ON_DEMAND,
            effective_gpu_cost=total_gpu_cost,
            allocated_cost=total_gpu_cost,
            avoidable_cost=total_gpu_cost,
            cost_source="computed",
            time_window_start=datetime.utcnow(),
            time_window_end=datetime.utcnow() + timedelta(hours=hours),
            assumptions=[f"GPU cost for {len(gpus)} GPUs over {hours}h"],
        )

    def compute_alternative_cost(
        self,
        alternative: DeploymentAlternative,
        hours: float = 1.0,
    ) -> Decimal:
        return alternative.estimated_cost_per_hour * Decimal(str(hours))


class RevenueCalculator:
    def estimate_revenue(
        self,
        estimated_requests: int,
        estimated_tokens: int,
        billing: BillingModel,
        token_rate: Optional[Decimal] = None,
        request_rate: Optional[Decimal] = None,
        hourly_rate: Optional[Decimal] = None,
    ) -> Decimal:
        if billing == BillingModel.PER_TOKEN and token_rate:
            return Decimal(str(estimated_tokens)) * token_rate
        elif billing == BillingModel.PER_REQUEST and request_rate:
            return Decimal(str(estimated_requests)) * request_rate
        elif billing in (BillingModel.PER_HOUR_RESERVED, BillingModel.PER_HOUR_ON_DEMAND) and hourly_rate:
            return hourly_rate
        return Decimal("0")

    def contribution_margin(self, revenue: Decimal, cost: Decimal) -> Decimal:
        return revenue - cost


class MarginAnalyzer:
    def compare_alternatives(
        self,
        baseline: ExpectedOutcome,
        candidate: ExpectedOutcome,
    ) -> Dict[str, str]:
        diff = candidate.expected_contribution_margin - baseline.expected_contribution_margin if (
            baseline.expected_contribution_margin is not None and
            candidate.expected_contribution_margin is not None
        ) else None
        result = {
            "baseline_margin": str(baseline.expected_contribution_margin or "N/A"),
            "candidate_margin": str(candidate.expected_contribution_margin or "N/A"),
            "margin_difference": str(diff) if diff is not None else "N/A",
        }
        if diff is not None and diff > 0:
            result["verdict"] = "candidate_improves_margin"
        elif diff is not None and diff < 0:
            result["verdict"] = "baseline_superior"
        else:
            result["verdict"] = "no_difference"
        return result
