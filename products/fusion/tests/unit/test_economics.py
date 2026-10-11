from decimal import Decimal
import pytest
from fusion.economics.accounting import CostAccountingEngine, RevenueCalculator, MarginAnalyzer
from fusion.models.types import GPUInventory, ExpectedOutcome, BillingModel


class TestCostAccountingEngine:
    def test_gpu_cost(self):
        engine = CostAccountingEngine()
        gpu = GPUInventory(
            gpu_id="gpu-0", node_id="n0",
            gpu_model="H100", memory_capacity_mb=81920,
            resource_cost_per_hour=Decimal("4.50"),
        )
        cost = engine.compute_gpu_cost(gpu, hours=2.0)
        assert cost == Decimal("9.00")

    def test_zero_cost(self):
        engine = CostAccountingEngine()
        gpu = GPUInventory(
            gpu_id="gpu-0", node_id="n0",
            gpu_model="H100", memory_capacity_mb=81920,
            resource_cost_per_hour=Decimal("0"),
        )
        cost = engine.compute_gpu_cost(gpu)
        assert cost == Decimal("0")


class TestRevenueCalculator:
    def test_per_token(self):
        calc = RevenueCalculator()
        rev = calc.estimate_revenue(
            estimated_requests=1000,
            estimated_tokens=100000,
            billing=BillingModel.PER_TOKEN,
            token_rate=Decimal("0.0001"),
        )
        assert rev == Decimal("10.00")

    def test_hourly_rate(self):
        calc = RevenueCalculator()
        rev = calc.estimate_revenue(
            estimated_requests=0,
            estimated_tokens=0,
            billing=BillingModel.PER_HOUR_RESERVED,
            hourly_rate=Decimal("100"),
        )
        assert rev == Decimal("100")


class TestMarginAnalyzer:
    def test_candidate_better(self):
        analyzer = MarginAnalyzer()
        baseline = ExpectedOutcome(
            expected_revenue=Decimal("100"),
            expected_operating_cost=Decimal("80"),
            expected_contribution_margin=Decimal("20"),
        )
        candidate = ExpectedOutcome(
            expected_revenue=Decimal("100"),
            expected_operating_cost=Decimal("50"),
            expected_contribution_margin=Decimal("50"),
        )
        result = analyzer.compare_alternatives(baseline, candidate)
        assert result["verdict"] == "candidate_improves_margin"

    def test_baseline_superior(self):
        analyzer = MarginAnalyzer()
        baseline = ExpectedOutcome(
            expected_revenue=Decimal("100"),
            expected_operating_cost=Decimal("50"),
            expected_contribution_margin=Decimal("50"),
        )
        candidate = ExpectedOutcome(
            expected_revenue=Decimal("100"),
            expected_operating_cost=Decimal("80"),
            expected_contribution_margin=Decimal("20"),
        )
        result = analyzer.compare_alternatives(baseline, candidate)
        assert result["verdict"] == "baseline_superior"
