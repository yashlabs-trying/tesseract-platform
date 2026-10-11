from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import statistics
from fusion.models.types import WorkloadDemand, ForecastResult


class DemandForecaster:
    def forecast(
        self,
        demand_history: List[WorkloadDemand],
        horizon_hours: int = 24,
        points_per_hour: int = 12,
    ) -> Optional[ForecastResult]:
        if not demand_history:
            return None

        deployment_id = demand_history[0].deployment_id
        values = [d.requests_per_second for d in demand_history if d.requests_per_second > 0]
        if not values:
            return None

        n_points = horizon_hours * points_per_hour
        interval = timedelta(hours=1) / points_per_hour

        # Moving average
        window = min(len(values), 12)
        ma = statistics.mean(values[-window:]) if window > 0 else 0
        std = statistics.stdev(values[-window:]) if len(values) >= window > 1 else ma * 0.5

        # Simple peak-to-average
        peak = max(values)
        avg = statistics.mean(values)
        burst_prob = (peak - avg) / avg if avg > 0 else 0.0
        burst_prob = min(max(burst_prob, 0.0), 1.0)

        now = datetime.utcnow()
        time_points = [now + i * interval for i in range(n_points)]

        # Forecast with decreasing confidence further out
        forecast_vals = []
        uncertainty_vals = []
        for i in range(n_points):
            decay = 1.0 + (i / n_points) * 2.0
            forecast_vals.append(ma)
            uncertainty_vals.append(std * decay)

        return ForecastResult(
            deployment_id=deployment_id,
            forecast_rps=forecast_vals,
            forecast_uncertainty=uncertainty_vals,
            time_points=time_points,
            method="moving_average",
            burst_probability=burst_prob,
            evaluation_score=self._evaluate(forecast_vals, values),
        )

    def _evaluate(self, forecast: List[float], actual: List[float]) -> float:
        if not actual or not forecast:
            return 0.0
        mse = sum((f - a) ** 2 for f, a in zip(forecast[:len(actual)], actual)) / len(actual)
        variance = statistics.variance(actual) if len(actual) > 1 else 1.0
        return max(0.0, 1.0 - (mse / variance)) if variance > 0 else 0.0
