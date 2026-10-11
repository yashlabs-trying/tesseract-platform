from datetime import datetime, timedelta
import pytest
from fusion.demand.forecaster import DemandForecaster
from fusion.demand.profile import WorkloadProfileBuilder
from fusion.models.types import WorkloadDemand


class TestDemandForecaster:
    def test_empty_history(self):
        forecaster = DemandForecaster()
        result = forecaster.forecast([])
        assert result is None

    def test_single_point(self):
        forecaster = DemandForecaster()
        demand = WorkloadDemand(deployment_id="d1", requests_per_second=10.0)
        result = forecaster.forecast([demand], horizon_hours=1)
        assert result is not None
        assert len(result.forecast_rps) > 0
