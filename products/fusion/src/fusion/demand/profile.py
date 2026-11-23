from __future__ import annotations
from typing import List, Optional, Tuple
import statistics
from datetime import datetime
from fusion.models.types import WorkloadDemand


class WorkloadProfileBuilder:
    def build_from_series(self, deployment_id: str, rps_values: List[float], timestamps: List[datetime]) -> WorkloadDemand:
        if not rps_values:
            return WorkloadDemand(deployment_id=deployment_id)
        return WorkloadDemand(
            deployment_id=deployment_id,
            requests_per_second=statistics.mean(rps_values),
            current_concurrency=0,
            queue_depth=0,
            observed_throughput_rps=max(rps_values),
            predicted_demand_rps=statistics.mean(rps_values[-10:]) if len(rps_values) >= 10 else statistics.mean(rps_values),
            time_window_seconds=len(rps_values) * 60,
            source="profile_builder",
            measured_at=datetime.utcnow(),
        )

    def compute_peak_to_average(self, rps_values: List[float]) -> float:
        if not rps_values:
            return 1.0
        avg = statistics.mean(rps_values)
        if avg == 0:
            return 1.0
        return max(rps_values) / avg
