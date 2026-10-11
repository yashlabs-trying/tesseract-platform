from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
import urllib.request
from urllib.parse import quote


class PrometheusMetricsAdapter:
    def __init__(self, base_url: str = "http://localhost:9090", timeout: int = 30):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def query(self, query: str) -> List[Dict[str, Any]]:
        url = f"{self._base_url}/api/v1/query?query={quote(query)}"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
            if data.get("status") != "success":
                raise RuntimeError(f"Prometheus query failed: {data.get('error', 'unknown')}")
            return data["data"]["result"]
        except Exception as e:
            raise RuntimeError(f"Prometheus query error: {e}")

    def query_range(self, query: str, start: datetime, end: datetime, step: str = "60s") -> List[Dict[str, Any]]:
        params = (
            f"query={quote(query)}"
            f"&start={start.timestamp()}"
            f"&end={end.timestamp()}"
            f"&step={step}"
        )
        url = f"{self._base_url}/api/v1/query_range?{params}"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
            if data.get("status") != "success":
                raise RuntimeError(f"Prometheus range query failed: {data.get('error', 'unknown')}")
            return data["data"]["result"]
        except Exception as e:
            raise RuntimeError(f"Prometheus range query error: {e}")

    def get_gpu_utilization(self) -> List[Dict[str, Any]]:
        return self.query('DCGM_FI_DEV_GPU_UTIL')

    def get_gpu_memory_utilization(self) -> List[Dict[str, Any]]:
        return self.query('DCGM_FI_DEV_MEM_COPY_UTIL')

    def get_request_rates(self) -> List[Dict[str, Any]]:
        return self.query('rate(vllm:requests_total[5m])')

    def get_inflight_requests(self) -> List[Dict[str, Any]]:
        return self.query('vllm:requests_running')
