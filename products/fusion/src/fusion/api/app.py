from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fusion.config import Settings, get_settings
from fusion.database.session import init_db, get_db
from fusion.inventory import FixtureInventoryAdapter, InventoryService
from fusion.models.types import (
    GPUInventory, ModelDeployment, TenantPolicy, PerformanceProfile,
    WorkloadDemand, OptimizationScenario, FusionRecommendation,
    SimulationResult, SimulationState, ForecastResult,
)
from fusion.optimizer.engine import OptimizationEngine
from fusion.simulator.engine import FleetSimulator
from fusion.simulator.scenario import ScenarioLoader, ScenarioDefinition
from fusion.demand.forecaster import DemandForecaster
from fusion.demand.profile import WorkloadProfileBuilder
import uuid
from fusion._time import utcnow_naive


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="Tesseract Fusion API",
        version="0.1.0",
        description="GPU Fleet Economics and Optimization Platform",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    inventory_service = InventoryService(FixtureInventoryAdapter())
    optimizer = OptimizationEngine(config={
        "max_candidates_per_workload": settings.max_candidates_per_workload,
    })
    simulator = FleetSimulator(seed=settings.optimizer_seed)
    forecaster = DemandForecaster()
    scenario_loader = ScenarioLoader()

    # In-memory stores for demo
    _deployments: Dict[str, ModelDeployment] = {}
    _policies: Dict[str, TenantPolicy] = {}
    _profiles: Dict[str, PerformanceProfile] = {}
    _demands: Dict[str, WorkloadDemand] = {}
    _recommendations: Dict[str, FusionRecommendation] = {}
    _simulations: Dict[str, SimulationResult] = {}
    _scenarios: Dict[str, OptimizationScenario] = {}
    _forecasts: Dict[str, ForecastResult] = {}
    _demand_history: Dict[str, List[WorkloadDemand]] = {}

    @app.on_event("startup")
    async def startup():
        init_db()

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "timestamp": utcnow_naive().isoformat()}

    @app.get("/readyz")
    async def readyz():
        return {"status": "ready"}

    @app.post("/v1/inventory/import")
    async def import_inventory(data: List[dict]):
        adapter = FixtureInventoryAdapter()
        inventory = adapter.load_from_dict(data)
        for gpu in inventory:
            inventory_service._cache.append(gpu)
        return {"imported": len(inventory)}

    @app.get("/v1/inventory")
    async def get_inventory(force_refresh: bool = False):
        return inventory_service.get_inventory(force_refresh)

    @app.get("/v1/inventory/capacity")
    async def get_capacity():
        return inventory_service.get_total_capacity_by_model()

    @app.post("/v1/workloads/import")
    async def import_workloads(data: List[dict]):
        from pydantic import ValidationError
        count = 0
        for entry in data:
            try:
                dep = ModelDeployment(**entry)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors())
            _deployments[dep.deployment_id] = dep
            count += 1
        return {"imported": count}

    @app.get("/v1/workloads")
    async def get_workloads():
        return list(_deployments.values())

    @app.post("/v1/performance-profiles/import")
    async def import_profile(data: dict):
        profile = PerformanceProfile(**data)
        _profiles[profile.profile_id] = profile
        _profiles[profile.candidate_id] = profile
        return {"imported": profile.profile_id}

    @app.get("/v1/performance-profiles/{profile_id}")
    async def get_profile(profile_id: str):
        profile = _profiles.get(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile

    @app.post("/v1/economics/evaluate")
    async def evaluate_economics(data: dict):
        from fusion.economics.accounting import CostAccountingEngine
        engine = CostAccountingEngine()
        result = {}
        if "gpu_cost" in data:
            gpu_data = data.get("gpu", {})
            result["computed_cost"] = str(engine.compute_gpu_cost(
                GPUInventory(**{k: v for k, v in gpu_data.items() if k in GPUInventory.model_fields}),
                data.get("hours", 1),
            ))
        return result

    @app.get("/v1/economics/reports/{report_id}")
    async def get_economics_report(report_id: str):
        return {"report_id": report_id, "status": "simulated"}

    @app.post("/v1/optimization/evaluate")
    async def evaluate_optimization(data: dict):
        scenario = OptimizationScenario(**data.get("scenario", {}))
        _scenarios[scenario.scenario_id] = scenario

        recommendation = optimizer.optimize(
            deployments=list(_deployments.values()),
            gpus=inventory_service.get_inventory(),
            tenant_policies=_policies,
            profiles=_profiles,
            demands=_demands,
            scenario=scenario,
        )
        _recommendations[recommendation.recommendation_id] = recommendation
        return recommendation

    @app.get("/v1/optimization/recommendations/{rec_id}")
    async def get_recommendation(rec_id: str):
        rec = _recommendations.get(rec_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        return rec

    @app.post("/v1/simulations")
    async def create_simulation(data: dict, background_tasks: BackgroundTasks):
        scenario = ScenarioDefinition(**data.get("scenario", {}))
        sim_result = simulator.run(scenario)
        _simulations[sim_result.simulation_id] = sim_result
        return sim_result

    @app.get("/v1/simulations/{sim_id}")
    async def get_simulation(sim_id: str):
        sim = _simulations.get(sim_id)
        if not sim:
            raise HTTPException(status_code=404, detail="Simulation not found")
        return sim

    @app.post("/v1/forecast/evaluate")
    async def evaluate_forecast(data: dict):
        demand_history = [WorkloadDemand(**d) for d in data.get("demand_history", [])]
        forecast = forecaster.forecast(
            demand_history,
            horizon_hours=data.get("horizon_hours", 24),
        )
        if forecast:
            _forecasts[forecast.deployment_id] = forecast
        return forecast or {"error": "insufficient_data"}

    return app
