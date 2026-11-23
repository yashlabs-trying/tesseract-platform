"""Constrained precision optimization (Milestone 2).

Greedy layer upgrades and a beam-search alternative minimize estimated serving
cost subject to quality tolerance, latency, memory, throughput and runtime
compatibility constraints. Uniform 4-bit and 8-bit baselines are always
included for comparison and a Pareto frontier is reported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from adaptive.errors import ConfigurationError
from adaptive.models.architecture import ModelArchitecture
from adaptive.models.enums import (
    EvidenceType,
    Granularity,
    QuantMethod,
    QuantScheme,
    RuntimeBackend,
)
from adaptive.models.optimization import (
    CandidateEvaluation,
    OptimizationConstraints,
    OptimizationResult,
)
from adaptive.models.quantization import PrecisionProfile
from adaptive.models.sensitivity import SensitivityMethod, SensitivityReport
from adaptive.optimizer.cost import AnalyticCostModel
from adaptive.optimizer.pareto import pareto_frontier
from adaptive.optimizer.quality_estimator import SensitivityQualityEstimator, bit_penalty
from adaptive.quantize.profiles import (
    capability,
    mixed_profile,
    uniform_profile,
    validate_profile,
)
from adaptive.quantize.storage import compute_storage

QualityEstimator = SensitivityQualityEstimator


def _quantizable_layer_names(architecture: ModelArchitecture) -> list[str]:
    return [
        spec.name
        for spec in architecture.layers
        if spec.quantizable and len(spec.shape) == 2
    ]


def _allowed_bits(
    constraints: OptimizationConstraints, runtime: RuntimeBackend, method: QuantMethod
) -> tuple[int, ...]:
    cap = capability(runtime)
    allowed = [
        bits
        for bits in constraints.allowed_bits
        if cap.supports_bits(bits, is_float=bits >= 16) and (bits >= 16 or method in cap.methods)
    ]
    if not allowed:
        raise ConfigurationError(
            f"runtime {runtime.value} supports none of bits {constraints.allowed_bits}"
        )
    return tuple(sorted(allowed))


def _score(sensitivity: SensitivityReport, layer: str, method: SensitivityMethod) -> float:
    by_method = sensitivity.scores.get(layer, {})
    if method.value in by_method:
        return by_method[method.value]
    if by_method:
        return max(by_method.values())
    return 1.0


class _Optimizer:
    def __init__(
        self,
        architecture: ModelArchitecture,
        sensitivity: SensitivityReport,
        constraints: OptimizationConstraints,
        *,
        baseline_metric_value: float,
        cost_model: AnalyticCostModel | None,
        quality_estimator: QualityEstimator | None,
        group_size: int | None,
    ) -> None:
        self.architecture = architecture
        self.sensitivity = sensitivity
        self.constraints = constraints
        self.layers = _quantizable_layer_names(architecture)
        self.allowed = _allowed_bits(constraints, constraints.runtime, constraints.method)
        self.cost_model = cost_model or AnalyticCostModel()
        self.group_size = group_size
        self.method = constraints.method
        self.quality = quality_estimator or SensitivityQualityEstimator(
            sensitivity,
            baseline_value=baseline_metric_value,
            metric=constraints.quality_metric,
        )
        self.evaluated: dict[str, CandidateEvaluation] = {}

    # -- profile construction --------------------------------------------
    def _mixed(self, bits_map: Mapping[str, int], default_bits: int, name: str) -> PrecisionProfile:
        return mixed_profile(
            self.architecture,
            bits_map,
            default_bits=default_bits,
            runtime=self.constraints.runtime,
            method=self.method,
            scheme=QuantScheme.symmetric,
            granularity=(Granularity.per_group if self.group_size else Granularity.per_channel),
            group_size=self.group_size,
            name=name,
        )

    def _uniform(self, bits: int) -> PrecisionProfile:
        return uniform_profile(
            self.architecture,
            bits,
            runtime=self.constraints.runtime,
            method=self.method,
            granularity=(Granularity.per_group if self.group_size else Granularity.per_channel),
            group_size=self.group_size,
        )

    # -- evaluation -------------------------------------------------------
    def evaluate(self, profile: PrecisionProfile) -> CandidateEvaluation:
        if profile.profile_id in self.evaluated:
            return self.evaluated[profile.profile_id]
        storage = compute_storage(self.architecture, profile)
        cost = self.cost_model.estimate(self.architecture, profile)
        degradation = self.quality.degradation(profile)
        predicted = self.quality.predict(profile)

        violations: list[str] = list(validate_profile(self.architecture, profile))
        if degradation > self.constraints.quality_tolerance:
            violations.append(
                f"quality degradation {degradation:.4f} > {self.constraints.quality_tolerance:.4f}"
            )
        if self.constraints.memory_budget_gb is not None:
            budget_bytes = self.constraints.memory_budget_gb * 1e9
            if storage.total_bytes > budget_bytes:
                violations.append(
                    f"memory {storage.total_bytes / 1e9:.3f} GB > "
                    f"{self.constraints.memory_budget_gb} GB"
                )
        if (
            self.constraints.p95_latency_ms is not None
            and cost.estimated_latency_ms > self.constraints.p95_latency_ms
        ):
            violations.append(
                f"latency {cost.estimated_latency_ms:.2f} ms > {self.constraints.p95_latency_ms} ms"
            )
        if (
            self.constraints.min_throughput_tps
            and cost.estimated_throughput_tps < self.constraints.min_throughput_tps
        ):
            violations.append(
                f"throughput {cost.estimated_throughput_tps:.2f} < "
                f"{self.constraints.min_throughput_tps} tps"
            )

        evaluation = CandidateEvaluation(
            profile=profile,
            storage=storage,
            cost=cost,
            predicted_quality=predicted,
            predicted_degradation=degradation,
            feasible=not violations,
            violations=tuple(violations),
        )
        self.evaluated[profile.profile_id] = evaluation
        return evaluation

    # -- algorithms -------------------------------------------------------
    def baselines(self) -> list[CandidateEvaluation]:
        candidates: list[CandidateEvaluation] = []
        for bits in (4, 8, 16):
            if bits in self.allowed:
                candidates.append(self.evaluate(self._uniform(bits)))
        return candidates

    def greedy(self) -> CandidateEvaluation:
        start = (
            self.constraints.start_bits
            if self.constraints.start_bits in self.allowed
            else self.allowed[0]
        )
        current = {name: start for name in self.layers}
        best = self.evaluate(self._mixed(current, start, "greedy"))
        if best.feasible:
            return best

        levels = list(self.allowed)
        for _ in range(len(self.layers) * (len(levels) - 1) + 1):
            candidate_name, candidate_next = self._best_upgrade(current, levels)
            if candidate_name is None:
                break
            current[candidate_name] = candidate_next
            evaluation = self.evaluate(self._mixed(current, start, "greedy"))
            best = evaluation
            if evaluation.feasible:
                break
        return best

    def _best_upgrade(
        self, current: Mapping[str, int], levels: Sequence[int]
    ) -> tuple[str | None, int]:
        best_layer: str | None = None
        best_next = 0
        best_gain = -1.0
        for name in self.layers:
            bits = current[name]
            next_bits = next((b for b in levels if b > bits), None)
            if next_bits is None:
                continue
            score = _score(self.sensitivity, name, SensitivityMethod.activation_weighted)
            gain = score * (bit_penalty(bits) - bit_penalty(next_bits))
            if gain > best_gain:
                best_gain = gain
                best_layer = name
                best_next = next_bits
        return best_layer, best_next

    def beam(self, width: int | None = None) -> CandidateEvaluation:
        beam_width = width or self.constraints.max_beam_width
        start = (
            self.constraints.start_bits
            if self.constraints.start_bits in self.allowed
            else self.allowed[0]
        )
        levels = list(self.allowed)
        beam: list[dict[str, int]] = [{name: start for name in self.layers}]
        seen: set[tuple[tuple[str, int], ...]] = {tuple(sorted(beam[0].items()))}
        best: CandidateEvaluation | None = None

        for _ in range(len(self.layers) * (len(levels) - 1) + 1):
            expansions: list[dict[str, int]] = []
            for state in beam:
                for name in self.layers:
                    next_bits = next((b for b in levels if b > state[name]), None)
                    if next_bits is None:
                        continue
                    candidate = dict(state)
                    candidate[name] = next_bits
                    key = tuple(sorted(candidate.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                    expansions.append(candidate)
            if not expansions:
                break
            scored: list[tuple[float, float, dict[str, int]]] = []
            for state in expansions:
                evaluation = self.evaluate(self._mixed(state, start, "beam"))
                violation_penalty = 0.0 if evaluation.feasible else 1.0
                scored.append((violation_penalty, evaluation.cost.lower_cost, state))
                if evaluation.feasible and (
                    best is None or evaluation.cost.lower_cost < best.cost.lower_cost
                ):
                    best = evaluation
            scored.sort(key=lambda item: (item[0], item[1]))
            beam = [state for _, _, state in scored[:beam_width]]

        if best is None:
            best = self.greedy()
        return best


def optimize(
    architecture: ModelArchitecture,
    sensitivity: SensitivityReport,
    constraints: OptimizationConstraints | None = None,
    *,
    baseline_metric_value: float = 1.0,
    cost_model: AnalyticCostModel | None = None,
    quality_estimator: QualityEstimator | None = None,
    algorithm: str = "greedy",
    group_size: int | None = 128,
) -> OptimizationResult:
    """Generate, evaluate and select mixed-precision profiles."""
    constraints = constraints or OptimizationConstraints()
    optimizer = _Optimizer(
        architecture,
        sensitivity,
        constraints,
        baseline_metric_value=baseline_metric_value,
        cost_model=cost_model,
        quality_estimator=quality_estimator,
        group_size=group_size,
    )
    baselines = optimizer.baselines()
    if algorithm == "beam":
        optimizer.beam()
    elif algorithm == "greedy":
        optimizer.greedy()
    else:
        raise ConfigurationError(f"unknown optimization algorithm: {algorithm}")

    candidates = list(optimizer.evaluated.values())
    points = [
        (candidate.profile_id, candidate.cost.lower_cost, candidate.predicted_degradation)
        for candidate in candidates
    ]
    frontier = pareto_frontier(points)

    feasible = [candidate for candidate in candidates if candidate.feasible]
    notes: list[str] = []
    selected_id: str | None
    if feasible:
        selected = min(feasible, key=lambda candidate: candidate.cost.lower_cost)
        selected_id = selected.profile_id
    else:
        baseline = next(
            (
                candidate
                for candidate in baselines
                if candidate.profile.name.startswith("uniform-16")
            ),
            None,
        )
        selected_id = baseline.profile_id if baseline is not None else None
        notes.append("no feasible profile found; recommending baseline precision")

    uniform_4 = next(
        (candidate for candidate in candidates if candidate.profile.name.startswith("uniform-4")),
        None,
    )
    uniform_8 = next(
        (candidate for candidate in candidates if candidate.profile.name.startswith("uniform-8")),
        None,
    )

    return OptimizationResult(
        model_id=architecture.model_id,
        structural_fingerprint=architecture.structural_fingerprint(),
        constraints=constraints,
        candidates=tuple(candidates),
        pareto_frontier=tuple(frontier),
        selected_profile_id=selected_id,
        uniform_4_profile_id=uniform_4.profile_id if uniform_4 else None,
        uniform_8_profile_id=uniform_8.profile_id if uniform_8 else None,
        algorithm=algorithm,
        evidence=EvidenceType.estimated,
        notes=tuple(notes),
    )
