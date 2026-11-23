"""M2: layer sensitivity analysis."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adaptive.errors import ConfigurationError
from adaptive.models.sensitivity import SensitivityMethod
from adaptive.sensitivity import DEFAULT_METHODS, analyze_sensitivity


def _analyze(arch, model, dataset, methods=DEFAULT_METHODS):
    return analyze_sensitivity(
        arch,
        model,
        dataset.sequences,
        methods=methods,
        bits=4,
        group_size=16,
        calibration_fingerprint=dataset.fingerprint(),
        max_batches=2,
    )


def _spearman(a: list[str], b: list[str]) -> float:
    rank_a = {name: i for i, name in enumerate(a)}
    rank_b = {name: i for i, name in enumerate(b)}
    names = list(rank_a)
    n = len(names)
    if n < 2:
        return 0.0
    mean_a = sum(rank_a[name] for name in names) / n
    mean_b = sum(rank_b[name] for name in names) / n
    cov = sum((rank_a[name] - mean_a) * (rank_b[name] - mean_b) for name in names)
    var_a = sum((rank_a[name] - mean_a) ** 2 for name in names)
    var_b = sum((rank_b[name] - mean_b) ** 2 for name in names)
    if var_a == 0 or var_b == 0:
        return 0.0
    return cov / (var_a**0.5 * var_b**0.5)


def test_report_shape(tiny_architecture, tiny_model, calibration_dataset) -> None:
    report = _analyze(tiny_architecture, tiny_model, calibration_dataset)
    assert report.methods == DEFAULT_METHODS
    assert report.bits == 4
    assert set(report.rankings) == {method.value for method in DEFAULT_METHODS}
    for method in DEFAULT_METHODS:
        ranking = report.ranking(method)
        assert ranking
        assert set(ranking) == set(report.scores)
    for by_method in report.scores.values():
        for score in by_method.values():
            assert 0.0 <= score <= 1.0


def test_rankings_are_deterministic(tiny_architecture, tiny_model, calibration_dataset) -> None:
    first = _analyze(tiny_architecture, tiny_model, calibration_dataset)
    second = _analyze(tiny_architecture, tiny_model, calibration_dataset)
    assert first.rankings == second.rankings


def test_rankings_consistent_across_methods(
    tiny_architecture, tiny_model, calibration_dataset
) -> None:
    report = _analyze(tiny_architecture, tiny_model, calibration_dataset)
    correlation = _spearman(
        report.ranking(SensitivityMethod.weight_error),
        report.ranking(SensitivityMethod.activation_weighted),
    )
    assert correlation > 0.0


def test_zeroed_layer_is_least_sensitive(
    tiny_architecture, tiny_model, calibration_dataset
) -> None:
    target = "model.layers.0.mlp.up_proj"
    module = dict(tiny_model.named_modules())[target]
    with torch.no_grad():
        module.weight.zero_()
    report = _analyze(tiny_architecture, tiny_model, calibration_dataset)
    raw = report.raw_scores
    for method in (SensitivityMethod.weight_error, SensitivityMethod.activation_weighted):
        values = {name: raw[name][method.value] for name in raw}
        assert values[target] == pytest.approx(0.0, abs=1e-12)
        assert values[target] == min(values.values())
        assert report.score(target, method) == pytest.approx(0.0, abs=1e-12)


def test_output_divergence_zeroed_layer_is_noop(
    tiny_architecture, tiny_model, calibration_dataset
) -> None:
    target = "model.layers.1.mlp.down_proj"
    module = dict(tiny_model.named_modules())[target]
    with torch.no_grad():
        module.weight.zero_()
    report = _analyze(
        tiny_architecture,
        tiny_model,
        calibration_dataset,
        methods=(SensitivityMethod.output_divergence,),
    )
    method = SensitivityMethod.output_divergence
    values = {
        name: report.raw_scores[name][method.value] for name in report.raw_scores
    }
    assert values[target] == pytest.approx(0.0, abs=1e-12)
    assert values[target] == min(values.values())
    assert max(values.values()) > 0.0


def test_family_attribution(tiny_architecture, tiny_model, calibration_dataset) -> None:
    report = _analyze(tiny_architecture, tiny_model, calibration_dataset)
    attribution = report.family_attribution["activation_weighted"]
    assert "attention" in attribution
    assert "mlp" in attribution
    for value in attribution.values():
        assert 0.0 <= value <= 1.0


def test_data_methods_require_calibration(tiny_architecture, tiny_model) -> None:
    with pytest.raises(ConfigurationError):
        analyze_sensitivity(
            tiny_architecture,
            tiny_model,
            [],
            methods=(SensitivityMethod.activation_weighted,),
        )


def test_fisher_method_runs(tiny_architecture, tiny_model, calibration_dataset) -> None:
    report = _analyze(
        tiny_architecture,
        tiny_model,
        calibration_dataset,
        methods=(SensitivityMethod.fisher,),
    )
    assert report.ranking(SensitivityMethod.fisher)
    for value in report.raw_scores[report.ranking(SensitivityMethod.fisher)[0]].values():
        assert value >= 0.0
