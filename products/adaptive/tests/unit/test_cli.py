"""M1: CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from adaptive.cli.main import app

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "models"
runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_analyze_command() -> None:
    result = runner.invoke(app, ["analyze", str(FIXTURES / "llama-3.2-1b.json")])
    assert result.exit_code == 0
    assert "parameters:" in result.stdout


def test_analyze_json_command() -> None:
    result = runner.invoke(
        app, ["analyze", str(FIXTURES / "qwen2.5-0.5b.json"), "--json"]
    )
    assert result.exit_code == 0
    assert '"family": "qwen2"' in result.stdout


def test_storage_command() -> None:
    result = runner.invoke(
        app, ["storage", str(FIXTURES / "llama-3.2-1b.json"), "--bits", "4"]
    )
    assert result.exit_code == 0
    assert "compression:" in result.stdout


def test_analyze_unknown_model_exits_nonzero(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"model_type": "mamba", "hidden_size": 8}')
    result = runner.invoke(app, ["analyze", str(config)])
    assert result.exit_code == 1


def test_sensitivity_command() -> None:
    pytest.importorskip("torch")
    result = runner.invoke(app, ["sensitivity", str(FIXTURES / "tiny-llama.json")])
    assert result.exit_code == 0
    assert "[weight_error]" in result.stdout
    assert "score=" in result.stdout


def test_optimize_command() -> None:
    pytest.importorskip("torch")
    result = runner.invoke(
        app, ["optimize", str(FIXTURES / "tiny-llama.json"), "--tolerance", "0.5"]
    )
    assert result.exit_code == 0
    assert "algorithm:" in result.stdout
    assert "selected:" in result.stdout
