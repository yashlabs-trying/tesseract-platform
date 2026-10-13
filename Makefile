PY ?= python3
.PHONY: help install test test-unit test-contracts test-integration lint fmt contracts serve reconcile benchmark clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n", $$1, $$2}'

install: ## Install the package with test extras
	$(PY) -m pip install -e ".[test]"

test: ## Run the full test suite
	$(PY) -m pytest

test-unit: ## Run unit tests
	$(PY) -m pytest tests/unit

test-contracts: ## Run contract tests
	$(PY) -m pytest tests/contracts

test-integration: ## Run integration tests
	$(PY) -m pytest tests/integration

lint: ## Lint with ruff
	$(PY) -m ruff check .

fmt: ## Format with ruff
	$(PY) -m ruff format .

contracts: ## Validate vendored JSON schemas
	$(PY) -m tesseract_platform.cli validate-contracts

serve: ## Run the control-plane API
	$(PY) -m tesseract_platform.cli serve

reconcile: ## Run one reconciliation pass
	$(PY) -m tesseract_platform.cli reconcile

benchmark: ## Run the local benchmark harness
	$(PY) benchmarks/runner.py --scenario benchmarks/scenarios/local-lifecycle.json

clean: ## Remove caches and local state
	rm -rf var .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
