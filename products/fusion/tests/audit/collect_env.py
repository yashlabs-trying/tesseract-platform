"""Environment metadata collector for the Tesseract Fusion audit report.

Usage::

    python tests/audit/collect_env.py

Prints a JSON document with the Python runtime, platform details and the
versions of the packages Fusion depends on.  Importing this module has no side
effects; ``main()`` does the printing.
"""
from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import platform
import sys
from typing import Any

#: Packages reported in the audit environment summary.
TRACKED_PACKAGES = (
    "tesseract-fusion",
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic-settings",
    "sqlalchemy",
    "alembic",
    "typer",
    "httpx",
    "prometheus-client",
    "opentelemetry-api",
    "pyyaml",
    "rich",
    "pytest",
    "pytest-asyncio",
    "pytest-timeout",
    "ortools",
)


def package_versions() -> dict[str, str | None]:
    """Return installed versions for :data:`TRACKED_PACKAGES` (None if absent)."""
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect() -> dict[str, Any]:
    """Collect environment metadata as a JSON-serializable dictionary."""
    return {
        "python_version": sys.version,
        "python_version_info": list(sys.version_info[:3]),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_versions(),
    }


def main() -> None:
    print(json.dumps(collect(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
