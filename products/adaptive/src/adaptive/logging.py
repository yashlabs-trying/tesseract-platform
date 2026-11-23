"""Structured logging configuration.

Emits JSON log records compatible with the wider Tesseract platform. Falls
back to a human-readable console formatter when ``json_logs`` is false.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

try:  # pragma: no cover - import guard for minimal installs
    from pythonjsonlogger.json import JsonFormatter

    _HAS_JSON_LOGGER = True
except Exception:  # pragma: no cover
    try:
        from pythonjsonlogger.jsonlogger import JsonFormatter  # type: ignore[attr-defined]

        _HAS_JSON_LOGGER = True
    except Exception:
        JsonFormatter = None  # type: ignore[assignment,misc]
        _HAS_JSON_LOGGER = False


_CONFIGURED = False


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure the root ``adaptive`` logger exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger = logging.getLogger("adaptive")
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stderr)

    if json_logs and _HAS_JSON_LOGGER and JsonFormatter is not None:
        fmt = JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
        )
        handler.setFormatter(fmt)
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )

    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``adaptive`` namespace."""
    return logging.getLogger(f"adaptive.{name}")


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured event record with a stable ``event`` field."""
    logger.info(event, extra={"event": event, **fields})
