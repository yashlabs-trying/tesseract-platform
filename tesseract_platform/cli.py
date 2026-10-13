"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from tesseract_platform import __version__
from tesseract_platform.config import get_settings


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    uvicorn.run(
        "tesseract_platform.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=settings.log_level.lower(),
    )
    return 0


def _reconcile(_: argparse.Namespace) -> int:
    from tesseract_platform.runtime import build_platform

    platform = build_platform()
    actions = platform.reconciler.reconcile()
    platform.close()
    print(json.dumps({"actions": actions}, indent=2))
    return 0


def _validate_contracts(_: argparse.Namespace) -> int:
    from tesseract_platform.contracts import load_schema

    names = [
        "workload",
        "telemetry",
        "reservation",
        "deployment",
        "lifecycle",
        "precision-profile",
        "rollback",
        "foundry-performance-profile",
    ]
    ok = True
    for name in names:
        try:
            schema = load_schema(name)
            print(f"ok   {name:32s} $id={schema.get('$id')}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"FAIL {name:32s} {exc}")
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tesseract-platform")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the control-plane API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=_serve)

    sub.add_parser("reconcile", help="run one reconciliation pass").set_defaults(
        func=_reconcile
    )
    sub.add_parser("validate-contracts", help="validate vendored JSON schemas").set_defaults(
        func=_validate_contracts
    )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
