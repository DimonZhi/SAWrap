#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


DEFAULT_CHECKS = {
    "/": "SAWrap",
    "/overview": 'data-panel="eda"',
    "/leaderboard": "leaderboard",
    "/link": None,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_app():
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.environ.setdefault("SAWRAP_SKIP_MISSING_RECALC", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

    from UI.app import app  # noqa: PLC0415

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test key SAWrap web pages through FastAPI TestClient.",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        default=None,
        help="Endpoint to check. Can be passed multiple times. Defaults to key pages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoints = args.endpoint or list(DEFAULT_CHECKS)

    from fastapi.testclient import TestClient

    client = TestClient(_load_app())
    failures: list[str] = []

    for endpoint in endpoints:
        response = client.get(endpoint)
        expected_text = DEFAULT_CHECKS.get(endpoint)

        if response.status_code != 200:
            failures.append(f"{endpoint}: expected 200, got {response.status_code}")
            continue

        if expected_text and expected_text not in response.text:
            failures.append(f"{endpoint}: missing marker {expected_text!r}")
            continue

        print(f"[ok] GET {endpoint} -> 200")

    if failures:
        print("\nSmoke test failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
