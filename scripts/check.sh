#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m py_compile $(find src tests -name '*.py' -print) main.py
uv run python -m unittest discover -s tests -v
