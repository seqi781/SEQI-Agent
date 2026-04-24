#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.terminal_agent.trace_replay import append_fixture_to_module, load_trace, render_fixture_snippets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract trace replay fixture snippets from a LangSmith/Harbor trace JSON file."
    )
    parser.add_argument("trace_json", help="Path to a trace JSON file")
    parser.add_argument("--name", help="Optional explicit fixture name")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append the extracted fixture bundle directly to tests/trace_replay_fixtures.py",
    )
    args = parser.parse_args()

    trace = load_trace(args.trace_json)
    if args.append:
        fixture_path = ROOT / "tests" / "trace_replay_fixtures.py"
        name = append_fixture_to_module(trace, str(fixture_path), args.name)
        print(f"Appended fixture '{name}' to {fixture_path}")
        return
    print(render_fixture_snippets(trace, args.name))


if __name__ == "__main__":
    main()
