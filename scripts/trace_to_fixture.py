#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.terminal_agent.trace_replay import load_trace, render_fixture_snippets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract trace replay fixture snippets from a LangSmith/Harbor trace JSON file."
    )
    parser.add_argument("trace_json", help="Path to a trace JSON file")
    parser.add_argument("--name", help="Optional explicit fixture name")
    args = parser.parse_args()

    trace = load_trace(args.trace_json)
    print(render_fixture_snippets(trace, args.name))


if __name__ == "__main__":
    main()
