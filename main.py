import argparse

from src.langgraph_terminal_agent import LangGraphTerminalBenchAgent


def main():
    parser = argparse.ArgumentParser(description="LangGraph terminal-bench agent helper.")
    parser.add_argument(
        "--print-import-path",
        action="store_true",
        help="Print the Harbor import path for the main agent class.",
    )
    args = parser.parse_args()

    if args.print_import_path:
        print(LangGraphTerminalBenchAgent.import_path())
        return

    print("Harbor import path:")
    print(LangGraphTerminalBenchAgent.import_path())
    print()
    print("Example:")
    print('  import_path = "src.langgraph_terminal_agent:LangGraphTerminalBenchAgent"')


if __name__ == "__main__":
    main()
