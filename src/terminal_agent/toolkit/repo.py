from typing import Any

from harbor.environments.base import BaseEnvironment
from langchain.tools import tool

from src.terminal_agent.toolkit.schemas import GitDiffInput, GitStatusInput


def register_repo_tools(tools: list[Any], agent: Any, environment: BaseEnvironment) -> None:
    working_dir = agent.working_dir

    @tool(args_schema=GitDiffInput)
    async def git_diff(cwd: str | None = None) -> str:
        """Show the current git diff."""
        command = "if command -v git >/dev/null 2>&1; then git diff -- .; else printf 'git not available in the task environment\\n' >&2; exit 127; fi"
        return await agent._run_shell_tool(
            tool_name="git_diff",
            environment=environment,
            command=command,
            cwd=cwd or working_dir,
            timeout_sec=30,
        )

    @tool(args_schema=GitStatusInput)
    async def git_status(cwd: str | None = None) -> str:
        """Show current git status and diff summary."""
        command = "if command -v git >/dev/null 2>&1; then git status --short && printf '\n---\n' && git diff --stat; else printf 'git not available in the task environment\\n' >&2; exit 127; fi"
        return await agent._run_shell_tool(
            tool_name="git_status",
            environment=environment,
            command=command,
            cwd=cwd or working_dir,
            timeout_sec=30,
        )

    tools.extend([git_diff, git_status])
