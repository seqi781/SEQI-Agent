import shlex
import textwrap
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from src.terminal_agent.formatting import make_write_command
from src.terminal_agent.toolkit.schemas import (
    CreateCommandShimInput,
    CreateHelperToolInput,
    CreatePythonToolInput,
    CreateShellToolInput,
    InstallHelperToolInput,
)
from src.terminal_agent.toolkit.web import download_via_host_fallback


def register_extension_tools(tools: list[Any], agent: Any, environment: BaseEnvironment) -> None:
    working_dir = agent.working_dir

    def helper_target(path: str) -> str:
        candidate = path.strip()
        if not candidate:
            return agent.helper_dir
        if candidate.startswith("/"):
            return candidate
        return f"{agent.helper_dir.rstrip('/')}/{candidate}"

    def helper_command_update(result: str, target_path: str, runtime: ToolRuntime) -> Command:
        helper_paths = list(runtime.state.get("helper_paths", []))
        if target_path not in helper_paths:
            helper_paths.append(target_path)
        return Command(
            update={
                "helper_paths": helper_paths,
                "messages": [
                    ToolMessage(
                        content=result,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    async def create_helper_impl(
        *,
        target_path: str,
        content: str,
        executable: bool,
        usage_note: str,
        tool_name: str,
        runtime: ToolRuntime | None,
    ) -> str | Command:
        quoted_path = shlex.quote(target_path)
        write_command = make_write_command(path=target_path, content=content, append=False)
        usage_comment = ""
        if usage_note.strip():
            usage_comment = f"printf 'usage: %s\\n' {shlex.quote(usage_note.strip())}\n"
        command = textwrap.dedent(
            f"""
            {write_command}
            {'chmod +x ' + quoted_path if executable else 'true'}
            {usage_comment}printf 'created %s\\n' {quoted_path}
            """
        ).strip()
        result = await agent._run_shell_tool(
            tool_name=tool_name,
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=60,
        )
        if runtime is None:
            return result
        return helper_command_update(result, target_path, runtime)

    async def create_python_impl(
        *,
        path: str,
        content: str,
        executable: bool,
        usage_note: str,
        runtime: ToolRuntime | None,
    ) -> str | Command:
        if not (agent._capabilities.get("python3") or agent._capabilities.get("python")):
            return await agent._run_shell_tool(
                tool_name="create_python_tool",
                environment=environment,
                command="printf 'python3/python is not available in the task environment; use create_shell_tool or create_command_shim instead\\n' >&2; exit 127",
                cwd=working_dir,
                timeout_sec=10,
            )
        normalized_content = content
        if not normalized_content.lstrip().startswith("#!"):
            normalized_content = "#!/usr/bin/env python3\n" + normalized_content.lstrip("\n")
        return await create_helper_impl(
            target_path=helper_target(path),
            content=normalized_content,
            executable=executable,
            usage_note=usage_note or path,
            tool_name="create_python_tool",
            runtime=runtime,
        )

    async def create_shell_impl(
        *,
        path: str,
        content: str,
        executable: bool,
        usage_note: str,
        runtime: ToolRuntime | None,
    ) -> str | Command:
        normalized_content = content
        if not normalized_content.lstrip().startswith("#!"):
            normalized_content = "#!/bin/sh\nset -eu\n" + normalized_content.lstrip("\n")
        return await create_helper_impl(
            target_path=helper_target(path),
            content=normalized_content,
            executable=executable,
            usage_note=usage_note or path,
            tool_name="create_shell_tool",
            runtime=runtime,
        )

    @tool(args_schema=CreateHelperToolInput)
    async def create_helper_tool(
        path: str,
        content: str,
        executable: bool = True,
        usage_note: str = "",
        runtime: ToolRuntime | None = None,
    ) -> str | Command:
        """Create a small helper script or local CLI in the task environment. Verifier helpers should print VERIFICATION_RESULT=PASS|FAIL|BLOCKED when possible."""
        target_path = helper_target(path)
        return await create_helper_impl(
            target_path=target_path,
            content=content,
            executable=executable,
            usage_note=usage_note,
            tool_name="create_helper_tool",
            runtime=runtime,
        )

    @tool(args_schema=CreatePythonToolInput)
    async def create_python_tool(
        path: str,
        content: str,
        executable: bool = True,
        usage_note: str = "",
        runtime: ToolRuntime | None = None,
    ) -> str | Command:
        """Create a small Python helper script. Verifier helpers should print VERIFICATION_RESULT=PASS|FAIL|BLOCKED and task-specific facts."""
        return await create_python_impl(
            path=path,
            content=content,
            executable=executable,
            usage_note=usage_note,
            runtime=runtime,
        )

    @tool(args_schema=CreateShellToolInput)
    async def create_shell_tool(
        path: str,
        content: str,
        executable: bool = True,
        usage_note: str = "",
        runtime: ToolRuntime | None = None,
    ) -> str | Command:
        """Create a small shell helper script. Verifier helpers should print VERIFICATION_RESULT=PASS|FAIL|BLOCKED and task-specific facts."""
        return await create_shell_impl(
            path=path,
            content=content,
            executable=executable,
            usage_note=usage_note,
            runtime=runtime,
        )

    @tool(args_schema=InstallHelperToolInput)
    async def install_helper_tool(
        source_url: str,
        destination: str,
        executable: bool = True,
        runtime: ToolRuntime | None = None,
    ) -> str | Command:
        """Install a helper script from a public URL into the task environment."""
        target_path = helper_target(destination)
        quoted_url = shlex.quote(source_url)
        quoted_destination = shlex.quote(target_path)
        command = textwrap.dedent(
            f"""
            mkdir -p {shlex.quote(str(Path(target_path).parent))}
            if command -v curl >/dev/null 2>&1; then
              curl -fsSL {quoted_url} -o {quoted_destination}
            elif command -v wget >/dev/null 2>&1; then
              wget -qO {quoted_destination} {quoted_url}
            else
              printf 'curl or wget is required in the task environment for install_helper_tool\n' >&2
              exit 127
            fi
            {'chmod +x ' + quoted_destination if executable else 'true'}
            printf 'installed %s\n' {quoted_destination}
            """
        ).strip()
        if agent._capabilities.get("curl") or agent._capabilities.get("wget"):
            result = await agent._run_shell_tool(
                tool_name="install_helper_tool",
                environment=environment,
                command=command,
                cwd=working_dir,
                timeout_sec=max(agent.tool_timeout_sec, 60),
            )
        else:
            result = await download_via_host_fallback(
                agent=agent,
                environment=environment,
                working_dir=working_dir,
                tool_name="install_helper_tool",
                url=source_url,
                destination=target_path,
                executable=executable,
            )
        if runtime is None:
            return result
        return helper_command_update(result, target_path, runtime)

    @tool(args_schema=CreateCommandShimInput)
    async def create_command_shim(
        command_name: str,
        content: str,
        kind: str = "shell",
        usage_note: str = "",
        runtime: ToolRuntime | None = None,
    ) -> str | Command:
        """Create an executable shim named like a missing command inside the helper directory so later shell commands can resolve it via PATH."""
        safe_name = Path(command_name).name.strip()
        target_name = safe_name or "helper-shim"
        target_path = f"{agent.helper_dir.rstrip('/')}/{target_name}"
        normalized_kind = kind.strip().lower()
        lowered_content = content.lower()
        blocked_markers = [
            "not available",
            "not implemented",
            "placeholder",
            "stub",
            "cannot do",
            "can't do",
        ]
        if any(marker in lowered_content for marker in blocked_markers):
            return await agent._run_shell_tool(
                tool_name="create_command_shim",
                environment=environment,
                command="printf 'create_command_shim rejected placeholder or non-functional shim content\\n' >&2; exit 1",
                cwd=working_dir,
                timeout_sec=10,
            )
        if normalized_kind == "python":
            return await create_python_impl(
                path=target_path,
                content=content,
                executable=True,
                usage_note=usage_note or target_name,
                runtime=runtime,
            )
        return await create_shell_impl(
            path=target_path,
            content=content,
            executable=True,
            usage_note=usage_note or target_name,
            runtime=runtime,
        )

    tools.extend(
        [
            create_helper_tool,
            create_python_tool,
            create_shell_tool,
            create_command_shim,
            install_helper_tool,
        ]
    )
