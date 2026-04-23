import difflib
import json
import shlex
import textwrap
from typing import Any

from harbor.environments.base import BaseEnvironment
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from src.terminal_agent.formatting import format_exec_result, shell_single_quote
from src.terminal_agent.toolkit.schemas import (
    CheckCommandAvailableInput,
    CompareOutputInput,
    ExecShellInput,
    ExtractTestSignalsInput,
    InspectServicesInput,
    ListPortsInput,
    ListProcessesInput,
    ProposeNextActionsInput,
    RunProgramWithInputInput,
    RunTestsInput,
    SummarizeFailuresInput,
    WaitForPortInput,
)


def register_verify_tools(tools: list[Any], agent: Any, environment: BaseEnvironment) -> None:
    working_dir = agent.working_dir
    tool_timeout_sec = agent.tool_timeout_sec
    test_timeout_sec = agent.test_timeout_sec

    def _analysis_command_update(
        *,
        runtime: ToolRuntime,
        result: str,
        signal_lines: list[str] | None = None,
        summary: str | None = None,
        next_actions: list[str] | None = None,
    ) -> Command:
        updates: dict[str, Any] = {
            "messages": [
                ToolMessage(
                    content=result,
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
        if signal_lines is not None:
            existing = list(runtime.state.get("failure_signals", []))
            for line in signal_lines:
                if line and line not in existing:
                    existing.append(line)
            updates["failure_signals"] = existing
        if summary is not None:
            updates["failure_summary"] = summary
        if next_actions is not None:
            existing_actions = list(runtime.state.get("next_actions", []))
            for action in next_actions:
                if action and action not in existing_actions:
                    existing_actions.append(action)
            updates["next_actions"] = existing_actions
        return Command(update=updates)

    @tool(args_schema=ExecShellInput)
    async def exec_shell(
        command: str,
        cwd: str | None = None,
        timeout_sec: int = tool_timeout_sec,
    ) -> str:
        """Run a shell command in the task environment and return stdout, stderr, and exit code."""
        return await agent._run_shell_tool(
            tool_name="exec_shell",
            environment=environment,
            command=command,
            cwd=cwd or working_dir,
            timeout_sec=timeout_sec,
        )

    @tool(args_schema=CheckCommandAvailableInput)
    async def check_command_available(command_name: str) -> str:
        """Check whether a command exists in the current environment."""
        quoted = shlex.quote(command_name)
        command = textwrap.dedent(
            f"""
            if command -v {quoted} >/dev/null 2>&1; then
              printf '{{"command":"%s","available":true,"path":"%s"}}\n' {quoted} "$(command -v {quoted})"
            else
              printf '{{"command":"%s","available":false}}\n' {quoted}
            fi
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="check_command_available",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=15,
        )

    @tool(args_schema=RunProgramWithInputInput)
    async def run_program_with_input(
        command: str,
        stdin_text: str = "",
        cwd: str | None = None,
        timeout_sec: int = test_timeout_sec,
    ) -> str:
        """Run a command and pipe provided stdin into it."""
        wrapped = textwrap.dedent(
            f"""
            cat <<'EOF_STDIN' | {command}
            {stdin_text}
            EOF_STDIN
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="run_program_with_input",
            environment=environment,
            command=wrapped,
            cwd=cwd or working_dir,
            timeout_sec=timeout_sec,
        )

    @tool(args_schema=CompareOutputInput)
    async def compare_output(
        actual: str,
        expected: str,
        mode: str = "exact",
    ) -> str:
        """Compare two text blobs. Modes: exact, contains, diff."""
        normalized_mode = mode.strip().lower() or "exact"
        action = f"compare_output mode={normalized_mode!r}"
        agent._emit("tool.start compare_output host")
        agent._emit_block("tool.command", action)

        match = False
        payload: dict[str, Any] = {"match": False, "mode": normalized_mode}
        return_code = 1

        if normalized_mode == "contains":
            match = expected in actual
        elif normalized_mode == "diff":
            match = actual == expected
            if not match:
                diff = "".join(
                    difflib.unified_diff(
                        expected.splitlines(keepends=True),
                        actual.splitlines(keepends=True),
                        fromfile="expected",
                        tofile="actual",
                        n=3,
                    )
                )
                payload["diff"] = diff[:4000]
        else:
            normalized_mode = "exact"
            payload["mode"] = normalized_mode
            match = actual == expected

        payload["match"] = match
        if not match:
            payload["expected_excerpt"] = expected[:1000]
            payload["actual_excerpt"] = actual[:1000]
        else:
            return_code = 0

        formatted = format_exec_result(
            action,
            return_code,
            json.dumps(payload, ensure_ascii=False, indent=2),
            "",
        )
        agent._emit(f"tool.finish compare_output rc={return_code}")
        agent._emit_block("tool.result", formatted)
        return formatted

    @tool(args_schema=RunTestsInput)
    async def run_tests(
        command: str,
        cwd: str | None = None,
        timeout_sec: int = test_timeout_sec,
    ) -> str:
        """Run verification commands such as tests, build checks, or linters."""
        return await agent._run_shell_tool(
            tool_name="run_tests",
            environment=environment,
            command=command,
            cwd=cwd or working_dir,
            timeout_sec=timeout_sec,
        )

    @tool(args_schema=ListProcessesInput)
    async def list_processes(max_entries: int = 80) -> str:
        """List running processes with a portable fallback strategy."""
        command = textwrap.dedent(
            f"""
            if command -v ps >/dev/null 2>&1; then
              ps -eo pid,ppid,stat,comm,args 2>/dev/null | head -n {int(max_entries)}
            else
              printf 'ps not available\n'
            fi
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="list_processes",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=20,
        )

    @tool(args_schema=ListPortsInput)
    async def list_ports(max_entries: int = 80) -> str:
        """List listening ports with best-effort fallbacks."""
        command = textwrap.dedent(
            f"""
            if command -v ss >/dev/null 2>&1; then
              ss -ltnp 2>/dev/null | head -n {int(max_entries)}
            elif command -v netstat >/dev/null 2>&1; then
              netstat -ltn 2>/dev/null | head -n {int(max_entries)}
            elif command -v lsof >/dev/null 2>&1; then
              lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | head -n {int(max_entries)}
            else
              printf 'no supported port-inspection command available\n' >&2
              exit 127
            fi
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="list_ports",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=20,
        )

    @tool(args_schema=WaitForPortInput)
    async def wait_for_port(
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout_sec: int = 30,
        interval_sec: int = 1,
    ) -> str:
        """Wait until a TCP port becomes reachable."""
        command = textwrap.dedent(
            f"""
            end=$(( $(date +%s) + {int(timeout_sec)} ))
            while [ "$(date +%s)" -le "$end" ]; do
              if command -v nc >/dev/null 2>&1; then
                if nc -z {shlex.quote(host)} {int(port)} >/dev/null 2>&1; then
                  printf '{{"ready":true,"host":"%s","port":%s}}\n' {shlex.quote(host)} {int(port)}
                  exit 0
                fi
              elif command -v bash >/dev/null 2>&1; then
                if HOST={shlex.quote(host)} PORT={int(port)} bash -lc 'exec 3<>/dev/tcp/$HOST/$PORT' >/dev/null 2>&1; then
                  printf '{{"ready":true,"host":"%s","port":%s}}\n' {shlex.quote(host)} {int(port)}
                  exit 0
                fi
              fi
              sleep {int(interval_sec)}
            done
            printf '{{"ready":false,"host":"%s","port":%s}}\n' {shlex.quote(host)} {int(port)}
            exit 1
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="wait_for_port",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=max(int(timeout_sec) + 5, 10),
        )

    @tool(args_schema=InspectServicesInput)
    async def inspect_services(max_entries: int = 50) -> str:
        """Collect a compact snapshot of processes and listening ports."""
        command = textwrap.dedent(
            f"""
            printf '== processes ==\n'
            if command -v ps >/dev/null 2>&1; then
              ps -eo pid,ppid,stat,comm,args 2>/dev/null | head -n {int(max_entries)}
            else
              printf 'ps not available\n'
            fi
            printf '\n== listening ports ==\n'
            if command -v ss >/dev/null 2>&1; then
              ss -ltnp 2>/dev/null | head -n {int(max_entries)}
            elif command -v netstat >/dev/null 2>&1; then
              netstat -ltn 2>/dev/null | head -n {int(max_entries)}
            elif command -v lsof >/dev/null 2>&1; then
              lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | head -n {int(max_entries)}
            else
              printf 'no supported port-inspection command available\n' >&2
              exit 127
            fi
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="inspect_services",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=25,
        )

    @tool(args_schema=ExtractTestSignalsInput)
    async def extract_test_signals(
        text: str,
        max_lines: int = 40,
        runtime: ToolRuntime | None = None,
    ) -> str | Command:
        """Extract likely failure lines from noisy test or verifier output."""
        command = textwrap.dedent(
            f"""
            cat <<'EOF_TEXT' > /tmp/agent_extract_test_signals.txt
            {text}
            EOF_TEXT
            if command -v rg >/dev/null 2>&1; then
              rg -n -i 'error|failed|failure|assert|exception|traceback|wrong output|segmentation|timeout|not found|mismatch' /tmp/agent_extract_test_signals.txt | head -n {int(max_lines)} || true
            else
              grep -Ein 'error|failed|failure|assert|exception|traceback|wrong output|segmentation|timeout|not found|mismatch' /tmp/agent_extract_test_signals.txt | head -n {int(max_lines)} || true
            fi
            rm -f /tmp/agent_extract_test_signals.txt
            """
        ).strip()
        result = await agent._run_shell_tool(
            tool_name="extract_test_signals",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=20,
        )
        if runtime is None:
            return result
        payload = json.loads(result)
        signal_lines = [
            line.strip()
            for line in str(payload.get("stdout", "") or "").splitlines()
            if line.strip()
        ]
        return _analysis_command_update(
            runtime=runtime,
            result=result,
            signal_lines=signal_lines,
        )

    @tool(args_schema=SummarizeFailuresInput)
    async def summarize_failures(
        text: str,
        max_lines: int = 30,
        runtime: ToolRuntime | None = None,
    ) -> str | Command:
        """Summarize recurring failure lines by frequency."""
        command = textwrap.dedent(
            f"""
            cat <<'EOF_TEXT' > /tmp/agent_summarize_failures.txt
            {text}
            EOF_TEXT
            if command -v rg >/dev/null 2>&1; then
              rg -i 'error|failed|failure|assert|exception|traceback|wrong output|segmentation|timeout|not found|mismatch' /tmp/agent_summarize_failures.txt \
                | sed 's/^[[:space:]]*//' \
                | sort | uniq -c | sort -nr | head -n {int(max_lines)}
            else
              grep -Ei 'error|failed|failure|assert|exception|traceback|wrong output|segmentation|timeout|not found|mismatch' /tmp/agent_summarize_failures.txt \
                | sed 's/^[[:space:]]*//' \
                | sort | uniq -c | sort -nr | head -n {int(max_lines)}
            fi
            rm -f /tmp/agent_summarize_failures.txt
            """
        ).strip()
        result = await agent._run_shell_tool(
            tool_name="summarize_failures",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=20,
        )
        if runtime is None:
            return result
        payload = json.loads(result)
        summary = str(payload.get("stdout", "") or "").strip()
        return _analysis_command_update(
            runtime=runtime,
            result=result,
            summary=summary,
        )

    @tool(args_schema=ProposeNextActionsInput)
    async def propose_next_actions(
        text: str,
        max_actions: int = 8,
        runtime: ToolRuntime | None = None,
    ) -> str | Command:
        """Generate heuristic next debugging actions from failure text."""
        command = textwrap.dedent(
            f"""
            cat <<'EOF_TEXT' > /tmp/agent_propose_next_actions.txt
            {text}
            EOF_TEXT
            actions_file=/tmp/agent_next_actions.txt
            : > "$actions_file"
            if grep -Eiq 'not found|command not found|no such file|cannot open' /tmp/agent_propose_next_actions.txt; then
              printf '%s\n' 'Check paths, required files, and missing commands in the environment.' >> "$actions_file"
            fi
            if grep -Eiq 'assert|wrong output|mismatch|expected' /tmp/agent_propose_next_actions.txt; then
              printf '%s\n' 'Run the program on the failing sample input and compare actual output with expected output.' >> "$actions_file"
            fi
            if grep -Eiq 'compile|compiler|gcc|clang|undefined reference|syntax error' /tmp/agent_propose_next_actions.txt; then
              printf '%s\n' 'Re-run the build command and inspect compiler diagnostics around the referenced file and symbol.' >> "$actions_file"
            fi
            if grep -Eiq 'timeout|hang' /tmp/agent_propose_next_actions.txt; then
              printf '%s\n' 'Reproduce with a shorter command, inspect processes, and check whether the service or program is blocked.' >> "$actions_file"
            fi
            if grep -Eiq 'segmentation|abort|core dumped' /tmp/agent_propose_next_actions.txt; then
              printf '%s\n' 'Inspect the crash path, reproduce with smaller input, and check recent edits affecting memory or pointer usage.' >> "$actions_file"
            fi
            if grep -Eiq 'port|connection refused|address already in use|listen' /tmp/agent_propose_next_actions.txt; then
              printf '%s\n' 'Inspect listening ports and processes, then wait for or restart the expected service.' >> "$actions_file"
            fi
            if [ ! -s "$actions_file" ]; then
              printf '%s\n' 'Identify the first concrete failing command, reproduce it directly, and inspect the smallest relevant file set.' >> "$actions_file"
            fi
            head -n {int(max_actions)} "$actions_file"
            rm -f /tmp/agent_propose_next_actions.txt "$actions_file"
            """
        ).strip()
        result = await agent._run_shell_tool(
            tool_name="propose_next_actions",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=20,
        )
        if runtime is None:
            return result
        payload = json.loads(result)
        next_actions = [
            line.strip()
            for line in str(payload.get("stdout", "") or "").splitlines()
            if line.strip()
        ]
        return _analysis_command_update(
            runtime=runtime,
            result=result,
            next_actions=next_actions,
        )

    tools.extend(
        [
            exec_shell,
            check_command_available,
            run_program_with_input,
            compare_output,
            run_tests,
            list_processes,
            list_ports,
            wait_for_port,
            inspect_services,
            extract_test_signals,
            summarize_failures,
            propose_next_actions,
        ]
    )
