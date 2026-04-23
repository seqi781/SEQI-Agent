import shlex
import textwrap
from typing import Any

from harbor.environments.base import BaseEnvironment
from langchain.tools import tool

from src.terminal_agent.toolkit.schemas import (
    FileInfoInput,
    FindFilesInput,
    InspectEnvInput,
    InspectFileBytesInput,
    ListFilesInput,
    ReadFileInput,
    ReadJsonInput,
    ReadManyFilesInput,
    ScanStringsInput,
    SearchTextInput,
)


def register_inspect_tools(tools: list[Any], agent: Any, environment: BaseEnvironment) -> None:
    working_dir = agent.working_dir
    excluded_paths = [
        "-not -path '*/.git*'",
        f"-not -path '{agent.helper_dir}'",
        f"-not -path '{agent.helper_dir}/*'",
    ]
    excluded_expr = " ".join(excluded_paths)

    @tool(args_schema=ListFilesInput)
    async def list_files(
        path: str = ".",
        max_entries: int = 200,
        include_hidden: bool = False,
    ) -> str:
        """List files under a directory to quickly inspect the repository structure."""
        hidden = "-not -path '*/.*'" if not include_hidden else ""
        quoted = shlex.quote(path)
        command = (
            f"find {quoted} {hidden} {excluded_expr} -mindepth 1 -maxdepth 4 "
            "-print | sort | head -n "
            f"{int(max_entries)}"
        )
        return await agent._run_shell_tool(
            tool_name="list_files",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=30,
        )

    @tool(args_schema=FindFilesInput)
    async def find_files(
        pattern: str,
        path: str = ".",
        max_entries: int = 200,
    ) -> str:
        """Find files by glob-style filename pattern such as *.py or *test*."""
        quoted_path = shlex.quote(path)
        quoted_pattern = shlex.quote(pattern)
        command = (
            f"find {quoted_path} {excluded_expr} -type f -name {quoted_pattern} "
            f"| sort | head -n {int(max_entries)}"
        )
        return await agent._run_shell_tool(
            tool_name="find_files",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=30,
        )

    @tool(args_schema=SearchTextInput)
    async def search_text(
        pattern: str,
        path: str = ".",
        max_matches: int = 200,
    ) -> str:
        """Search text with ripgrep. Use for symbols, errors, TODOs, tests, and config lookup."""
        quoted_pattern = shlex.quote(pattern)
        quoted_path = shlex.quote(path)
        command = (
            f"(command -v rg >/dev/null 2>&1 && "
            f"rg -n --hidden --glob '!*.git' --glob '!.agent-tools/**' --max-count {int(max_matches)} "
            f"{quoted_pattern} {quoted_path}) || "
            f"grep -REIn --exclude-dir=.git --exclude-dir=.agent-tools {quoted_pattern} {quoted_path} | head -n {int(max_matches)}"
        )
        return await agent._run_shell_tool(
            tool_name="search_text",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=60,
        )

    @tool(args_schema=FileInfoInput)
    async def file_info(path: str) -> str:
        """Return basic file metadata. Only include line counts for text files."""
        quoted_path = shlex.quote(path)
        command = textwrap.dedent(
            f"""
            target={quoted_path}
            if [ -e "$target" ]; then
              if [ -d "$target" ]; then
                printf '{{"path":"%s","exists":true,"kind":"directory"}}\n' "$target"
                exit 0
              fi
              bytes=$(wc -c < "$target" | tr -d ' ')
              is_text=true
              if command -v perl >/dev/null 2>&1; then
                if TARGET="$target" perl -e '
                  use strict;
                  use warnings;
                  my $path = $ENV{{TARGET}};
                  open my $fh, "<", $path or die "open failed";
                  binmode $fh;
                  read($fh, my $buf, 4096);
                  close $fh;
                  exit(index($buf, "\\0") >= 0 ? 0 : 1);
                '; then
                  is_text=false
                fi
              elif command -v od >/dev/null 2>&1; then
                if od -An -tx1 -N 4096 -v "$target" 2>/dev/null | tr -d ' \n' | grep -qi '00'; then
                  is_text=false
                fi
              fi
              if [ "$is_text" = true ]; then
                lines=$(wc -l < "$target" | tr -d ' ')
                printf '{{"path":"%s","exists":true,"kind":"file","bytes":%s,"is_text":true,"lines":%s}}\n' "$target" "$bytes" "$lines"
              else
                printf '{{"path":"%s","exists":true,"kind":"file","bytes":%s,"is_text":false}}\n' "$target" "$bytes"
              fi
            else
              printf '{{"path":"%s","exists":false}}\n' "$target"
            fi
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="file_info",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=30,
        )

    @tool(args_schema=ReadFileInput)
    async def read_file(
        path: str,
        start_line: int = 1,
        end_line: int = 200,
    ) -> str:
        """Read a file range with line numbers."""
        start = max(1, int(start_line))
        end = max(start, int(end_line))
        quoted_path = shlex.quote(path)
        command = f"nl -ba {quoted_path} | sed -n '{start},{end}p'"
        return await agent._run_shell_tool(
            tool_name="read_file",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=30,
        )

    @tool(args_schema=ReadManyFilesInput)
    async def read_many_files(
        paths: list[str],
        start_line: int = 1,
        end_line: int = 200,
    ) -> str:
        """Read the same line range from multiple files in one call."""
        start = max(1, int(start_line))
        end = max(start, int(end_line))
        if not paths:
            return "no paths provided"
        quoted_paths = " ".join(shlex.quote(path) for path in paths[:20])
        command = textwrap.dedent(
            f"""
            for file in {quoted_paths}; do
              printf '===== %s =====\n' "$file"
              nl -ba "$file" | sed -n '{start},{end}p' || true
              printf '\n'
            done
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="read_many_files",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=60,
        )

    @tool(args_schema=ReadJsonInput)
    async def read_json(path: str) -> str:
        """Pretty-print a JSON file for inspection."""
        quoted_path = shlex.quote(path)
        command = textwrap.dedent(
            f"""
            if command -v jq >/dev/null 2>&1; then
              jq . {quoted_path}
            else
              cat {quoted_path}
            fi
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="read_json",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=30,
        )

    @tool(args_schema=InspectEnvInput)
    async def inspect_env(max_env_vars: int = 80) -> str:
        """Inspect basic environment information, selected env vars, and available shells."""
        command = textwrap.dedent(
            f"""
            printf '== uname ==\n'
            uname -a 2>/dev/null || true
            printf '\n== pwd ==\n'
            pwd
            printf '\n== whoami ==\n'
            whoami 2>/dev/null || true
            printf '\n== shell ==\n'
            printf '%s\n' "${{SHELL:-}}"
            printf '\n== env ==\n'
            env | sort | head -n {int(max_env_vars)}
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="inspect_env",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=30,
        )

    @tool(args_schema=InspectFileBytesInput)
    async def inspect_file_bytes(
        path: str,
        offset: int = 0,
        length: int = 256,
    ) -> str:
        """Inspect raw bytes from a file region using od or xxd. Useful for file headers and binary format probing."""
        start = max(0, int(offset))
        size = max(1, min(int(length), 4096))
        quoted_path = shlex.quote(path)
        command = textwrap.dedent(
            f"""
            if command -v od >/dev/null 2>&1; then
              od -Ax -tx1 -v -j {start} -N {size} {quoted_path}
            elif command -v xxd >/dev/null 2>&1; then
              xxd -g 1 -s {start} -l {size} {quoted_path}
            else
              printf 'inspect_file_bytes requires od or xxd in the task environment\n' >&2
              exit 127
            fi
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="inspect_file_bytes",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=30,
        )

    @tool(args_schema=ScanStringsInput)
    async def scan_strings(
        path: str,
        pattern: str = "",
        max_matches: int = 80,
        min_length: int = 4,
    ) -> str:
        """Extract printable strings from a file and optionally filter by pattern."""
        quoted_path = shlex.quote(path)
        grep_filter = ""
        if pattern.strip():
            quoted_pattern = shlex.quote(pattern)
            grep_filter = f" | grep -Ein {quoted_pattern} | head -n {int(max_matches)}"
        else:
            grep_filter = f" | head -n {int(max_matches)}"
        command = textwrap.dedent(
            f"""
            if command -v strings >/dev/null 2>&1; then
              strings -n {max(1, int(min_length))} {quoted_path}{grep_filter}
            elif command -v perl >/dev/null 2>&1; then
              TARGET={quoted_path} MIN_LENGTH={max(1, int(min_length))} perl -e '
                use strict;
                use warnings;
                my $path = $ENV{{TARGET}};
                my $min = int($ENV{{MIN_LENGTH}} || 4);
                open my $fh, "<", $path or die "open failed";
                binmode $fh;
                local $/;
                my $data = <$fh>;
                close $fh;
                while ($data =~ /([ -~]{{$min,}})/g) {{
                  print "$1\n";
                }}
              '{grep_filter}
            elif command -v od >/dev/null 2>&1; then
              od -An -v -c {quoted_path} \
                | sed 's/  */ /g' \
                | tr ' ' '\n' \
                | tr -d '\r' \
                | perl -ne 'BEGIN {{ $m={max(1, int(min_length))}; $s="" }} chomp; if (/^[ -~]$/) {{ $s .= $_ }} else {{ print "$s\n" if length($s) >= $m; $s="" }} END {{ print "$s\n" if length($s) >= $m }}' \
                {grep_filter}
            else
              printf 'scan_strings requires strings, perl, or od in the task environment\n' >&2
              exit 127
            fi
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="scan_strings",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=60,
        )

    tools.extend(
        [
            list_files,
            find_files,
            search_text,
            file_info,
            read_file,
            read_many_files,
            read_json,
            inspect_env,
            inspect_file_bytes,
            scan_strings,
        ]
    )
