import shlex
import textwrap
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment
from langchain.tools import tool

from src.terminal_agent.formatting import make_write_command, shell_single_quote
from src.terminal_agent.toolkit.schemas import (
    AppendFileInput,
    ApplyUnifiedDiffInput,
    CopyFileInput,
    DeleteFileInput,
    MakeDirectoryInput,
    MoveFileInput,
    ReplaceInFileInput,
    WriteFileInput,
    WriteJsonInput,
)


def register_edit_tools(tools: list[Any], agent: Any, environment: BaseEnvironment) -> None:
    working_dir = agent.working_dir

    @tool(args_schema=WriteFileInput)
    async def write_file(path: str, content: str) -> str:
        """Write full file contents, creating parent directories if needed."""
        command = make_write_command(path=path, content=content, append=False)
        return await agent._run_shell_tool(
            tool_name="write_file",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=60,
        )

    @tool(args_schema=WriteJsonInput)
    async def write_json(path: str, content: str) -> str:
        """Write pretty JSON from a JSON string into a file."""
        write_command = make_write_command(path=path, content=content, append=False)
        quoted_path = shlex.quote(path)
        command = textwrap.dedent(
            f"""
            {write_command}
            if command -v jq >/dev/null 2>&1; then
              tmp_file=$(mktemp)
              jq . {quoted_path} > "$tmp_file" && mv "$tmp_file" {quoted_path}
            elif command -v perl >/dev/null 2>&1; then
              tmp_file=$(mktemp)
              TARGET={quoted_path} TMP_FILE="$tmp_file" perl -MJSON::PP -e '
                use strict;
                use warnings;
                my $target = $ENV{{TARGET}};
                my $tmp = $ENV{{TMP_FILE}};
                open my $fh, "<", $target or die "open failed";
                local $/;
                my $text = <$fh>;
                close $fh;
                my $obj = JSON::PP->new->decode($text);
                open my $out, ">", $tmp or die "tmp write failed";
                print $out JSON::PP->new->ascii->pretty->canonical->encode($obj);
                close $out;
              ' && mv "$tmp_file" {quoted_path}
            else
              printf 'write_json requires jq or perl JSON::PP in the task environment\n' >&2
              exit 127
            fi
            printf 'ok\n'
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="write_json",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=60,
        )

    @tool(args_schema=MakeDirectoryInput)
    async def make_directory(path: str) -> str:
        """Create a directory and any missing parents."""
        command = f"mkdir -p {shlex.quote(path)} && printf 'ok\\n'"
        return await agent._run_shell_tool(
            tool_name="make_directory",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=30,
        )

    @tool(args_schema=AppendFileInput)
    async def append_file(path: str, content: str) -> str:
        """Append text to a file, creating parent directories if needed."""
        command = make_write_command(path=path, content=content, append=True)
        return await agent._run_shell_tool(
            tool_name="append_file",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=60,
        )

    @tool(args_schema=ReplaceInFileInput)
    async def replace_in_file(
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> str:
        """Replace text in a file. Supports multiline replacements."""
        quoted_path = shell_single_quote(path)
        quoted_old_text = shell_single_quote(old_text)
        quoted_new_text = shell_single_quote(new_text)
        replace_all_value = "1" if replace_all else "0"
        command = textwrap.dedent(
            f"""
            if command -v perl >/dev/null 2>&1; then
              TARGET={quoted_path} OLD_TEXT={quoted_old_text} NEW_TEXT={quoted_new_text} REPLACE_ALL={replace_all_value} perl -e '
                use strict;
                use warnings;
                my $path = $ENV{{TARGET}};
                my $old_text = $ENV{{OLD_TEXT}};
                my $new_text = $ENV{{NEW_TEXT}};
                my $replace_all = $ENV{{REPLACE_ALL}};
                open my $fh, "<", $path or die "open failed";
                local $/;
                my $text = <$fh>;
                close $fh;
                die "old_text not found" if index($text, $old_text) < 0;
                if ($replace_all) {{
                  $text =~ s/\\Q$old_text\\E/$new_text/g;
                }} else {{
                  $text =~ s/\\Q$old_text\\E/$new_text/;
                }}
                open my $out, ">", $path or die "write failed";
                print $out $text;
                close $out;
              '
              printf 'ok\n'
            else
              printf 'replace_in_file requires perl in the task environment\n' >&2
              exit 1
            fi
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="replace_in_file",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=60,
        )

    @tool(args_schema=CopyFileInput)
    async def copy_file(source: str, destination: str) -> str:
        """Copy a file or directory to a new location."""
        command = textwrap.dedent(
            f"""
            mkdir -p {shlex.quote(str(Path(destination).parent))}
            cp -R {shlex.quote(source)} {shlex.quote(destination)}
            printf 'ok\n'
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="copy_file",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=60,
        )

    @tool(args_schema=MoveFileInput)
    async def move_file(source: str, destination: str) -> str:
        """Move or rename a file or directory."""
        command = textwrap.dedent(
            f"""
            mkdir -p {shlex.quote(str(Path(destination).parent))}
            mv {shlex.quote(source)} {shlex.quote(destination)}
            printf 'ok\n'
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="move_file",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=60,
        )

    @tool(args_schema=DeleteFileInput)
    async def delete_file(path: str, recursive: bool = False, force: bool = True) -> str:
        """Delete a file or directory."""
        flags: list[str] = []
        if recursive:
            flags.append("-r")
        if force:
            flags.append("-f")
        command = f"rm {' '.join(flags)} {shlex.quote(path)} && printf 'ok\\n'"
        return await agent._run_shell_tool(
            tool_name="delete_file",
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=30,
        )

    @tool(args_schema=ApplyUnifiedDiffInput)
    async def apply_unified_diff(diff: str, cwd: str | None = None) -> str:
        """Apply a unified diff using patch or git apply when available."""
        command = textwrap.dedent(
            f"""
            diff_file=$(mktemp)
            cat > "$diff_file" <<'EOF_DIFF'
            {diff}
            EOF_DIFF
            if command -v git >/dev/null 2>&1; then
              git apply --whitespace=nowarn "$diff_file"
            elif command -v patch >/dev/null 2>&1; then
              patch -p0 < "$diff_file"
            else
              printf 'apply_unified_diff requires git or patch in the task environment and cannot safely emulate unified diff application without them\n' >&2
              rm -f "$diff_file"
              exit 1
            fi
            rm -f "$diff_file"
            printf 'ok\n'
            """
        ).strip()
        return await agent._run_shell_tool(
            tool_name="apply_unified_diff",
            environment=environment,
            command=command,
            cwd=cwd or working_dir,
            timeout_sec=60,
        )

    tools.extend(
        [
            write_file,
            write_json,
            make_directory,
            append_file,
            replace_in_file,
            copy_file,
            move_file,
            delete_file,
            apply_unified_diff,
        ]
    )
