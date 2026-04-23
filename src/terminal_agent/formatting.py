import json
import shlex
import uuid
from pathlib import Path
from typing import Any

from src.terminal_agent.constants import MAX_TOOL_OUTPUT_CHARS


def truncate_text(text: str | None, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n...[truncated {omitted} chars]..."


def format_exec_result(
    command: str,
    return_code: int,
    stdout: str | None,
    stderr: str | None,
) -> str:
    payload = {
        "command": command,
        "return_code": return_code,
        "stdout": truncate_text(stdout),
        "stderr": truncate_text(stderr),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def shell_single_quote(value: str) -> str:
    return shlex.quote(value)


def make_write_command(path: str, content: str, append: bool = False) -> str:
    delimiter = f"EOF_{uuid.uuid4().hex}"
    operator = ">>" if append else ">"
    parent = shlex.quote(str(Path(path).parent))
    target = shlex.quote(path)
    return (
        f"mkdir -p {parent} && cat {operator} {target} <<'{delimiter}'\n"
        f"{content}\n"
        f"{delimiter}"
    )


def extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)
