import json
import os
import shlex
import html
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment
from langchain.tools import tool

from src.terminal_agent.formatting import format_exec_result, make_write_command
from src.terminal_agent.toolkit.schemas import (
    BraveWebSearchInput,
    DownloadUrlInput,
    FetchUrlInput,
)


BRAVE_WEB_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_USER_AGENT = "langgraph-terminal-agent/0.7 (+https://example.invalid)"
MAX_TEXT_DOWNLOAD_BYTES = 200_000


def _is_probably_text(content_type: str, data: bytes) -> bool:
    lowered = content_type.lower()
    if lowered.startswith("text/"):
        return True
    if any(
        marker in lowered
        for marker in [
            "json",
            "javascript",
            "xml",
            "yaml",
            "yml",
            "toml",
            "csv",
            "x-sh",
            "shellscript",
        ]
    ):
        return True
    if b"\x00" in data[:4096]:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _decode_text(data: bytes, content_type: str) -> str:
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
    try:
        return data.decode(charset, errors="replace")
    except LookupError:
        return data.decode("utf-8", errors="replace")


def _extract_text_for_agent(text: str, content_type: str) -> str:
    lowered = content_type.lower()
    if "html" not in lowered:
        return text
    cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def _host_request(url: str, *, headers: dict[str, str] | None = None, timeout_sec: int = 20) -> tuple[bytes, str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        data = response.read()
        content_type = response.headers.get("Content-Type", "application/octet-stream")
        final_url = response.geturl()
    return data, content_type, final_url


async def download_via_host_fallback(
    *,
    agent: Any,
    environment: BaseEnvironment,
    working_dir: str,
    tool_name: str,
    url: str,
    destination: str,
    executable: bool,
) -> str:
    action = (
        f"{tool_name} url={url!r} destination={destination!r} "
        f"executable={bool(executable)}"
    )
    agent._emit(f"tool.start {tool_name} host-fallback")
    agent._emit_block("tool.command", action)
    try:
        data, content_type, _ = _host_request(
            url,
            timeout_sec=max(getattr(agent, "tool_timeout_sec", 20), 20),
        )
        if len(data) > MAX_TEXT_DOWNLOAD_BYTES:
            raise ValueError(
                f"fallback download supports files up to {MAX_TEXT_DOWNLOAD_BYTES} bytes"
            )
        quoted_destination = shlex.quote(destination)
        if _is_probably_text(content_type, data):
            text = _decode_text(data, content_type)
            command = make_write_command(destination, text, append=False)
        else:
            if not getattr(agent, "_capabilities", {}).get("perl"):
                raise ValueError(
                    "binary host fallback requires perl in the task environment"
                )
            hex_payload = data.hex()
            command = (
                f"mkdir -p {shlex.quote(str(Path(destination).parent))}\n"
                f"perl -e 'open my $fh, \">\", shift or die \"open failed\"; "
                f"binmode $fh; print $fh pack(\"H*\", shift); close $fh;' "
                f"{quoted_destination} {shlex.quote(hex_payload)}"
            )
        if executable:
            command += f"\nchmod +x {quoted_destination}"
        command += f"\nwc -c {quoted_destination}"
        return await agent._run_shell_tool(
            tool_name=tool_name,
            environment=environment,
            command=command,
            cwd=working_dir,
            timeout_sec=max(getattr(agent, "tool_timeout_sec", 30), 60),
        )
    except (urllib.error.URLError, ValueError) as exc:
        formatted = format_exec_result(action, 1, "", f"{type(exc).__name__}: {exc}")
        agent._emit(f"tool.finish {tool_name} rc=1")
        agent._emit_block("tool.result", formatted)
        return formatted


def register_web_tools(tools: list[Any], agent: Any, environment: BaseEnvironment) -> None:
    working_dir = agent.working_dir

    @tool(args_schema=BraveWebSearchInput)
    async def brave_web_search(
        query: str,
        count: int = 5,
        country: str = "us",
        search_lang: str = "en",
    ) -> str:
        """Search the web via Brave Search. Use when local evidence is insufficient and external information may unblock progress."""
        action = (
            f"brave_web_search query={query!r} count={int(count)} "
            f"country={country!r} search_lang={search_lang!r}"
        )
        agent._emit(f"tool.start brave_web_search host")
        agent._emit_block("tool.command", action)
        api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
        if not api_key:
            formatted = format_exec_result(
                action,
                1,
                "",
                "BRAVE_SEARCH_API_KEY is not set in the agent environment.",
            )
            agent._emit("tool.finish brave_web_search rc=1")
            agent._emit_block("tool.result", formatted)
            return formatted

        params = urllib.parse.urlencode(
            {
                "q": query,
                "count": max(1, min(int(count), 10)),
                "country": country,
                "search_lang": search_lang,
            }
        )
        request_url = f"{BRAVE_WEB_SEARCH_ENDPOINT}?{params}"
        try:
            payload_bytes, content_type, final_url = _host_request(
                request_url,
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            )
            payload = json.loads(_decode_text(payload_bytes, content_type))
            results = payload.get("web", {}).get("results", [])
            lines = [f"Final URL: {final_url}", f"Result count: {len(results)}", ""]
            for index, item in enumerate(results, start=1):
                title = item.get("title", "").strip()
                url = item.get("url", "").strip()
                description = item.get("description", "").strip()
                age = item.get("age", "").strip()
                lines.append(f"{index}. {title}")
                lines.append(url)
                if age:
                    lines.append(f"Age: {age}")
                if description:
                    lines.append(description)
                lines.append("")
            stdout = "\n".join(lines).strip()
            formatted = format_exec_result(action, 0, stdout, "")
            agent._emit("tool.finish brave_web_search rc=0")
            agent._emit_block("tool.result", formatted)
            return formatted
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            formatted = format_exec_result(action, 1, "", f"{type(exc).__name__}: {exc}")
            agent._emit("tool.finish brave_web_search rc=1")
            agent._emit_block("tool.result", formatted)
            return formatted

    @tool(args_schema=FetchUrlInput)
    async def fetch_url(
        url: str,
        max_chars: int = 12000,
    ) -> str:
        """Fetch a URL and return textual content. Prefer for docs, READMEs, raw text, and small public files."""
        action = f"fetch_url url={url!r} max_chars={int(max_chars)}"
        agent._emit(f"tool.start fetch_url host")
        agent._emit_block("tool.command", action)
        try:
            data, content_type, final_url = _host_request(url)
            if not _is_probably_text(content_type, data):
                formatted = format_exec_result(
                    action,
                    1,
                    "",
                    f"URL appears to be binary content ({content_type}). Use download_url instead.",
                )
            else:
                text = _extract_text_for_agent(
                    _decode_text(data[:MAX_TEXT_DOWNLOAD_BYTES], content_type),
                    content_type,
                )
                header = f"Final URL: {final_url}\nContent-Type: {content_type}\n\n"
                stdout = (header + text)[: max(1, int(max_chars))]
                formatted = format_exec_result(action, 0, stdout, "")
        except urllib.error.URLError as exc:
            formatted = format_exec_result(action, 1, "", f"{type(exc).__name__}: {exc}")
        agent._emit(
            f"tool.finish fetch_url rc={json.loads(formatted).get('return_code', 1)}"
        )
        agent._emit_block("tool.result", formatted)
        return formatted

    @tool(args_schema=DownloadUrlInput)
    async def download_url(
        url: str,
        destination: str,
        executable: bool = False,
    ) -> str:
        """Download a public URL into the task environment. Prefers curl/wget in the task env, then falls back to agent-side text download and write."""
        quoted_url = shlex.quote(url)
        quoted_destination = shlex.quote(destination)
        command = (
            f"mkdir -p {shlex.quote(str(Path(destination).parent))} && "
            f"if command -v curl >/dev/null 2>&1; then "
            f"curl -fsSL {quoted_url} -o {quoted_destination}; "
            f"elif command -v wget >/dev/null 2>&1; then "
            f"wget -qO {quoted_destination} {quoted_url}; "
            f"else exit 127; fi && "
            f"{'chmod +x ' + quoted_destination + ' && ' if executable else ''}"
            f"wc -c {quoted_destination}"
        )
        if agent._capabilities.get("curl") or agent._capabilities.get("wget"):
            return await agent._run_shell_tool(
                tool_name="download_url",
                environment=environment,
                command=command,
                cwd=working_dir,
                timeout_sec=max(agent.tool_timeout_sec, 60),
            )

        return await download_via_host_fallback(
            agent=agent,
            environment=environment,
            working_dir=working_dir,
            tool_name="download_url",
            url=url,
            destination=destination,
            executable=executable,
        )

    tools.extend(
        [
            brave_web_search,
            fetch_url,
            download_url,
        ]
    )
