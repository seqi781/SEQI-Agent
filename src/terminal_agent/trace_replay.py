import json
import re
from pathlib import Path
from typing import Any


def sanitize_fixture_name(raw: str) -> str:
    lowered = raw.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_") or "trace_fixture"


def extract_tool_payloads(trace: dict[str, Any]) -> list[dict[str, Any]]:
    messages = trace.get("outputs", {}).get("messages", [])
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if message.get("type") != "tool":
            continue
        tool_name = str(message.get("name", "")).strip()
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        payloads.append(
            {
                "_tool_name": tool_name,
                "return_code": parsed.get("return_code", 0),
                "stdout": parsed.get("stdout", ""),
                "stderr": parsed.get("stderr", ""),
            }
        )
    return payloads


def extract_payload_expectation(trace: dict[str, Any]) -> dict[str, Any]:
    outputs = trace.get("outputs", {})
    return {
        "verification_state": outputs.get("verification_state", "unverified"),
        "verification_summary": outputs.get("verification_summary", ""),
        "blocked_verifiers": outputs.get("blocked_verifiers", []),
        "verified_failures": outputs.get("verified_failures", []),
        "verified_successes": outputs.get("verified_successes", []),
    }


def extract_state_fixture(trace: dict[str, Any]) -> dict[str, Any]:
    outputs = trace.get("outputs", {})
    return {
        "helper_roles": outputs.get("helper_roles", {}),
        "blocked_verifiers": outputs.get("blocked_verifiers", []),
        "rejected_solution_patterns": outputs.get("rejected_solution_patterns", []),
        "evidence_log": outputs.get("evidence_log", []),
    }


def extract_guard_expectation_stub(trace: dict[str, Any]) -> dict[str, Any]:
    outputs = trace.get("outputs", {})
    rejected_patterns = outputs.get("rejected_solution_patterns", [])
    blocked_verifiers = outputs.get("blocked_verifiers", [])
    helper_roles = outputs.get("helper_roles", {})
    verifier_helpers = [path for path, role in helper_roles.items() if role == "verifier"]

    edit_contains = [pattern for pattern in rejected_patterns if isinstance(pattern, str)]
    probe_cases: list[dict[str, Any]] = []
    for blocker in blocked_verifiers:
        if not isinstance(blocker, str) or not blocker.endswith("_missing"):
            continue
        command_name = blocker.removesuffix("_missing")
        contains: list[str] = []
        if verifier_helpers:
            contains.append(Path(verifier_helpers[0]).name)
        else:
            contains.append("already confirmed missing")
        probe_cases.append({"args": {"command_name": command_name}, "contains": contains})

    return {
        "edit_tool": "write_file",
        "edit_args": {"path": "/app/out.html", "content": "<!-- fill with rejected payload family candidate -->"},
        "edit_contains": edit_contains,
        "probe_tool": "check_command_available",
        "probe_cases": probe_cases,
    }


def fixture_name_from_trace(trace: dict[str, Any], explicit_name: str | None = None) -> str:
    if explicit_name:
        return sanitize_fixture_name(explicit_name)
    outputs = trace.get("outputs", {})
    metadata = trace.get("metadata", {})
    current_step = str(outputs.get("current_step", "")).strip()
    revision = str(metadata.get("revision_id", "")).strip()
    base = current_step or revision or "trace_fixture"
    return sanitize_fixture_name(base)


def render_fixture_snippets(trace: dict[str, Any], fixture_name: str | None = None) -> str:
    name = fixture_name_from_trace(trace, fixture_name)
    payload_fixture = extract_tool_payloads(trace)
    payload_expectation = extract_payload_expectation(trace)
    state_fixture = extract_state_fixture(trace)
    guard_expectation = extract_guard_expectation_stub(trace)
    lines = [
        f"# Fixture name: {name}",
        "",
        "TRACE_REPLAY_PAYLOAD_FIXTURES update:",
        json.dumps({name: payload_fixture}, ensure_ascii=False, indent=4),
        "",
        "TRACE_REPLAY_PAYLOAD_EXPECTATIONS update:",
        json.dumps({name: payload_expectation}, ensure_ascii=False, indent=4),
        "",
        "TRACE_REPLAY_STATE_FIXTURES update:",
        json.dumps({name: state_fixture}, ensure_ascii=False, indent=4),
        "",
        "TRACE_REPLAY_GUARD_EXPECTATIONS update:",
        json.dumps({name: guard_expectation}, ensure_ascii=False, indent=4),
    ]
    return "\n".join(lines)


def load_trace(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
