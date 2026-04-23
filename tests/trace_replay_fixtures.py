"""Trace replay fixtures for regression coverage.

Add new real-world traces here instead of duplicating test logic in
`tests/test_agent_evidence.py`.
"""

TRACE_REPLAY_PAYLOAD_FIXTURES = {
    "negative_verification_without_false_success_or_browser_block": [
        {
            "_tool_name": "check_command_available",
            "return_code": 0,
            "stdout": '{"command":"pytest","available":false}',
            "stderr": "",
        },
        {
            "_tool_name": "check_command_available",
            "return_code": 0,
            "stdout": '{"command":"google-chrome","available":false}',
            "stderr": "",
        },
        {
            "_tool_name": "check_command_available",
            "return_code": 0,
            "stdout": '{"command":"chromium","available":true,"path":"/usr/bin/chromium"}',
            "stderr": "",
        },
        {
            "_tool_name": "exec_shell",
            "return_code": 0,
            "stdout": "NO_ALERT TimeoutException\n",
            "stderr": "",
        },
    ],
}

TRACE_REPLAY_PAYLOAD_EXPECTATIONS = {
    "negative_verification_without_false_success_or_browser_block": {
        "verification_state": "negatively_verified",
        "verification_summary": "NO_ALERT TimeoutException",
        "blocked_verifiers": ["pytest_missing"],
        "verified_failures": ["NO_ALERT TimeoutException"],
        "verified_successes": [],
    }
}

TRACE_REPLAY_STATE_FIXTURES = {
    "followup_guard_state": {
        "helper_roles": {"/app/.agent-tools/verify_alert.py": "verifier"},
        "blocked_verifiers": ["pytest_missing"],
        "rejected_solution_patterns": ["on*_attributes", "script_tags", "banned_tags"],
        "evidence_log": [
            {
                "type": "environment",
                "claim": "chromium_present",
                "scope": "environment",
                "confidence": "high",
                "source": "check_command_available",
                "detail": '{"command":"chromium","available":true,"path":"/usr/bin/chromium"}',
            }
        ],
    }
}

TRACE_REPLAY_GUARD_EXPECTATIONS = {
    "followup_guard_state": {
        "edit_tool": "write_file",
        "edit_args": {
            "path": "/app/out.html",
            "content": '<img src=x onerror=alert(1)><script>alert(1)</script>',
        },
        "edit_contains": ["on*_attributes", "script_tags"],
        "probe_tool": "check_command_available",
        "probe_cases": [
            {"args": {"command_name": "pytest"}, "contains": ["verify_alert.py"]},
            {"args": {"command_name": "google-chrome"}, "contains": ["verify_alert.py"]},
        ],
    }
}
