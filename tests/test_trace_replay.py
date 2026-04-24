import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

from src.terminal_agent.trace_replay import (
    extract_guard_expectation_stub,
    extract_payload_expectation,
    extract_state_fixture,
    extract_tool_payloads,
    fixture_name_from_trace,
    load_trace,
    render_fixture_snippets,
)


class TraceReplayTests(unittest.TestCase):
    def sample_trace(self) -> dict:
        return {
            "outputs": {
                "messages": [
                    {
                        "type": "tool",
                        "name": "check_command_available",
                        "content": '{"command":"check","return_code":0,"stdout":"{\\"command\\":\\"pytest\\",\\"available\\":false}\\n","stderr":""}',
                    },
                    {
                        "type": "tool",
                        "name": "exec_shell",
                        "content": '{"command":"python verify.py","return_code":0,"stdout":"NO_ALERT TimeoutException\\n","stderr":""}',
                    },
                ],
                "verification_state": "negatively_verified",
                "verification_summary": "NO_ALERT TimeoutException",
                "blocked_verifiers": ["pytest_missing"],
                "verified_failures": ["NO_ALERT TimeoutException"],
                "verified_successes": [],
                "helper_roles": {"/app/.agent-tools/verify_alert.py": "verifier"},
                "rejected_solution_patterns": ["on*_attributes"],
                "evidence_log": [
                    {
                        "type": "environment",
                        "claim": "chromium_present",
                        "scope": "environment",
                        "confidence": "high",
                        "source": "check_command_available",
                        "detail": '{"command":"chromium","available":true}',
                    }
                ],
                "current_step": "Print /app/out.html after filtering",
            },
            "metadata": {"revision_id": "de7934e-dirty"},
        }

    def test_extract_tool_payloads(self) -> None:
        payloads = extract_tool_payloads(self.sample_trace())
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["_tool_name"], "check_command_available")
        self.assertEqual(payloads[1]["stdout"], "NO_ALERT TimeoutException\n")

    def test_extract_expectation_and_state_fixture(self) -> None:
        trace = self.sample_trace()
        expectation = extract_payload_expectation(trace)
        state_fixture = extract_state_fixture(trace)
        guard_expectation = extract_guard_expectation_stub(trace)

        self.assertEqual(expectation["verification_state"], "negatively_verified")
        self.assertEqual(expectation["blocked_verifiers"], ["pytest_missing"])
        self.assertEqual(state_fixture["helper_roles"], {"/app/.agent-tools/verify_alert.py": "verifier"})
        self.assertEqual(state_fixture["rejected_solution_patterns"], ["on*_attributes"])
        self.assertEqual(guard_expectation["edit_contains"], ["on*_attributes"])
        self.assertEqual(guard_expectation["probe_cases"][0]["args"], {"command_name": "pytest"})
        self.assertEqual(guard_expectation["probe_cases"][0]["contains"], ["verify_alert.py"])

    def test_fixture_name_and_render(self) -> None:
        trace = self.sample_trace()
        self.assertEqual(fixture_name_from_trace(trace), "print_app_out_html_after_filtering")
        rendered = render_fixture_snippets(trace, "Manual Name")
        self.assertIn("manual_name", rendered)
        self.assertIn("TRACE_REPLAY_PAYLOAD_FIXTURES update:", rendered)
        self.assertIn("TRACE_REPLAY_PAYLOAD_EXPECTATIONS update:", rendered)
        self.assertIn("TRACE_REPLAY_STATE_FIXTURES update:", rendered)
        self.assertIn("TRACE_REPLAY_GUARD_EXPECTATIONS update:", rendered)

    def test_load_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.json"
            path.write_text('{"outputs":{"messages":[]}}')
            loaded = load_trace(str(path))
        self.assertEqual(loaded["outputs"]["messages"], [])

    def test_cli_renders_fixture_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.json"
            path.write_text(
                '{"outputs":{"messages":[],"verification_state":"unverified","verification_summary":"","blocked_verifiers":[],"verified_failures":[],"verified_successes":[],"helper_roles":{},"rejected_solution_patterns":[],"evidence_log":[],"current_step":"Step"},"metadata":{"revision_id":"abc"}}'
            )
            result = subprocess.run(
                [sys.executable, "scripts/trace_to_fixture.py", str(path), "--name", "cli case"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )
        self.assertIn("# Fixture name: cli_case", result.stdout)
        self.assertIn("TRACE_REPLAY_PAYLOAD_FIXTURES update:", result.stdout)
        self.assertIn("TRACE_REPLAY_GUARD_EXPECTATIONS update:", result.stdout)


if __name__ == "__main__":
    unittest.main()
