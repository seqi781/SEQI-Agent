import tempfile
import unittest
from pathlib import Path

from src.terminal_agent.agent import LangGraphTerminalBenchAgent


class AgentEvidenceTests(unittest.TestCase):
    def make_agent(self) -> LangGraphTerminalBenchAgent:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return LangGraphTerminalBenchAgent(logs_dir=Path(temp_dir.name))

    def test_verifier_protocol_drives_positive_and_negative_state(self) -> None:
        agent = self.make_agent()

        pass_evidence = agent._extract_evidence_from_payload(
            {
                "_tool_name": "run_tests",
                "return_code": 0,
                "stdout": "VERIFICATION_RESULT=PASS\nALERT_PRESENT=1\n",
                "stderr": "",
            }
        )
        pass_state = agent._derive_state_from_evidence(pass_evidence, [], [], [], [])
        self.assertEqual(pass_state[0], "positively_verified")
        self.assertIn("alert_triggered", {item["claim"] for item in pass_evidence})

        fail_evidence = agent._extract_evidence_from_payload(
            {
                "_tool_name": "exec_shell",
                "return_code": 0,
                "stdout": "VERIFICATION_RESULT=FAIL\nALERT_PRESENT=0\n",
                "stderr": "",
            }
        )
        fail_state = agent._derive_state_from_evidence(fail_evidence, [], [], [], [])
        self.assertEqual(fail_state[0], "negatively_verified")
        self.assertIn("alert_not_triggered", {item["claim"] for item in fail_evidence})
        self.assertNotIn("alert_triggered", {item["claim"] for item in fail_evidence})
        self.assertEqual(fail_state[4], [])

    def test_no_alert_output_does_not_count_as_alert_triggered(self) -> None:
        agent = self.make_agent()
        evidence = agent._extract_evidence_from_payload(
            {
                "_tool_name": "exec_shell",
                "return_code": 0,
                "stdout": "NO_ALERT TimeoutException\n",
                "stderr": "",
            }
        )
        state = agent._derive_state_from_evidence(evidence, [], [], [], [])

        self.assertEqual(state[0], "negatively_verified")
        self.assertEqual(state[3], ["NO_ALERT TimeoutException"])
        self.assertEqual(state[4], [])
        self.assertIn("alert_not_triggered", {item["claim"] for item in evidence})
        self.assertNotIn("alert_triggered", {item["claim"] for item in evidence})

    def test_verifier_protocol_blocks_when_helper_itself_fails(self) -> None:
        agent = self.make_agent()
        evidence = agent._extract_evidence_from_payload(
            {
                "_tool_name": "exec_shell",
                "return_code": 0,
                "stdout": "VERIFICATION_RESULT=FAIL filter_rc=1 stderr= python: can't open file '/tests/filter.py': [Errno 2] No such file or directory\n",
                "stderr": "",
            }
        )
        state = agent._derive_state_from_evidence(evidence, [], [], [], [])

        self.assertEqual(state[0], "verification_blocked")
        self.assertTrue(state[2])
        self.assertIn("filter_rc=1", state[2][0])
        self.assertIn("verification_blocked", {item["claim"] for item in evidence})
        self.assertNotIn("verification_failed", {item["claim"] for item in evidence})

    def test_missing_commands_only_block_verification_when_verifier_related(self) -> None:
        agent = self.make_agent()

        jq_evidence = agent._extract_evidence_from_payload(
            {
                "_tool_name": "check_command_available",
                "return_code": 0,
                "stdout": '{"command":"jq","available":false}',
                "stderr": "",
            }
        )
        jq_state = agent._derive_state_from_evidence(jq_evidence, [], [], [], [])
        self.assertEqual(jq_state[0], "unverified")
        self.assertEqual(jq_state[2], [])

        pytest_evidence = agent._extract_evidence_from_payload(
            {
                "_tool_name": "check_command_available",
                "return_code": 0,
                "stdout": '{"command":"pytest","available":false}',
                "stderr": "",
            }
        )
        pytest_state = agent._derive_state_from_evidence(pytest_evidence, [], [], [], [])
        self.assertEqual(pytest_state[0], "verification_blocked")
        self.assertEqual(pytest_state[2], ["pytest_missing"])

    def test_browser_missing_blockers_are_suppressed_when_substitute_browser_exists(self) -> None:
        agent = self.make_agent()
        evidence = []
        evidence.extend(
            agent._extract_evidence_from_payload(
                {
                    "_tool_name": "check_command_available",
                    "return_code": 0,
                    "stdout": '{"command":"google-chrome","available":false}',
                    "stderr": "",
                }
            )
        )
        evidence.extend(
            agent._extract_evidence_from_payload(
                {
                    "_tool_name": "check_command_available",
                    "return_code": 0,
                    "stdout": '{"command":"chromium","available":true,"path":"/usr/bin/chromium"}',
                    "stderr": "",
                }
            )
        )

        state = agent._derive_state_from_evidence(evidence, [], [], [], [])

        self.assertEqual(state[0], "unverified")
        self.assertEqual(state[2], [])

    def test_verifier_blocker_rules_drive_state_and_actions(self) -> None:
        agent = self.make_agent()
        for blocker, rule in agent.VERIFIER_BLOCKER_RULES.items():
            evidence = [
                {
                    "type": "environment",
                    "claim": blocker,
                    "scope": "verifier",
                    "confidence": "high",
                    "source": "test",
                    "detail": "detail",
                }
            ]
            state = agent._derive_state_from_evidence(evidence, [], [], [], [])
            self.assertEqual(state[0], "verification_blocked")
            self.assertIn(blocker, state[2])

            actions = agent._derive_next_actions_from_state(
                verification_state=state[0],
                blocked_verifiers=state[2],
                verified_failures=[],
                rejected_solution_patterns=[],
                helper_roles={},
                existing=[],
            )
            self.assertIn(rule["next_action"], actions)

    def test_next_actions_use_verifier_helpers_and_rejected_patterns(self) -> None:
        agent = self.make_agent()
        actions = agent._derive_next_actions_from_state(
            verification_state="negatively_verified",
            blocked_verifiers=[],
            verified_failures=["ALERT_PRESENT=0"],
            rejected_solution_patterns=["on*_attributes", "script_tags", "banned_tags"],
            helper_roles={"/app/.agent-tools/run_check.py": "verifier"},
            existing=[],
        )

        joined = "\n".join(actions)
        self.assertIn("different solution family", joined)
        self.assertIn("/app/.agent-tools/run_check.py", joined)
        self.assertIn("Avoid all `on*` attributes", joined)
        self.assertIn("Avoid `<script>` tags", joined)
        self.assertIn("Avoid frame/iframe/object/embed", joined)

    def test_tool_category_evidence_tracks_inspection_edits_and_helpers(self) -> None:
        agent = self.make_agent()

        read_claims = {
            item["claim"]
            for item in agent._extract_evidence_from_payload(
                {
                    "_tool_name": "read_file",
                    "return_code": 0,
                    "stdout": "content",
                    "stderr": "",
                }
            )
        }
        self.assertIn("read_file_inspected", read_claims)

        write_claims = {
            item["claim"]
            for item in agent._extract_evidence_from_payload(
                {
                    "_tool_name": "write_file",
                    "return_code": 0,
                    "command": "write /app/out",
                    "stdout": "",
                    "stderr": "",
                }
            )
        }
        self.assertIn("write_file_applied", write_claims)

        helper_claims = {
            item["claim"]
            for item in agent._extract_evidence_from_payload(
                {
                    "_tool_name": "create_python_tool",
                    "return_code": 0,
                    "stdout": "created /app/.agent-tools/check.py",
                    "stderr": "",
                }
            )
        }
        self.assertIn("helper_created", helper_claims)

    def test_domain_evidence_extracts_filter_rejected_patterns(self) -> None:
        agent = self.make_agent()
        source = '''
for script in soup("script"):
    script.decompose()
for bad in ["frame", "iframe", "object", "embed"]:
    pass
if attr.startswith("on"):
    del tag.attrs[attr]
'''
        evidence = agent._extract_evidence_from_payload(
            {
                "_tool_name": "read_file",
                "return_code": 0,
                "stdout": source,
                "stderr": "",
            }
        )
        state = agent._derive_state_from_evidence(evidence, [], [], [], [])

        self.assertIn("filter_strips_on_attributes", {item["claim"] for item in evidence})
        self.assertIn("filter_strips_script_tags", {item["claim"] for item in evidence})
        self.assertIn("filter_strips_banned_tags", {item["claim"] for item in evidence})
        self.assertEqual(set(state[5]), {"on*_attributes", "script_tags", "banned_tags"})

    def test_rejected_pattern_rules_drive_state_guidance_and_actions(self) -> None:
        agent = self.make_agent()
        for pattern, rule in agent.REJECTED_PATTERN_RULES.items():
            evidence = [
                {
                    "type": "inspection",
                    "claim": rule["evidence_claim"],
                    "scope": "strategy",
                    "confidence": "high",
                    "source": "test",
                    "detail": "detail",
                }
            ]
            state = agent._derive_state_from_evidence(evidence, [], [], [], [])
            self.assertIn(pattern, state[5])

            guidance = agent._pattern_avoidance_guidance({"rejected_solution_patterns": [pattern]})
            self.assertIsNotNone(guidance)
            self.assertIn(rule["guidance"], guidance or "")

            actions = agent._derive_next_actions_from_state(
                verification_state="unverified",
                blocked_verifiers=[],
                verified_failures=[],
                rejected_solution_patterns=[pattern],
                helper_roles={},
                existing=[],
            )
            self.assertIn(rule["next_action"], actions)

    def test_replanner_prompt_surfaces_evidence_derived_actions(self) -> None:
        agent = self.make_agent()
        prompt = agent._replan_prompt(
            {
                "messages": [],
                "completed_steps": ["Tried a failed candidate"],
                "current_step": "Try another candidate",
                "plan_text": "1. Try another candidate",
                "next_actions": ["Switch to a different solution family before editing again."],
                "verification_state": "negatively_verified",
                "verification_summary": "ALERT_PRESENT=0",
                "rejected_solution_patterns": ["on*_attributes"],
            }
        )

        self.assertIn("Evidence-derived next actions", prompt)
        self.assertIn("Switch to a different solution family", prompt)
        self.assertIn("Do not propose a solution family", prompt)
        self.assertIn("on*_attributes", prompt)

    def test_rejected_pattern_edit_guard_blocks_repeat_payload_families(self) -> None:
        agent = self.make_agent()
        reason = agent._rejected_pattern_edit_reason(
            {
                "rejected_solution_patterns": ["on*_attributes", "script_tags", "banned_tags"],
            },
            "write_file",
            {
                "path": "/app/out.html",
                "content": '<script>alert(1)</script><img src=x onerror=alert(1)><object data="x"></object>',
            },
        )

        self.assertIsNotNone(reason)
        self.assertIn("on*_attributes", reason or "")
        self.assertIn("script_tags", reason or "")
        self.assertIn("banned_tags", reason or "")

    def test_rejected_pattern_edit_guard_ignores_clean_content(self) -> None:
        agent = self.make_agent()
        reason = agent._rejected_pattern_edit_reason(
            {
                "rejected_solution_patterns": ["on*_attributes", "script_tags", "banned_tags"],
            },
            "write_file",
            {
                "path": "/app/out.html",
                "content": "<!doctype html><html><body><meta charset='utf-8'></body></html>",
            },
        )

        self.assertIsNone(reason)

    def test_redundant_verifier_probe_guard_prefers_existing_helper(self) -> None:
        agent = self.make_agent()
        reason = agent._redundant_verifier_probe_reason(
            {
                "helper_roles": {"/app/.agent-tools/verify_alert.py": "verifier"},
                "blocked_verifiers": [],
                "evidence_log": [],
            },
            "check_command_available",
            {"command_name": "pytest"},
        )

        self.assertIsNotNone(reason)
        self.assertIn("verify_alert.py", reason or "")

    def test_redundant_browser_probe_guard_uses_existing_browser_capability(self) -> None:
        agent = self.make_agent()
        reason = agent._redundant_verifier_probe_reason(
            {
                "helper_roles": {},
                "blocked_verifiers": [],
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
            },
            "check_command_available",
            {"command_name": "google-chrome"},
        )

        self.assertIsNotNone(reason)
        self.assertIn("chromium_present", reason or "")

    def test_redundant_missing_verifier_probe_is_blocked(self) -> None:
        agent = self.make_agent()
        reason = agent._redundant_verifier_probe_reason(
            {
                "helper_roles": {},
                "blocked_verifiers": ["pytest_missing"],
                "evidence_log": [],
            },
            "check_command_available",
            {"command_name": "pytest"},
        )

        self.assertIsNotNone(reason)
        self.assertIn("already confirmed missing", reason or "")

    def test_chromium_dbus_noise_does_not_trigger_failure_guidance(self) -> None:
        from langchain_core.messages import ToolMessage

        agent = self.make_agent()
        message = ToolMessage(
            name="exec_shell",
            tool_call_id="call-1",
            content='{"command":"/usr/bin/chromium --headless --dump-dom file:///app/out.html","return_code":0,"stdout":"[1:2:ERROR:dbus/bus.cc:408] Failed to connect to the bus\\n<!doctype html>","stderr":""}',
        )

        self.assertIsNone(agent._tool_failure_guidance([message]))


if __name__ == "__main__":
    unittest.main()
