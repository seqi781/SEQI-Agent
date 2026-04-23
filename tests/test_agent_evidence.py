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


if __name__ == "__main__":
    unittest.main()
