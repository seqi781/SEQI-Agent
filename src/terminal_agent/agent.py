import json
import re
import textwrap
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.model_config import llm
from src.terminal_agent.constants import (
    DEFAULT_MAX_STEPS,
    DEFAULT_TEST_TIMEOUT_SEC,
    DEFAULT_TOOL_TIMEOUT_SEC,
    MAX_STREAM_SNIPPET_CHARS,
)
from src.terminal_agent.formatting import extract_text, format_exec_result, truncate_text
from src.terminal_agent.streaming import StreamEmitter
from src.terminal_agent.tools import build_agent_tools
from src.terminal_agent.types import AgentState, EvidenceItem


class LangGraphTerminalBenchAgent(BaseAgent):
    SUPPORTS_ATIF = False
    REJECTED_PATTERN_RULES: dict[str, dict[str, str]] = {
        "on*_attributes": {
            "evidence_claim": "filter_strips_on_attributes",
            "guidance": "Do not use any `on*` event-handler attributes; this family is already disproven.",
            "next_action": "Avoid all `on*` attributes in future candidates.",
        },
        "script_tags": {
            "evidence_claim": "filter_strips_script_tags",
            "guidance": "Do not use `<script>` tags; this family is already disproven.",
            "next_action": "Avoid `<script>` tags in future candidates.",
        },
        "banned_tags": {
            "evidence_claim": "filter_strips_banned_tags",
            "guidance": "Do not use frame/iframe/object/embed candidates; this family is already disproven.",
            "next_action": "Avoid frame/iframe/object/embed candidates.",
        },
    }
    VERIFIER_BLOCKER_RULES: dict[str, dict[str, str]] = {
        "pytest_missing": {
            "next_action": "Use an available verifier helper or create a small verifier helper instead of relying on pytest.",
        },
        "chromedriver_missing": {
            "next_action": "Create or choose a verifier path that does not require chromedriver, or bootstrap a browser verifier dependency.",
        },
        "chromium_missing": {
            "next_action": "Create or choose a verifier path that does not require Chromium, or bootstrap a browser verifier dependency.",
        },
        "google-chrome_missing": {
            "next_action": "Create or choose a verifier path that does not require Chrome, or bootstrap a browser verifier dependency.",
        },
        "chrome_missing": {
            "next_action": "Create or choose a verifier path that does not require Chrome, or bootstrap a browser verifier dependency.",
        },
        "firefox_missing": {
            "next_action": "Create or choose a verifier path that does not require Firefox, or bootstrap a browser verifier dependency.",
        },
        "geckodriver_missing": {
            "next_action": "Create or choose a verifier path that does not require geckodriver, or bootstrap a browser verifier dependency.",
        },
    }

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        tool_timeout_sec: int = DEFAULT_TOOL_TIMEOUT_SEC,
        test_timeout_sec: int = DEFAULT_TEST_TIMEOUT_SEC,
        working_dir: str = "/app",
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.max_steps = max_steps
        self.tool_timeout_sec = tool_timeout_sec
        self.test_timeout_sec = test_timeout_sec
        self.working_dir = working_dir
        self.helper_dir = f"{self.working_dir.rstrip('/')}/.agent-tools"
        self._stream_log_path = self.logs_dir / "stream.log"
        self._emitter = StreamEmitter(
            logger=self.logger,
            logs_dir=self.logs_dir,
            stream_log_path=self._stream_log_path,
        )
        self._capabilities: dict[str, bool] = {}

    @staticmethod
    def name() -> str:
        return "langgraph-terminal-agent"

    def version(self) -> str:
        return "0.9.17"

    def _emit(self, message: str) -> None:
        self._emitter.emit(message)

    def _emit_block(self, title: str, content: str) -> None:
        self._emitter.emit_block(title, content)

    def _langsmith_config(self, phase: str) -> dict[str, Any]:
        return {
            "run_name": f"{self.name()}:{phase}",
            "tags": [
                "langgraph-terminal-agent",
                f"phase:{phase}",
            ],
            "metadata": {
                "agent_name": self.name(),
                "agent_version": self.version(),
                "working_dir": self.working_dir,
                "helper_dir": self.helper_dir,
                "max_steps": self.max_steps,
                "model_name": getattr(llm, "model", None),
            },
        }

    async def setup(self, environment: BaseEnvironment) -> None:
        self._emit(f"setup.start cwd={self.working_dir}")
        capability_names = [
            "rg",
            "python3",
            "python",
            "jq",
            "perl",
            "xxd",
            "od",
            "file",
            "gcc",
            "cc",
            "make",
            "git",
            "tar",
            "curl",
            "wget",
        ]
        checks = [
            "pwd",
            "whoami || true",
            "git status --short || true",
            "command -v rg || true",
            "command -v python3 || command -v python || true",
            f"mkdir -p {self.helper_dir}",
        ]
        capability_probe = "printf '__CAPS__\\n'\n" + "\n".join(
            f"if command -v {name} >/dev/null 2>&1; then printf '{name}=1\\n'; else printf '{name}=0\\n'; fi"
            for name in capability_names
        )
        result = await environment.exec(
            command="set -e\n" + "\n".join(checks) + "\n" + capability_probe,
            cwd=self.working_dir,
            timeout_sec=60,
        )
        stdout = result.stdout or ""
        self._capabilities = {}
        if "__CAPS__\n" in stdout:
            prelude, capability_block = stdout.split("__CAPS__\n", 1)
            result.stdout = prelude
            stdout = prelude
            for line in capability_block.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                self._capabilities[key.strip()] = value.strip() == "1"
        formatted = format_exec_result(
            "setup-preflight",
            result.return_code,
            result.stdout,
            result.stderr,
        )
        (self.logs_dir / "setup.log").write_text(formatted)
        self._emit_block("setup.finish", formatted)
        if self._capabilities:
            capability_summary = json.dumps(self._capabilities, ensure_ascii=False, indent=2)
            (self.logs_dir / "capabilities.json").write_text(capability_summary)
            self._emit_block("setup.capabilities", capability_summary)

    def _build_model(self):
        return llm

    def _system_prompt(self) -> str:
        return textwrap.dedent(
            f"""
            You are an autonomous software engineering agent inside a benchmark task.

            Your job is not to explain the task. Your job is to make progress toward passing it.

            Expected goals:
            - Solve the task instance, not the benchmark harness.
            - Produce a real passing artifact, not a plausible story.
            - Verify success with positive evidence from this run.
            - Stay inside the task boundary unless the task explicitly asks you to repair the harness.

            Operating principles:
            - The benchmark conditions are fixed.
            - The concrete task instance may have a narrow passing path even when a general solution looks hard.
            - Local evidence is more important than prior assumptions.
            - A small working instance-specific solution is better than a broad theoretical design.
            - Work in a compact loop: plan, execute, re-plan, repeat.
            - Do not confuse “no failure was printed” with “the task passed.”
            - Do not modify verifier-side files, test harness files, or `/tests/*` just to make verification easier unless the task explicitly requires that repair.

            Core workflow:
            1. Create a short plan with a concrete current step.
            2. Execute that step with tools and small experiments.
            3. Re-plan based on what changed and what failed.
            4. Repeat until you pass or exhaust the step budget.

            Behavioral rules:
            - Do not give up early.
            - Do not negotiate the benchmark constraints.
            - Do not turn “a general solution is difficult” into “this instance cannot be solved.”
            - Do not claim infeasibility before substantial concrete inspection and at least one focused implementation attempt.
            - Do not submit placeholders, stubs, mocks, or knowingly unsolved implementations.
            - If a command or idea fails, narrow the problem and try another concrete path.
            - Prefer evidence from the current workspace over assumptions about standard formats or typical systems.
            - When the environment is limited, adapt; use simpler commands and local workarounds.
            - If the current tools are insufficient, extend the toolbox deliberately: search, fetch, download, or create a small helper script only when that is the shortest path to progress.
            - Do not stop at “tool missing.” Try substitution, direct shell composition, web lookup, download, or helper creation first.
            - Do not ask the user to provide more files, metadata, companion artifacts, or relaxed constraints. Work with the current instance and available expansion tools.
            - Keep moving. Avoid long theoretical justifications when another inspection step or experiment is available.

            Tool strategy:
            - Use tools to inspect facts, edit deterministically, run experiments, and verify outcomes.
            - Prefer targeted inspection before broad command use.
            - Prefer small experiments before large rewrites.
            - When output matters, run the artifact and compare actual output.
            - When logs are noisy, extract the key failure signals first.
            - Use web search as a fallback when local evidence is insufficient.
            - Prefer downloading or creating a tiny task-specific helper over designing a broad framework.
            - Treat helper scripts and downloaded utilities as temporary tools you can invoke with shell commands afterward.
            - When you need a temporary script, prefer the dedicated helper-creation tools for Python or shell before resorting to large edits.
            - When creating a verification helper, make it print machine-readable evidence when possible: `VERIFICATION_RESULT=PASS`, `VERIFICATION_RESULT=FAIL`, or `VERIFICATION_RESULT=BLOCKED`. Domain-specific booleans such as `ALERT_PRESENT=1` / `ALERT_PRESENT=0` are also useful.
            - For opaque or binary artifacts, prefer direct probing with byte inspection, string scanning, file-size checks, sidecar discovery, and small parsers before concluding anything about the format.
            - Call one registered tool at a time with that tool's exact schema. Do not invent wrapper payloads such as parallel-tool objects unless a tool explicitly asks for them.

            Finish policy:
            - Before finishing, verify the relevant constraints explicitly.
            - If you still fail, the final answer must be brief, factual, and grounded in direct evidence from this instance.
            - Default working directory: `{self.working_dir}`
            - Temporary helper directory: `{self.helper_dir}`
            - Maximum steps: {self.max_steps}

            When you are done, respond normally in English and do not call tools.
            """
        ).strip()

    def _plan_prompt(self, instruction: str, completed_steps: list[str]) -> str:
        completed = "\n".join(f"- {step}" for step in completed_steps[-12:]) or "- none"
        artifact_hint = ""
        lowered = instruction.lower()
        if any(marker in lowered for marker in ["/", ".ckpt", ".bin", ".json", ".yaml", ".toml", ".bpe", "file", "artifact", "checkpoint"]):
            artifact_hint = (
                "\nAdditional requirement:\n"
                "- Because this task depends on concrete local artifacts, the first step should gather evidence with tools rather than making assumptions.\n"
            )
        return textwrap.dedent(
            f"""
            Create a compact execution plan for this task.

            Requirements:
            - Use 3 to 6 short steps.
            - Prefer concrete, benchmark-oriented actions.
            - The current step must be a directly executable next move.
            - Do not ask the user for more files or permissions.
            {artifact_hint}

            Task:
            {instruction}

            Already completed:
            {completed}

            Respond in exactly this format:
            PLAN:
            1. ...
            2. ...
            3. ...
            CURRENT_STEP: ...
            DONE: no
            """
        ).strip()

    def _tool_usage_guide(self) -> str:
        return textwrap.dedent(
            f"""
            Tool usage guide:
            - For repository/file discovery: use `list_files`, `find_files`, `file_info`, `read_file`, `read_many_files`.
            - For text/symbol search: use `search_text`.
            - For binary or opaque artifacts: use `file_info`, `inspect_file_bytes`, `scan_strings`, and sidecar discovery with `find_files`.
            - For environment checks: use `check_command_available`, `inspect_env`, `list_processes`, `list_ports`, `inspect_services`.
            - For direct experiments: use `exec_shell`, `run_program_with_input`, `run_tests`.
            - For output validation: use `compare_output`.
            - For edits: prefer `replace_in_file`, `apply_unified_diff`, `write_file`, or a helper in `{self.helper_dir}`.
            - For missing commands: first `check_command_available`, then `create_command_shim`, `create_shell_tool`, `create_python_tool`, `download_url`, or `install_helper_tool`.
            - For web fallback: use `brave_web_search`, `fetch_url`, or `download_url` only when local evidence is insufficient.
            - Do not patch verifier-side files or `/tests/*` unless the task explicitly requires repairing the harness.
            - If an inspected test file looks like a pytest module, do not run `python test_file.py`. Prefer `run_tests` with `pytest -q <file>` or another real test runner.
            - If you create a verifier helper, have it print `VERIFICATION_RESULT=PASS|FAIL|BLOCKED` and any relevant task-specific facts such as `ALERT_PRESENT=1|0`.

            Evidence rule:
            - Do not claim that a file exists, a format is known, a tool is missing, or a constraint is satisfied unless a tool result in this run supports it.
            - If you are unsure, call the relevant tool instead of guessing.
            - Treat a verifier as passed only when a real verifier command ran and produced positive success evidence or a trusted zero-exit test runner result.
            """
        ).strip()

    def _evidence_summary(self, state: AgentState, limit: int = 12) -> str:
        payloads = self._tool_payloads(state["messages"])
        lines: list[str] = []
        evidence_log = state.get("evidence_log", [])
        failure_signals = state.get("failure_signals", [])
        failure_summary = state.get("failure_summary", "").strip()
        next_actions = state.get("next_actions", [])
        verification_state = state.get("verification_state", "").strip()
        verification_summary = state.get("verification_summary", "").strip()
        rejected_patterns = state.get("rejected_solution_patterns", [])
        helper_roles = state.get("helper_roles", {})
        if verification_state:
            lines.append(f"Verification state: {verification_state}")
        if verification_summary:
            lines.append(f"Verification summary: {truncate_text(verification_summary, 220)}")
        if rejected_patterns:
            lines.append("Rejected patterns:")
            lines.extend(f"- {line}" for line in rejected_patterns[-6:])
        if helper_roles:
            lines.append("Helper roles:")
            lines.extend(f"- {path}: {role}" for path, role in list(helper_roles.items())[-6:])
        if evidence_log:
            lines.append("Structured evidence:")
            for item in evidence_log[-6:]:
                lines.append(
                    f"- [{item.get('type')}/{item.get('scope')}/{item.get('confidence')}] "
                    f"{item.get('claim')}: {truncate_text(item.get('detail', ''), 160)}"
                )
        if failure_signals:
            lines.append("Failure signals:")
            lines.extend(f"- {line}" for line in failure_signals[-6:])
        if failure_summary:
            lines.append("Failure summary:")
            lines.append(truncate_text(failure_summary, 400))
        if next_actions:
            lines.append("Suggested next actions:")
            lines.extend(f"- {line}" for line in next_actions[-6:])
        if not payloads and not lines:
            return "No tool evidence collected yet."
        for payload in payloads[-limit:]:
            tool_name = str(payload.get("_tool_name", "tool"))
            return_code = int(payload.get("return_code", 0) or 0)
            stdout = truncate_text(str(payload.get("stdout", "") or "").strip(), 180).replace("\n", " ")
            stderr = truncate_text(str(payload.get("stderr", "") or "").strip(), 180).replace("\n", " ")
            detail = stdout or stderr or "no output"
            lines.append(f"- {tool_name} rc={return_code}: {detail}")
        return "\n".join(lines)

    def _preferred_tools_for_step(self, step_text: str) -> str:
        lowered = step_text.lower()
        suggestions: list[str] = []
        if any(marker in lowered for marker in ["inspect", "find", "list", "discover", "locate"]):
            suggestions.extend(["list_files", "find_files", "file_info", "read_file"])
        if any(marker in lowered for marker in ["binary", "format", "header", "checkpoint", "artifact", ".ckpt", ".bin", ".bpe"]):
            suggestions.extend(["file_info", "inspect_file_bytes", "scan_strings", "find_files"])
        if any(marker in lowered for marker in ["tool", "command", "missing", "environment", "dependency"]):
            suggestions.extend(["check_command_available", "inspect_env", "create_command_shim", "create_shell_tool"])
        if any(marker in lowered for marker in ["verify", "test", "compile", "run", "output"]):
            suggestions.extend(["run_tests", "exec_shell", "run_program_with_input", "compare_output"])
        if any(marker in lowered for marker in ["download", "search", "docs", "web"]):
            suggestions.extend(["brave_web_search", "fetch_url", "download_url"])
        deduped = list(dict.fromkeys(suggestions))
        if not deduped:
            deduped = ["list_files", "file_info", "read_file", "exec_shell"]
        return ", ".join(f"`{name}`" for name in deduped)

    def _pattern_avoidance_guidance(self, state: AgentState) -> str | None:
        rejected = state.get("rejected_solution_patterns", [])
        if not rejected:
            return None
        guidance = [
            self.REJECTED_PATTERN_RULES[pattern]["guidance"]
            for pattern in rejected
            if pattern in self.REJECTED_PATTERN_RULES
        ]
        if not guidance:
            return None
        return "Rejected solution families from this run:\n- " + "\n- ".join(guidance)

    def _next_action_guidance(self, state: AgentState) -> str | None:
        next_actions = [action for action in state.get("next_actions", []) if action.strip()]
        if not next_actions:
            return None
        return (
            "Evidence-derived next actions. Treat these as current constraints, not optional notes:\n- "
            + "\n- ".join(next_actions[-6:])
        )

    def _replan_prompt(self, state: AgentState) -> str:
        completed = "\n".join(f"- {step}" for step in state.get("completed_steps", [])[-12:]) or "- none"
        current_step = state.get("current_step", "").strip() or "none"
        plan_text = state.get("plan_text", "").strip() or "none"
        rejected_patterns = state.get("rejected_solution_patterns", [])
        rejected_text = "\n".join(f"- {item}" for item in rejected_patterns[-12:]) or "- none"
        next_actions = state.get("next_actions", [])
        next_actions_text = "\n".join(f"- {item}" for item in next_actions[-12:]) or "- none"
        verification_state = state.get("verification_state", "unverified")
        verification_summary = state.get("verification_summary", "").strip() or "none"
        return textwrap.dedent(
            f"""
            Re-plan after the latest execution round.

            Rules:
            - If the task is complete, set DONE: yes and provide the final answer in FINAL_RESPONSE.
            - Otherwise, provide an updated compact plan and one concrete CURRENT_STEP.
            - Keep the plan short and action-oriented.
            - Do not ask the user for more files, metadata, or permissions.
            - Do not repeat a verifier command that was already rejected as invalid. Choose a different concrete next step.
            - Do not propose a solution family that is already recorded as rejected in this run.
            - If evidence-derived next actions are listed, the new CURRENT_STEP should directly satisfy one of them unless there is a stronger new fact.

            Previous plan:
            {plan_text}

            Previous current step:
            {current_step}

            Completed steps:
            {completed}

            Verification state:
            {verification_state}

            Verification summary:
            {verification_summary}

            Rejected solution patterns:
            {rejected_text}

            Evidence-derived next actions:
            {next_actions_text}

            Respond in one of these formats:

            If not done:
            PLAN:
            1. ...
            2. ...
            CURRENT_STEP: ...
            DONE: no

            If done:
            PLAN:
            1. ...
            CURRENT_STEP: none
            DONE: yes
            FINAL_RESPONSE:
            ...
            """
        ).strip()

    def _parse_plan_response(self, text: str) -> tuple[str, str, bool, str | None]:
        plan_match = re.search(r"PLAN:\s*(.*?)(?:CURRENT_STEP:|$)", text, re.DOTALL | re.IGNORECASE)
        current_match = re.search(r"CURRENT_STEP:\s*(.*?)(?:DONE:|FINAL_RESPONSE:|$)", text, re.DOTALL | re.IGNORECASE)
        done_match = re.search(r"DONE:\s*(yes|no)", text, re.IGNORECASE)
        final_match = re.search(r"FINAL_RESPONSE:\s*(.*)$", text, re.DOTALL | re.IGNORECASE)

        raw_plan = (plan_match.group(1).strip() if plan_match else text.strip()) or "1. Inspect the task and continue."
        plan_lines = [line.strip(" -") for line in raw_plan.splitlines() if line.strip()]
        if not plan_lines:
            plan_lines = ["Inspect the task and continue."]
        normalized_lines = [re.sub(r"^\d+\.\s*", "", line).strip() for line in plan_lines[:6]]
        plan_text = "\n".join(f"{idx}. {line}" for idx, line in enumerate(normalized_lines, start=1))
        current_step = (current_match.group(1).strip() if current_match else "") or "Inspect the task and continue."
        numbered_step_match = re.fullmatch(r"(\d+)", current_step)
        if numbered_step_match:
            index = int(numbered_step_match.group(1)) - 1
            if 0 <= index < len(normalized_lines):
                current_step = normalized_lines[index]
        else:
            numbered_prefix_match = re.fullmatch(r"(\d+)\.\s*(.*)", current_step)
            if numbered_prefix_match:
                suffix = numbered_prefix_match.group(2).strip()
                if suffix:
                    current_step = suffix
        done = (done_match.group(1).strip().lower() == "yes") if done_match else False
        final_response = final_match.group(1).strip() if final_match else None
        return plan_text, current_step, done, final_response

    def _recent_system_messages(self, messages: list[BaseMessage], limit: int = 8) -> list[str]:
        texts: list[str] = []
        for message in messages[-limit:]:
            if isinstance(message, SystemMessage):
                texts.append(extract_text(message.content))
        return texts

    def _should_emit_guidance(self, state: AgentState, guidance: str) -> bool:
        if not guidance.strip():
            return False
        recent = self._recent_system_messages(state["messages"])
        return guidance not in recent

    def _is_step_repeated(self, state: AgentState, candidate_step: str) -> bool:
        normalized = candidate_step.strip().lower()
        if not normalized:
            return False
        completed = [step.strip().lower() for step in state.get("completed_steps", [])[-4:]]
        return normalized in completed

    def _executor_guidance(self, state: AgentState) -> list[BaseMessage]:
        guidance: list[BaseMessage] = []
        missing = self._missing_tool_candidates(state["messages"])
        if missing:
            missing_summary = ", ".join(missing)
            guidance.append(
                SystemMessage(
                    content=(
                        f"Missing commands detected recently: {missing_summary}. "
                        f"Bootstrap the missing capability inside `{self.helper_dir}` before giving up. "
                        "Everything in that directory is already on PATH."
                    )
                )
            )
        recon_gap = self._reconnaissance_gap(state)
        if recon_gap:
            guidance.append(SystemMessage(content=recon_gap))
        verification_gap = self._verification_gap(state)
        if verification_gap:
            guidance.append(SystemMessage(content=verification_gap))
        verifier_hint = self._recommended_verifier_hint(state)
        if verifier_hint:
            guidance.append(SystemMessage(content=verifier_hint))
        pattern_guidance = self._pattern_avoidance_guidance(state)
        if pattern_guidance:
            guidance.append(SystemMessage(content=pattern_guidance))
        next_action_guidance = self._next_action_guidance(state)
        if next_action_guidance:
            guidance.append(SystemMessage(content=next_action_guidance))
        helper_roles = state.get("helper_roles", {})
        verifier_helpers = [path for path, role in helper_roles.items() if role == "verifier"]
        if verifier_helpers:
            guidance.append(
                SystemMessage(
                    content=(
                        "A verifier helper already exists in this run. Prefer reusing it for verification before falling back to weaker checks.\n"
                        + "\n".join(f"- {path}" for path in verifier_helpers[-4:])
                    )
                )
            )
        payloads = self._tool_payloads(state["messages"])
        invalid_verifier_rejections = [
            payload
            for payload in payloads[-6:]
            if str(payload.get("_tool_name", "")).lower() in {"exec_shell", "run_tests"}
            and "invalid verifier command" in f"{payload.get('stdout', '')}\n{payload.get('stderr', '')}".lower()
        ]
        if invalid_verifier_rejections:
            followup = verifier_hint or (
                "Use `run_tests` with a real test runner or choose another verification path that yields concrete pass/fail evidence."
            )
            guidance.append(
                SystemMessage(
                    content=(
                        "A recent verifier command was rejected as invalid. "
                        "Do not retry the same `python test_file.py` pattern. "
                        f"{followup} "
                        "Choose a different verification action now."
                    )
                )
            )
        verification_state = state.get("verification_state", "")
        if verification_state == "negatively_verified":
            guidance.append(
                SystemMessage(
                    content=(
                        "A real verification attempt already failed for the current approach. "
                        "Do not keep polishing the same payload family. Switch to a different solution family and verify again."
                    )
                )
            )
        return guidance

    def _has_invalid_completion(self, messages: list[BaseMessage]) -> bool:
        invalid_markers = [
            "placeholder",
            "stub",
            "dummy",
            "mock implementation",
            "not implemented",
            "not supported",
            "does not actually",
            "not actually",
            "cannot realistically",
            "can't realistically",
            "if you can provide",
            "please provide",
            "if you provide",
            "if you can upload",
            "upload the",
            "provide any checkpoint companion files",
            "provide companion files",
            "provide more files",
            "provide the metadata",
            "provide metadata",
            "provide a manifest",
            "pre-extracted",
            "companion files",
            "task is unsolved",
            "not solved",
        ]
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            text = extract_text(message.content).lower()
            if any(marker in text for marker in invalid_markers):
                return True
        return False

    def _has_constraint_negotiation(self, messages: list[BaseMessage]) -> bool:
        negotiation_markers = [
            "relax",
            "increase the size limit",
            "choose one",
            "which option",
            "allow a conversion step",
            "provide an explicit mapping",
            "to make this solvable",
            "reply with which option",
            "provide any checkpoint companion files",
            "provide more files",
            "provide metadata",
            "provide a manifest",
            "if you provide",
        ]
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            text = extract_text(message.content).lower()
            if any(marker in text for marker in negotiation_markers):
                return True
        return False

    def _has_premature_impossibility_claim(self, messages: list[BaseMessage]) -> bool:
        impossibility_markers = [
            "cannot satisfy",
            "not achievable",
            "not feasible",
            "can't be done",
            "cannot realistically",
            "cannot fit",
            "is impossible",
            "not possible",
        ]
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            text = extract_text(message.content).lower()
            if any(marker in text for marker in impossibility_markers):
                return True
        return False

    def _has_runtime_validation(self, commands: str) -> bool:
        runtime_patterns = [
            "./",
            "/app/a.out",
            " a.out",
            "python ",
            "python3 ",
            "node ",
            "cargo run",
            "go run",
            "java ",
        ]
        compile_prefixes = ("gcc ", "clang ", "cc ", "make ", "cmake ", "go build", "cargo build")
        for raw_line in commands.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(compile_prefixes):
                continue
            if any(pattern in line for pattern in runtime_patterns):
                return True
        return False

    def _finalizable_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if not messages:
            return messages
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return messages[:-1]
        return messages

    def _tool_payloads(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            try:
                payload = json.loads(extract_text(message.content))
            except Exception:
                payloads.append(
                    {
                        "_tool_name": message.name,
                        "command": "",
                        "stdout": extract_text(message.content),
                        "stderr": "",
                    }
                )
                continue
            if isinstance(payload, dict):
                payload["_tool_name"] = message.name
                payloads.append(payload)
        return payloads

    def _merge_unique_strings(self, existing: list[str], new_items: list[str]) -> list[str]:
        merged = list(existing)
        for item in new_items:
            normalized = item.strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        return merged

    def _merge_evidence(self, existing: list[EvidenceItem], new_items: list[EvidenceItem]) -> list[EvidenceItem]:
        merged = list(existing)
        seen = {
            (
                item.get("type", ""),
                item.get("claim", ""),
                item.get("source", ""),
                item.get("detail", ""),
            )
            for item in merged
        }
        for item in new_items:
            key = (
                item.get("type", ""),
                item.get("claim", ""),
                item.get("source", ""),
                item.get("detail", ""),
            )
            if key in seen:
                continue
            merged.append(item)
            seen.add(key)
        return merged

    def _make_evidence(
        self,
        *,
        source: str,
        evidence_type: EvidenceItem["type"],
        claim: str,
        scope: EvidenceItem["scope"],
        confidence: EvidenceItem["confidence"],
        detail: str,
    ) -> EvidenceItem:
        return {
            "type": evidence_type,
            "claim": claim,
            "scope": scope,
            "confidence": confidence,
            "source": source,
            "detail": truncate_text(detail.strip(), 240),
        }

    def _extract_environment_evidence(
        self,
        *,
        tool_name: str,
        source: str,
        command: str,
        stdout: str,
        combined: str,
    ) -> list[EvidenceItem]:
        if tool_name != "check_command_available":
            return []
        match = re.search(r'"command"\s*:\s*"([^"]+)"', stdout)
        command_name = match.group(1) if match else ""
        if '"available":true' in combined:
            claim = f"{command_name or source}_present"
        elif '"available":false' in combined:
            claim = f"{command_name or source}_missing"
        else:
            return []
        return [
            self._make_evidence(
                source=source,
                evidence_type="environment",
                claim=claim,
                scope="environment",
                confidence="high",
                detail=stdout or command,
            )
        ]

    def _extract_tool_category_evidence(
        self,
        *,
        tool_name: str,
        source: str,
        command: str,
        stdout: str,
        return_code: int,
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        if tool_name in {"read_file", "read_many_files", "list_files", "find_files", "file_info", "search_text"} and stdout:
            items.append(
                self._make_evidence(
                    source=source,
                    evidence_type="inspection",
                    claim=f"{tool_name}_inspected",
                    scope="artifact",
                    confidence="high",
                    detail=stdout,
                )
            )
        if tool_name in {"write_file", "write_json", "append_file", "replace_in_file", "apply_unified_diff"} and return_code == 0:
            items.append(
                self._make_evidence(
                    source=source,
                    evidence_type="artifact_change",
                    claim=f"{tool_name}_applied",
                    scope="artifact",
                    confidence="high",
                    detail=command or stdout,
                )
            )
        if tool_name in {"create_python_tool", "create_shell_tool", "create_helper_tool", "create_command_shim"} and return_code == 0:
            items.append(
                self._make_evidence(
                    source=source,
                    evidence_type="artifact_change",
                    claim="helper_created",
                    scope="environment",
                    confidence="high",
                    detail=stdout or command,
                )
            )
        return items

    def _extract_verification_evidence(
        self,
        *,
        tool_name: str,
        source: str,
        command: str,
        stdout: str,
        stderr: str,
        combined: str,
        return_code: int,
    ) -> list[EvidenceItem]:
        if tool_name not in {"exec_shell", "run_tests", "run_program_with_input", "compare_output"}:
            return []
        detail = stdout or stderr
        items: list[EvidenceItem] = []

        def add(
            evidence_type: EvidenceItem["type"],
            claim: str,
            scope: EvidenceItem["scope"],
            confidence: EvidenceItem["confidence"] = "high",
            item_detail: str | None = None,
        ) -> None:
            items.append(
                self._make_evidence(
                    source=source,
                    evidence_type=evidence_type,
                    claim=claim,
                    scope=scope,
                    confidence=confidence,
                    detail=item_detail if item_detail is not None else detail,
                )
            )

        marker_rules: list[tuple[str, EvidenceItem["type"], str, EvidenceItem["scope"]]] = [
            ("verification_result=pass", "verification", "verification_passed", "solution"),
            ("verification_result=fail", "verification", "verification_failed", "solution"),
            ("verification_result=blocked", "failure", "verification_blocked", "verifier"),
            ("alert_present=1", "verification", "alert_triggered", "solution"),
            ("alert_present=true", "verification", "alert_triggered", "solution"),
            ("alert_present=0", "verification", "alert_not_triggered", "solution"),
            ("alert_present=false", "verification", "alert_not_triggered", "solution"),
            ("no_alert", "verification", "alert_not_triggered", "solution"),
            ("no module named pytest", "environment", "pytest_missing", "verifier"),
            ("refusing to run an invalid verifier command", "failure", "invalid_verifier_command", "verifier"),
        ]
        for marker, evidence_type, claim, scope in marker_rules:
            if marker in combined:
                add(evidence_type, claim, scope)
        if "alert " in combined or "alert_successfully_triggered" in combined or "alert successfully triggered" in combined:
            add("verification", "alert_triggered", "solution")
        if return_code != 0:
            add("failure", f"{tool_name}_failed", "strategy", "medium", stderr or stdout or command)
        return items

    def _extract_domain_evidence(
        self,
        *,
        tool_name: str,
        source: str,
        stdout: str,
    ) -> list[EvidenceItem]:
        if tool_name not in {"read_file", "exec_shell"}:
            return []
        items: list[EvidenceItem] = []

        def add(claim: str, confidence: EvidenceItem["confidence"] = "high") -> None:
            items.append(
                self._make_evidence(
                    source=source,
                    evidence_type="inspection",
                    claim=claim,
                    scope="strategy",
                    confidence=confidence,
                    detail=stdout,
                )
            )

        lowered_stdout = stdout.lower()
        if "if attr.startswith(\"on\")" in stdout or "if attr.startswith(\\\"on\\\")" in stdout:
            add("filter_strips_on_attributes")
        if "script.decompose" in lowered_stdout:
            add("filter_strips_script_tags")
        if "frame" in lowered_stdout and "iframe" in lowered_stdout and "object" in lowered_stdout:
            add("filter_strips_banned_tags", "medium")
        return items

    def _extract_evidence_from_payload(self, payload: dict[str, Any]) -> list[EvidenceItem]:
        tool_name = str(payload.get("_tool_name", "")).lower()
        command = str(payload.get("command", "") or "").strip()
        stdout = str(payload.get("stdout", "") or "").strip()
        stderr = str(payload.get("stderr", "") or "").strip()
        combined = f"{stdout}\n{stderr}".lower()
        return_code = int(payload.get("return_code", 0) or 0)
        source = tool_name or "tool"
        return [
            *self._extract_environment_evidence(
                tool_name=tool_name,
                source=source,
                command=command,
                stdout=stdout,
                combined=combined,
            ),
            *self._extract_tool_category_evidence(
                tool_name=tool_name,
                source=source,
                command=command,
                stdout=stdout,
                return_code=return_code,
            ),
            *self._extract_verification_evidence(
                tool_name=tool_name,
                source=source,
                command=command,
                stdout=stdout,
                stderr=stderr,
                combined=combined,
                return_code=return_code,
            ),
            *self._extract_domain_evidence(
                tool_name=tool_name,
                source=source,
                stdout=stdout,
            ),
        ]

    def _derive_state_from_evidence(
        self,
        evidence_log: list[EvidenceItem],
        existing_blocked: list[str],
        existing_failures: list[str],
        existing_successes: list[str],
        existing_rejected: list[str],
    ) -> tuple[str, str, list[str], list[str], list[str], list[str]]:
        blocked = list(existing_blocked)
        failures = list(existing_failures)
        successes = list(existing_successes)
        rejected = list(existing_rejected)
        verification_state = "unverified"
        verification_summary = "No verification evidence yet."

        for item in evidence_log:
            claim = item.get("claim", "")
            detail = item.get("detail", "")
            if claim in self.VERIFIER_BLOCKER_RULES:
                blocked = self._merge_unique_strings(blocked, [claim])
            if claim == "alert_not_triggered":
                failures = self._merge_unique_strings(failures, [detail or claim])
                verification_state = "negatively_verified"
                verification_summary = detail or "Verification helper reported that no alert was triggered."
            if claim == "alert_triggered":
                successes = self._merge_unique_strings(successes, [detail or claim])
                verification_state = "positively_verified"
                verification_summary = detail or "Verification reported a successful alert trigger."
            if claim == "verification_failed":
                failures = self._merge_unique_strings(failures, [detail or claim])
                verification_state = "negatively_verified"
                verification_summary = detail or "Verifier reported failure."
            if claim == "verification_passed":
                successes = self._merge_unique_strings(successes, [detail or claim])
                verification_state = "positively_verified"
                verification_summary = detail or "Verifier reported success."
            if claim == "verification_blocked":
                blocked = self._merge_unique_strings(blocked, [detail or claim])
                if verification_state == "unverified":
                    verification_state = "verification_blocked"
                    verification_summary = detail or "Verifier reported a blocked state."
            if claim == "invalid_verifier_command":
                blocked = self._merge_unique_strings(blocked, ["invalid_verifier_command"])
                if verification_state == "unverified":
                    verification_state = "verification_blocked"
                    verification_summary = detail or "A verifier command was rejected as invalid."
            rejected_claims = [
                pattern
                for pattern, rule in self.REJECTED_PATTERN_RULES.items()
                if claim == rule["evidence_claim"]
            ]
            rejected = self._merge_unique_strings(rejected, rejected_claims)

        if verification_state == "unverified" and blocked:
            verification_state = "verification_blocked"
            verification_summary = "Verification is currently blocked by missing or invalid verifier tooling."
        elif verification_state == "unverified" and any(
            item.get("type") in {"verification", "failure"} for item in evidence_log
        ):
            verification_state = "weakly_checked"
            verification_summary = "Some experiments ran, but there is no positive or negative verifier-grade outcome yet."

        return verification_state, verification_summary, blocked, failures, successes, rejected

    def _derive_next_actions_from_state(
        self,
        *,
        verification_state: str,
        blocked_verifiers: list[str],
        verified_failures: list[str],
        rejected_solution_patterns: list[str],
        helper_roles: dict[str, str],
        existing: list[str],
    ) -> list[str]:
        actions = list(existing)

        def add(action: str) -> None:
            if action and action not in actions:
                actions.append(action)

        verifier_helpers = [path for path, role in helper_roles.items() if role == "verifier"]
        if verification_state == "negatively_verified":
            add("Switch to a different solution family before editing again; do not only tweak the last failed approach.")
            if verifier_helpers:
                add(f"After each new candidate, verify with `{verifier_helpers[0]}`.")
        if verification_state == "verification_blocked":
            blocker_actions = [
                self.VERIFIER_BLOCKER_RULES[blocker]["next_action"]
                for blocker in blocked_verifiers
                if blocker in self.VERIFIER_BLOCKER_RULES
            ]
            if blocker_actions:
                for action in blocker_actions:
                    add(action)
            else:
                add("Resolve verifier blockage or create a local verifier helper that emits VERIFICATION_RESULT markers.")
        for pattern in rejected_solution_patterns:
            rule = self.REJECTED_PATTERN_RULES.get(pattern)
            if rule:
                add(rule["next_action"])
        if verifier_helpers and verification_state in {"unverified", "weakly_checked"}:
            add(f"Use existing verifier helper `{verifier_helpers[0]}` for the next verification step.")
        return actions

    def _execution_like_payloads(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        execution_like_tools = {
            "exec_shell",
            "run_tests",
            "run_program_with_input",
            "compare_output",
            "wait_for_port",
            "inspect_services",
            "list_ports",
            "list_processes",
            "check_command_available",
        }
        return [
            payload
            for payload in payloads
            if str(payload.get("_tool_name", "")).lower() in execution_like_tools
        ]

    def _is_protected_benchmark_path(self, candidate: str) -> bool:
        path = candidate.strip()
        if not path:
            return False
        normalized = path.replace("\\", "/")
        return normalized == "/tests" or normalized.startswith("/tests/")

    def _edited_paths_for_tool(self, tool_name: str, args: dict[str, Any]) -> list[str]:
        edit_arg_keys = {
            "write_file": ["path"],
            "write_json": ["path"],
            "append_file": ["path"],
            "replace_in_file": ["path"],
            "make_directory": ["path"],
            "copy_file": ["source", "destination"],
            "move_file": ["source", "destination"],
            "delete_file": ["path"],
            "apply_unified_diff": ["path"],
        }
        keys = edit_arg_keys.get(tool_name, [])
        paths: list[str] = []
        for key in keys:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                paths.append(value.strip())
        return paths

    def _reconnaissance_score(self, payloads: list[dict[str, Any]]) -> int:
        score = 0
        for payload in payloads:
            tool_name = str(payload.get("_tool_name", "")).lower()
            command = str(payload.get("command", "")).lower()
            if tool_name in {"list_files", "find_files", "file_info", "read_file", "read_many_files", "read_json", "inspect_env"}:
                score += 1
                continue
            if any(
                marker in command
                for marker in [
                    "head ",
                    "od ",
                    "strings ",
                    "ls ",
                    "stat ",
                    "wc -c",
                    "tar ",
                    "cat ",
                ]
            ):
                score += 1
        return score

    def _reconnaissance_gap(self, state: AgentState) -> str | None:
        messages = state["messages"]
        if not messages:
            return None

        first_message = messages[0]
        if isinstance(first_message, dict):
            instruction = extract_text(first_message.get("content", "")).lower()
        else:
            instruction = extract_text(first_message.content).lower()

        payloads = self._tool_payloads(messages)
        score = self._reconnaissance_score(payloads)
        command_text = "\n".join(str(payload.get("command", "")) for payload in payloads).lower()

        artifact_markers = [
            "/",
            ".ckpt",
            ".bin",
            ".json",
            ".yaml",
            ".toml",
            ".csv",
            ".bpe",
            ".model",
            ".meta",
            ".index",
            ".txt",
            "file",
            "checkpoint",
            "artifact",
        ]
        refers_to_local_artifacts = any(marker in instruction for marker in artifact_markers)
        has_concrete_probe = any(
            marker in command_text
            for marker in ["head ", "od ", "strings ", "ls ", "wc -c", "stat ", "tar ", "cat "]
        )

        if refers_to_local_artifacts and score < 4 and not has_concrete_probe:
            guidance = (
                "Before choosing a solution strategy, inspect the concrete task-local artifacts more directly. "
                "Check neighboring files, sizes, file headers, sample contents, and any sidecar metadata first. "
                "Avoid assuming the file format or structure before inspecting the actual instance."
            )
            if self._should_emit_guidance(state, guidance):
                return guidance
        return None

    def _tool_failure_guidance(self, tool_messages: list[ToolMessage]) -> str | None:
        payloads = self._tool_payloads(tool_messages)
        if not payloads:
            return None

        execution_like_tools = {
            "exec_shell",
            "run_tests",
            "run_program_with_input",
            "compare_output",
            "wait_for_port",
            "inspect_services",
            "list_ports",
            "list_processes",
            "check_command_available",
            "brave_web_search",
            "fetch_url",
            "download_url",
            "install_helper_tool",
            "create_helper_tool",
            "create_python_tool",
            "create_shell_tool",
            "create_command_shim",
            "apply_unified_diff",
            "replace_in_file",
            "write_file",
            "write_json",
            "append_file",
            "copy_file",
            "move_file",
            "delete_file",
        }
        failure_signals: list[str] = []
        missing_tool_contexts: list[str] = []
        for payload in payloads:
            tool_name = str(payload.get("_tool_name", ""))
            return_code = int(payload.get("return_code", 0) or 0)
            stdout = str(payload.get("stdout", "") or "")
            stderr = str(payload.get("stderr", "") or "")
            command = str(payload.get("command", "") or "")
            combined = f"{stdout}\n{stderr}".lower()
            is_source_view = tool_name in {"read_file", "read_many_files"} or (
                tool_name == "exec_shell" and any(marker in command for marker in ["cat ", "nl -ba ", "sed -n", "head "]) and ".py" in command
            )

            if tool_name == "compare_output" and '"match":false' in combined:
                failure_signals.append("Output comparison failed.")
            elif tool_name == "search_text" and return_code == 1 and not stdout and not stderr:
                continue
            elif return_code != 0:
                failure_signals.append(
                    f"Tool `{tool_name}` failed with return code {return_code}."
                )
            elif tool_name in execution_like_tools and not is_source_view and any(
                marker in combined
                for marker in [
                    "wrong output",
                    "traceback",
                    "exception",
                    "assert",
                    "failed",
                    "error:",
                    "command not found",
                ]
            ):
                failure_signals.append(f"Tool `{tool_name}` reported a likely failure signal.")

            command_matches = re.findall(r"([A-Za-z0-9_.+-]+): command not found", combined)
            requirement_matches = re.findall(
                r"(?:requires|need|needs)\s+([A-Za-z0-9_.+-]+)\s+(?:in|to be in)",
                combined,
            )
            for candidate in [*command_matches, *requirement_matches]:
                if candidate not in {"tool", "command"}:
                    missing_tool_contexts.append(candidate)

        if not failure_signals:
            return None

        unique_signals = list(dict.fromkeys(failure_signals))
        guidance = [
            "Recent tool results indicate a likely failure or mismatch.",
            *unique_signals[:4],
        ]
        if missing_tool_contexts:
            missing_summary = ", ".join(dict.fromkeys(missing_tool_contexts))
            guidance.extend(
                [
                    f"Missing or unavailable commands were detected: {missing_summary}.",
                    f"Do not stop at the missing command. Prefer a local workaround in `{self.helper_dir}`: "
                    "check for substitutes, then consider `create_command_shim`, `create_shell_tool`, `create_python_tool`, `create_helper_tool`, "
                    "`download_url`, or `install_helper_tool`.",
                ]
            )
        invalid_verifier_present = any(
            "invalid verifier command" in f"{payload.get('stdout', '')}\n{payload.get('stderr', '')}".lower()
            for payload in payloads
        )
        if invalid_verifier_present:
            followup = self._recommended_verifier_hint({"messages": tool_messages}) or (
                "Use `run_tests` with a real test runner instead of `python test_file.py`."
            )
            guidance.extend(
                [
                    "A verifier command was rejected as invalid.",
                    followup,
                ]
            )
        guidance.extend(
            [
                "Before making another large edit, use the analysis tools when helpful: `extract_test_signals`, `summarize_failures`, and `propose_next_actions`.",
                "Then reproduce the smallest failing command, inspect the minimal relevant files, and apply a targeted fix.",
            ]
        )
        return " ".join(guidance)

    def _missing_tool_candidates(self, messages: list[BaseMessage]) -> list[str]:
        candidates: list[str] = []
        for payload in self._tool_payloads(messages):
            combined = f"{payload.get('stdout', '')}\n{payload.get('stderr', '')}".lower()
            command_matches = re.findall(r"([A-Za-z0-9_.+-]+): command not found", combined)
            requirement_matches = re.findall(
                r"(?:requires|need|needs)\s+([A-Za-z0-9_.+-]+)\s+(?:in|to be in)",
                combined,
            )
            for candidate in [*command_matches, *requirement_matches]:
                if candidate not in {"tool", "command"} and candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    def _tool_bootstrap_gap(self, state: AgentState) -> str | None:
        missing = self._missing_tool_candidates(state["messages"])
        if not missing:
            return None
        missing_summary = ", ".join(missing)
        guidance = (
            f"Missing commands detected: {missing_summary}. Do not stop here. "
            f"First decide how to replace or bootstrap the missing capability inside `{self.helper_dir}`. "
            "Everything inside that helper directory is automatically added to PATH for later shell commands. "
            "Preferred order: use an existing substitute, create a command shim with `create_command_shim`, create a tiny shell helper, create a tiny Python helper if Python exists, "
            "download a small public helper into the helper directory, install a helper into the helper directory, then continue the task using that helper. "
            "Do not ask for more files or dependencies. Build the missing capability yourself and keep going."
        )
        if self._should_emit_guidance(state, guidance):
            return guidance
        return None

    def _helper_paths_from_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        existing: list[str],
    ) -> list[str]:
        helper_paths = list(existing)
        tracked_tools = {
            "create_helper_tool": "path",
            "create_python_tool": "path",
            "create_shell_tool": "path",
            "create_command_shim": "command_name",
            "install_helper_tool": "destination",
            "download_url": "destination",
        }
        for call in tool_calls:
            tool_name = str(call.get("name", ""))
            arg_key = tracked_tools.get(tool_name)
            if not arg_key:
                continue
            args = call.get("args", {}) or {}
            candidate = str(args.get(arg_key, "")).strip()
            if not candidate:
                continue
            if tool_name == "create_command_shim":
                candidate = f"{self.helper_dir.rstrip('/')}/{Path(candidate).name}"
            if tool_name == "download_url" and not candidate.startswith(self.helper_dir):
                continue
            if candidate not in helper_paths:
                helper_paths.append(candidate)
        return helper_paths

    def _helper_roles_from_paths(
        self,
        helper_paths: list[str],
        existing: dict[str, str],
    ) -> dict[str, str]:
        roles = dict(existing)
        for raw_path in helper_paths:
            path = str(raw_path).strip()
            if not path:
                continue
            name = Path(path).name.lower()
            role = roles.get(path, "")
            if "selenium" in name or "verify" in name or "check" in name:
                role = role or "verifier"
            elif "probe" in name or "inspect" in name or "scan" in name:
                role = role or "inspector"
            elif "shim" in name or name in {"python", "python3", "jq", "file"}:
                role = role or "bootstrap"
            else:
                role = role or "helper"
            roles[path] = role
        return roles

    def _has_pytest_style_test_module(self, read_payload_texts: str) -> bool:
        lowered = read_payload_texts.lower()
        return "def test_" in lowered and "__main__" not in lowered

    def _inspected_python_sources(self, payloads: list[dict[str, Any]]) -> list[tuple[str, str]]:
        inspected: list[tuple[str, str]] = []
        for payload in payloads:
            tool_name = str(payload.get("_tool_name", "")).lower()
            command = str(payload.get("command", "") or "")
            stdout = str(payload.get("stdout", "") or "")
            if not stdout:
                continue
            candidate_paths = re.findall(r"/[^\s\"'|;]+\.py", command)
            if not candidate_paths:
                continue
            if tool_name in {"read_file", "read_many_files"}:
                for path in candidate_paths:
                    inspected.append((path, stdout))
                continue
            if tool_name == "exec_shell" and any(marker in command for marker in ["cat ", "nl -ba ", "sed -n", "head "]):
                for path in candidate_paths:
                    inspected.append((path, stdout))
        return inspected

    def _pytest_style_test_paths(self, payloads: list[dict[str, Any]]) -> list[str]:
        paths: list[str] = []
        for path, source_text in self._inspected_python_sources(payloads):
            if "def test_" not in source_text.lower() or "__main__" in source_text.lower():
                continue
            if path not in paths:
                paths.append(path)
        return paths

    def _recommended_verifier_hint(self, state: AgentState) -> str | None:
        payloads = self._tool_payloads(state["messages"])
        helper_roles = state.get("helper_roles", {})
        verifier_helpers = [path for path, role in helper_roles.items() if role == "verifier"]
        if verifier_helpers:
            helper = verifier_helpers[0]
            return (
                f"A verifier helper already exists: `{helper}`. Prefer reusing it to get direct positive or negative evidence before inventing another verification path."
            )
        pytest_paths = self._pytest_style_test_paths(payloads)
        if pytest_paths:
            target = pytest_paths[0]
            return (
                f"Prefer a real test runner. First check whether `pytest` exists with `check_command_available`, "
                f"then run `run_tests(command=\"pytest -q {target}\")` if it is available."
            )
        read_payload_texts = "\n".join(text for _, text in self._inspected_python_sources(payloads))
        if self._has_pytest_style_test_module(read_payload_texts):
            return (
                "The inspected verifier looks like a pytest-style test module. "
                "Prefer `run_tests` with a real test runner over `python <test_file>`."
            )
        return None

    def _invalid_verifier_command_reason(self, state: AgentState, tool_name: str, args: dict[str, Any]) -> str | None:
        if tool_name not in {"exec_shell", "run_tests"}:
            return None
        payloads = self._tool_payloads(state["messages"])
        pytest_paths = self._pytest_style_test_paths(payloads)
        read_payload_texts = "\n".join(text for _, text in self._inspected_python_sources(payloads))
        pytest_style_test_module = self._has_pytest_style_test_module(read_payload_texts)
        if not pytest_paths:
            pytest_paths = ["/app/test_outputs.py"] if pytest_style_test_module else []
        if not pytest_paths:
            return None
        command = str(args.get("command", "") or "")
        lowered = command.lower()
        offenders: list[str] = []
        for path in pytest_paths:
            basename = Path(path).name
            direct_patterns = [
                f"python {path}".lower(),
                f"python3 {path}".lower(),
                f"python {basename}".lower(),
                f"python3 {basename}".lower(),
            ]
            if any(pattern in lowered for pattern in direct_patterns):
                offenders.append(path)
        if not offenders and pytest_style_test_module:
            for match in re.findall(r"\bpython(?:3)?\s+([^\s\"']+\.py)\b", lowered):
                basename = Path(match).name
                if "test" in basename:
                    offenders.append(match)
        if not offenders and pytest_style_test_module:
            bare_matches = re.findall(r"\bpython(?:3)?\s+([^\s\"']+)\b", lowered)
            for match in bare_matches:
                basename = Path(match).name
                if basename.endswith(".py") and "test" in basename:
                    offenders.append(match)
        if not offenders:
            return None
        offender_summary = ", ".join(offenders)
        hint = self._recommended_verifier_hint(state)
        return (
            "Refusing to run an invalid verifier command. "
            f"The following files look like pytest-style test modules: {offender_summary}. "
            "Do not execute them as `python <file>`. Use a real test runner such as "
            f"`run_tests(command=\"pytest -q {offenders[0]}\")` if pytest is available, "
            "or another genuine verifier command that produces positive pass/fail evidence. "
            f"{hint or ''}".strip()
        )

    def _messages_for_replanner(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if not messages:
            return messages
        trimmed = list(messages)
        last = trimmed[-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            trimmed = trimmed[:-1]
        return trimmed

    def _verification_gap(self, state: AgentState) -> str | None:
        messages = state["messages"]
        if not messages:
            return None
        verification_state = state.get("verification_state", "")
        verification_summary = state.get("verification_summary", "")

        first_message = messages[0]
        if isinstance(first_message, dict):
            instruction = extract_text(first_message.get("content", "")).lower()
        else:
            instruction = extract_text(first_message.content).lower()
        payloads = self._tool_payloads(messages)
        execution_payloads = self._execution_like_payloads(payloads)
        tool_names = {str(payload.get("_tool_name", "")).lower() for payload in payloads}
        commands = "\n".join(str(payload.get("command", "")) for payload in execution_payloads).lower()
        combined_outputs = "\n".join(
            f"{payload.get('stdout', '')}\n{payload.get('stderr', '')}" for payload in execution_payloads
        ).lower()
        all_outputs = "\n".join(
            f"{payload.get('stdout', '')}\n{payload.get('stderr', '')}" for payload in payloads
        ).lower()
        reconnaissance_score = self._reconnaissance_score(payloads)
        read_payload_texts = "\n".join(
            str(payload.get("stdout", "") or "")
            for payload in payloads
            if str(payload.get("_tool_name", "")).lower() in {"read_file", "read_many_files"}
        )
        pytest_style_test_module = self._has_pytest_style_test_module(read_payload_texts)

        def has_meaningful_verifier_run() -> bool:
            if not execution_payloads:
                return False
            for payload in execution_payloads:
                tool_name = str(payload.get("_tool_name", "")).lower()
                command = str(payload.get("command", "")).lower()
                stdout = str(payload.get("stdout", "") or "").lower()
                stderr = str(payload.get("stderr", "") or "").lower()
                combined = f"{stdout}\n{stderr}"
                return_code = int(payload.get("return_code", 0) or 0)

                if "test_outputs.py" in command and "python /app/test_outputs.py" in command and pytest_style_test_module:
                    continue
                if tool_name == "compare_output" and '"match":true' in combined:
                    return True
                if tool_name == "run_tests" and return_code == 0:
                    return True
                if any(marker in command for marker in ["pytest", "python -m pytest"]) and return_code == 0:
                    return True
                if return_code == 0 and any(
                    marker in combined
                    for marker in [
                        "passed",
                        "alert successfully triggered",
                        "all tests passed",
                        "ok",
                    ]
                ):
                    return True
            return False

        gaps: list[str] = []
        if not payloads:
            gaps.append("You have not run any repository inspection or verification tools yet.")

        size_markers = [
            "5000 bytes",
            "under 5000 bytes",
            "less than 5000 bytes",
            "smaller than 5000 bytes",
            "<5000 bytes",
            "< 5000 bytes",
        ]
        if any(marker in instruction for marker in size_markers):
            has_size_check = "file_info" in tool_names or any(
                marker in commands or marker in all_outputs
                for marker in ["wc -c", "stat", "st_size", "bytes"]
            )
            if not has_size_check:
                gaps.append(
                    "This task has a file-size constraint. Verify the produced file size explicitly before finishing."
                )

        compile_markers = ["compile", "gcc", "clang", "make"]
        if any(marker in instruction for marker in compile_markers):
            if not any(marker in commands for marker in ["gcc", "clang", "make", "cc "]):
                gaps.append("This task requires a build or compile step. Run it before finishing.")

        test_markers = ["test", "verifier", "assert", "expected output", "reward"]
        if any(marker in instruction for marker in test_markers):
            if verification_state == "positively_verified":
                pass
            elif verification_state == "negatively_verified":
                gaps.append(
                    "A real verification attempt already produced a negative result. Change the solution strategy instead of declaring success."
                )
            elif verification_state == "verification_blocked":
                gaps.append(
                    f"Verification is currently blocked. {verification_summary or 'Repair the verifier path or use an available verification helper before finishing.'}"
                )
            if not (
                any(marker in commands for marker in ["pytest", "python -m pytest", "run_tests"])
                or has_meaningful_verifier_run()
            ):
                guidance = "This task is benchmarked by tests or a verifier. Run a relevant verification command before finishing."
                if pytest_style_test_module:
                    guidance += " The inspected test file looks like a pytest test module, so prefer `pytest -q <file>` or another real test runner instead of `python <file>`."
                gaps.append(guidance)
            elif "test_outputs.py" in commands and "python /app/test_outputs.py" in commands and pytest_style_test_module:
                gaps.append(
                    "Do not treat `python test_outputs.py` as a valid verifier run when the file is a pytest-style test module. Use `pytest` or another real verifier command."
                )
            elif "python /app/test_outputs.py" in commands and not has_meaningful_verifier_run():
                gaps.append(
                    "A bare `python /app/test_outputs.py` run is not enough positive evidence yet. Use a real verifier command and require an actual passing signal, not just exit code 0."
                )

        output_markers = [
            "expected output",
            "expected text",
            "expected string",
            "must output",
            "stdout",
            "print",
            "sample input",
            "sample output",
            "wrong output",
            "program output",
            "输出",
            "运行",
            "结果",
            "示例输入",
            "示例输出",
        ]
        if any(marker in instruction for marker in output_markers):
            if not self._has_runtime_validation(commands):
                gaps.append(
                    "This task depends on real program output. Run the produced artifact or script and check the output before finishing."
                )

        if self._has_invalid_completion(messages):
            gaps.append(
                "You already described the current solution as a placeholder or otherwise unsolved. Replace it with a real task-satisfying implementation before finishing."
            )

        if self._has_constraint_negotiation(messages):
            gaps.append(
                "Do not negotiate or ask to relax the benchmark constraints. Treat them as fixed and continue searching for a passing path."
            )

        if self._has_premature_impossibility_claim(messages) and reconnaissance_score < 6:
            gaps.append(
                "Do not conclude that the instance is infeasible yet. First inspect the concrete task-local files, format details, headers, and surrounding artifacts more thoroughly."
            )

        if not gaps:
            return None

        guidance = [
            "Do not finalize yet.",
            *gaps,
            "Use tools now, then return to the solution once the constraints are satisfied.",
        ]
        combined = " ".join(guidance)
        if self._should_emit_guidance(state, combined):
            return combined
        return None

    async def _run_shell_tool(
        self,
        *,
        tool_name: str,
        environment: BaseEnvironment,
        command: str,
        cwd: str,
        timeout_sec: int,
    ) -> str:
        self._emit(f"tool.start {tool_name} cwd={cwd} timeout={timeout_sec}s")
        self._emit_block("tool.command", command)
        helper_wrapped_command = (
            f"mkdir -p {self.helper_dir}\n"
            f"export PATH={self.helper_dir}:$PATH\n"
            "set -o pipefail 2>/dev/null || true\n"
            f"{command}"
        )
        result = await environment.exec(
            command=helper_wrapped_command,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )
        formatted = format_exec_result(
            command=helper_wrapped_command,
            return_code=result.return_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        self._emit(f"tool.finish {tool_name} rc={result.return_code}")
        self._emit_block("tool.result", formatted)
        return formatted

    def _build_tools(self, environment: BaseEnvironment):
        return build_agent_tools(self, environment)

    async def _invoke_model_streaming(
        self,
        runnable: Any,
        messages: list[BaseMessage],
        *,
        phase: str,
    ) -> AIMessage:
        self._emit(f"model.start phase={phase}")
        aggregated: BaseMessageChunk | None = None
        emitted_text = False

        try:
            async for chunk in runnable.astream(
                messages,
                config=self._langsmith_config(phase),
            ):
                if aggregated is None:
                    aggregated = chunk
                else:
                    aggregated = aggregated + chunk

                text = extract_text(getattr(chunk, "content", ""))
                if text:
                    emitted_text = True
                    snippet = (
                        text
                        if len(text) <= MAX_STREAM_SNIPPET_CHARS
                        else text[:MAX_STREAM_SNIPPET_CHARS] + "..."
                    )
                    self._emit(f"model.chunk phase={phase} {snippet}")
        except Exception as exc:
            self._emit(
                f"model.stream_error phase={phase} error={exc!r}; falling back to ainvoke"
            )
            response = await runnable.ainvoke(
                messages,
                config=self._langsmith_config(f"{phase}:fallback"),
            )
            text = extract_text(response.content)
            if text:
                self._emit_block(f"model.response phase={phase}", text)
            if getattr(response, "tool_calls", None):
                self._emit(
                    f"model.tool_calls phase={phase} count={len(response.tool_calls)}"
                )
            return response

        if aggregated is None:
            response = await runnable.ainvoke(
                messages,
                config=self._langsmith_config(f"{phase}:nonstream"),
            )
        else:
            response = message_chunk_to_message(aggregated)

        if not emitted_text:
            text = extract_text(response.content)
            if text:
                self._emit_block(
                    f"model.response phase={phase}",
                    truncate_text(text),
                )
        if getattr(response, "tool_calls", None):
            self._emit(f"model.tool_calls phase={phase} count={len(response.tool_calls)}")
        self._emit(f"model.finish phase={phase}")
        return response

    def _log_message_update(self, message: BaseMessage) -> None:
        if isinstance(message, ToolMessage):
            preview = truncate_text(extract_text(message.content), 600)
            self._emit_block(f"graph.tool_message id={message.tool_call_id}", preview)
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                names = ", ".join(call.get("name", "unknown") for call in message.tool_calls)
                self._emit(f"graph.ai tool_calls={names}")
            else:
                text = truncate_text(extract_text(message.content), 600)
                self._emit_block("graph.ai final", text)

    def _build_graph(self, environment: BaseEnvironment, instruction: str):
        tools = self._build_tools(environment)
        llm_with_tools = self._build_model().bind_tools(tools)
        plain_llm = self._build_model()
        system_prompt = SystemMessage(content=self._system_prompt())
        extra_prompts = [system_prompt]

        async def planner(state: AgentState) -> dict[str, Any]:
            prompt = SystemMessage(content=self._plan_prompt(instruction, state.get("completed_steps", [])))
            response = await self._invoke_model_streaming(
                plain_llm,
                [*extra_prompts, prompt, *state["messages"]],
                phase="planner",
            )
            text = extract_text(response.content)
            plan_text, current_step, done, final_response = self._parse_plan_response(text)
            updates: dict[str, Any] = {
                "plan_text": plan_text,
                "current_step": current_step,
                "done": done,
            }
            if done and final_response:
                updates["messages"] = [AIMessage(content=final_response)]
            return updates

        async def executor(state: AgentState) -> dict[str, Any]:
            next_step = state["step_count"] + 1
            is_last_step = next_step >= self.max_steps
            model = plain_llm if is_last_step else llm_with_tools
            prompts = [*extra_prompts]
            helper_paths = state.get("helper_paths", [])
            helper_roles = state.get("helper_roles", {})
            if helper_paths:
                inventory = "\n".join(
                    f"- {path} ({helper_roles.get(path, 'helper')})" for path in helper_paths[-20:]
                )
                prompts.append(
                    SystemMessage(
                        content=(
                            "Temporary helpers already available in this run. Reuse them before creating duplicates:\n"
                            f"{inventory}"
                        )
                    )
                )
            if state.get("failure_summary", "").strip():
                prompts.append(
                    SystemMessage(
                        content=(
                            "Recent failure analysis is available. Reuse it instead of rediscovering the same issue.\n\n"
                            f"FAILURE_SUMMARY:\n{state.get('failure_summary', '').strip()}"
                        )
                    )
                )
            current_step = state.get("current_step", "").strip() or "Inspect the task and continue."
            plan_text = state.get("plan_text", "").strip() or "No plan available."
            evidence_summary = self._evidence_summary(state)
            preferred_tools = self._preferred_tools_for_step(current_step)
            prompts.append(
                SystemMessage(
                    content=(
                        "Execute the current step of the plan. "
                        "Focus on this step before moving on. "
                        "Use tools when needed, keep experiments tight, and do not explain failure prematurely.\n\n"
                        f"PLAN:\n{plan_text}\n\nCURRENT_STEP:\n{current_step}"
                    )
                )
            )
            prompts.append(SystemMessage(content=self._tool_usage_guide()))
            prompts.append(
                SystemMessage(
                    content=(
                        f"Preferred tools for the current step: {preferred_tools}\n\n"
                        f"Recent evidence:\n{evidence_summary}"
                    )
                )
            )
            prompts.extend(self._executor_guidance(state))
            if is_last_step:
                prompts.append(
                    SystemMessage(
                        content="This is the last allowed model turn in this run. Do not call tools. Give the best final answer in English based only on the current information."
                    )
                )
            response = await self._invoke_model_streaming(
                model,
                [*prompts, *state["messages"]],
                phase=f"step-{next_step}",
            )
            return {
                "messages": [response],
                "step_count": next_step,
            }

        async def run_tools(state: AgentState) -> dict[str, Any]:
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage):
                return {"messages": []}

            tool_messages: list[ToolMessage] = []
            helper_paths = self._helper_paths_from_tool_calls(
                last_message.tool_calls,
                state.get("helper_paths", []),
            )
            helper_roles = self._helper_roles_from_paths(
                helper_paths,
                state.get("helper_roles", {}),
            )
            completed_steps = list(state.get("completed_steps", []))
            failure_signals = list(state.get("failure_signals", []))
            failure_summary = state.get("failure_summary", "")
            next_actions = list(state.get("next_actions", []))
            evidence_log = list(state.get("evidence_log", []))
            verification_state = state.get("verification_state", "unverified")
            verification_summary = state.get("verification_summary", "")
            blocked_verifiers = list(state.get("blocked_verifiers", []))
            verified_failures = list(state.get("verified_failures", []))
            verified_successes = list(state.get("verified_successes", []))
            rejected_solution_patterns = list(state.get("rejected_solution_patterns", []))

            def merge_command_update(update: dict[str, Any]) -> None:
                nonlocal helper_paths, helper_roles, completed_steps, failure_signals, failure_summary, next_actions
                helper_update = update.get("helper_paths")
                if isinstance(helper_update, list):
                    for path in helper_update:
                        if path not in helper_paths:
                            helper_paths.append(path)
                    helper_roles = self._helper_roles_from_paths(helper_paths, helper_roles)
                helper_role_update = update.get("helper_roles")
                if isinstance(helper_role_update, dict):
                    for path, role in helper_role_update.items():
                        if isinstance(path, str) and isinstance(role, str) and path.strip() and role.strip():
                            helper_roles[path.strip()] = role.strip()
                completed_update = update.get("completed_steps")
                if isinstance(completed_update, list):
                    for step in completed_update:
                        if step not in completed_steps:
                            completed_steps.append(step)
                failure_signal_update = update.get("failure_signals")
                if isinstance(failure_signal_update, list):
                    for signal in failure_signal_update:
                        if signal not in failure_signals:
                            failure_signals.append(signal)
                summary_update = update.get("failure_summary")
                if isinstance(summary_update, str) and summary_update.strip():
                    failure_summary = summary_update.strip()
                next_action_update = update.get("next_actions")
                if isinstance(next_action_update, list):
                    for action in next_action_update:
                        if action not in next_actions:
                            next_actions.append(action)
                evidence_update = update.get("evidence_log")
                if isinstance(evidence_update, list):
                    evidence_log[:] = self._merge_evidence(evidence_log, [item for item in evidence_update if isinstance(item, dict)])
                for key, target in [
                    ("blocked_verifiers", blocked_verifiers),
                    ("verified_failures", verified_failures),
                    ("verified_successes", verified_successes),
                    ("rejected_solution_patterns", rejected_solution_patterns),
                ]:
                    update_value = update.get(key)
                    if isinstance(update_value, list):
                        target[:] = self._merge_unique_strings(target, [str(item) for item in update_value])
                state_update_verification = update.get("verification_state")
                if isinstance(state_update_verification, str) and state_update_verification.strip():
                    nonlocal_verification_state[0] = state_update_verification.strip()
                state_update_summary = update.get("verification_summary")
                if isinstance(state_update_summary, str) and state_update_summary.strip():
                    nonlocal_verification_summary[0] = state_update_summary.strip()
                message_update = update.get("messages")
                if isinstance(message_update, list):
                    for item in message_update:
                        if isinstance(item, ToolMessage):
                            tool_messages.append(item)
                        elif isinstance(item, BaseMessage):
                            tool_messages.append(
                                ToolMessage(
                                    content=extract_text(item.content),
                                    tool_call_id=getattr(item, "tool_call_id", "unknown"),
                                    name=getattr(item, "name", None),
                                )
                                )

            nonlocal_verification_state = [verification_state]
            nonlocal_verification_summary = [verification_summary]

            for call in last_message.tool_calls:
                name = call["name"]
                args = call.get("args", {})
                self._emit(f"graph.tool_dispatch name={name}")
                invalid_verifier_reason = self._invalid_verifier_command_reason(state, name, args)
                if invalid_verifier_reason:
                    error_text = format_exec_result(
                        command=f"{name}({json.dumps(args, ensure_ascii=False)})",
                        return_code=1,
                        stdout="",
                        stderr=invalid_verifier_reason,
                    )
                    self._emit_block("graph.tool_error", error_text)
                    tool_messages.append(
                        ToolMessage(
                            content=error_text,
                            tool_call_id=call["id"],
                            name=name,
                        )
                    )
                    continue
                edited_paths = self._edited_paths_for_tool(name, args)
                protected_targets = [
                    path for path in edited_paths if self._is_protected_benchmark_path(path)
                ]
                if protected_targets:
                    error_text = format_exec_result(
                        command=f"{name}({json.dumps(args, ensure_ascii=False)})",
                        return_code=1,
                        stdout="",
                        stderr=(
                            "Refusing to modify protected benchmark harness paths: "
                            + ", ".join(protected_targets)
                            + ". Solve the task artifact instead of patching `/tests`."
                        ),
                    )
                    self._emit_block("graph.tool_error", error_text)
                    tool_messages.append(
                        ToolMessage(
                            content=error_text,
                            tool_call_id=call["id"],
                            name=name,
                        )
                    )
                    continue
                try:
                    selected_tool = next(tool for tool in tools if tool.name == name)
                except StopIteration:
                    error_text = format_exec_result(
                        command=f"invalid tool selection: {name}",
                        return_code=1,
                        stdout="",
                        stderr=f"Unknown tool requested: {name}",
                    )
                    self._emit_block("graph.tool_error", error_text)
                    tool_messages.append(
                        ToolMessage(
                            content=error_text,
                            tool_call_id=call["id"],
                            name=name,
                        )
                    )
                    continue
                try:
                    result = await selected_tool.ainvoke(args)
                except Exception as exc:
                    error_text = format_exec_result(
                        command=f"{name}({json.dumps(args, ensure_ascii=False)})",
                        return_code=1,
                        stdout="",
                        stderr=f"{type(exc).__name__}: {exc}",
                    )
                    self._emit_block("graph.tool_error", error_text)
                    result = error_text
                if isinstance(result, Command):
                    merge_command_update(result.update or {})
                    continue
                tool_message = ToolMessage(
                    content=result,
                    tool_call_id=call["id"],
                    name=name,
                )
                tool_messages.append(tool_message)
                try:
                    parsed_payload = json.loads(extract_text(result))
                except Exception:
                    parsed_payload = None
                if isinstance(parsed_payload, dict):
                    parsed_payload["_tool_name"] = name
                    new_evidence = self._extract_evidence_from_payload(parsed_payload)
                    evidence_log = self._merge_evidence(evidence_log, new_evidence)
                    (
                        verification_state,
                        verification_summary,
                        blocked_verifiers,
                        verified_failures,
                        verified_successes,
                        rejected_solution_patterns,
                    ) = self._derive_state_from_evidence(
                        evidence_log,
                        blocked_verifiers,
                        verified_failures,
                        verified_successes,
                        rejected_solution_patterns,
                    )
                    nonlocal_verification_state[0] = verification_state
                    nonlocal_verification_summary[0] = verification_summary
                    next_actions = self._derive_next_actions_from_state(
                        verification_state=verification_state,
                        blocked_verifiers=blocked_verifiers,
                        verified_failures=verified_failures,
                        rejected_solution_patterns=rejected_solution_patterns,
                        helper_roles=helper_roles,
                        existing=next_actions,
                    )
            verification_state = nonlocal_verification_state[0]
            verification_summary = nonlocal_verification_summary[0]
            next_actions = self._derive_next_actions_from_state(
                verification_state=verification_state,
                blocked_verifiers=blocked_verifiers,
                verified_failures=verified_failures,
                rejected_solution_patterns=rejected_solution_patterns,
                helper_roles=helper_roles,
                existing=next_actions,
            )
            if helper_paths != state.get("helper_paths", []):
                inventory = json.dumps(helper_paths, ensure_ascii=False, indent=2)
                self._emit_block("graph.helper_inventory", inventory)
            remediation_prompt = self._tool_failure_guidance(tool_messages)
            extra_messages: list[BaseMessage] = list(tool_messages)
            if remediation_prompt and self._should_emit_guidance(state, remediation_prompt):
                self._emit_block("graph.remediation_prompt", remediation_prompt)
                extra_messages.append(SystemMessage(content=remediation_prompt))
            return {
                "messages": extra_messages,
                "completed_steps": completed_steps,
                "helper_paths": helper_paths,
                "helper_roles": helper_roles,
                "failure_signals": failure_signals,
                "failure_summary": failure_summary,
                "next_actions": next_actions,
                "evidence_log": evidence_log,
                "verification_state": verification_state,
                "verification_summary": verification_summary,
                "blocked_verifiers": blocked_verifiers,
                "verified_failures": verified_failures,
                "verified_successes": verified_successes,
                "rejected_solution_patterns": rejected_solution_patterns,
            }

        async def replanner(state: AgentState) -> dict[str, Any]:
            completed_steps = list(state.get("completed_steps", []))
            current_step = state.get("current_step", "").strip()
            if current_step and current_step not in completed_steps:
                completed_steps.append(current_step)
            prompt = SystemMessage(content=self._replan_prompt(state))
            replanner_messages = self._messages_for_replanner(state["messages"])
            response = await self._invoke_model_streaming(
                plain_llm,
                [*extra_prompts, prompt, *replanner_messages],
                phase="replanner",
            )
            text = extract_text(response.content)
            plan_text, next_step_text, done, final_response = self._parse_plan_response(text)
            if not done and self._is_step_repeated(state, next_step_text):
                next_step_text = (
                    "Choose a different concrete step than the repeated one. "
                    "Prefer a new experiment, tool bootstrap, or direct verification action."
                )
            done_gap = self._verification_gap(state)
            if done and done_gap:
                self._emit_block("graph.replanner_blocked_done", done_gap)
                done = False
                if not next_step_text or next_step_text.lower() == "none":
                    next_step_text = "Run a real verifier command and inspect its concrete result."
            updates: dict[str, Any] = {
                "plan_text": plan_text,
                "current_step": next_step_text,
                "completed_steps": completed_steps,
                "done": done,
            }
            if done and final_response:
                updates["messages"] = [AIMessage(content=final_response)]
            return updates

        async def force_finalize(state: AgentState) -> dict[str, Any]:
            gap = self._verification_gap(state)
            gap_text = f" Remaining issue: {gap}" if gap else ""
            safe_messages = self._finalizable_messages(state["messages"])
            prompt = SystemMessage(
                content=(
                    "You have reached the tool-use step budget. "
                    "Provide the best final answer based on the work already completed. "
                    f"Do not call tools.{gap_text}"
                )
            )
            response = await self._invoke_model_streaming(
                plain_llm,
                [*extra_prompts, prompt, *safe_messages],
                phase="force-finalize",
            )
            return {"messages": [response]}

        def route_after_executor(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage):
                return END
            if last_message.tool_calls:
                if state["step_count"] >= self.max_steps:
                    return "force_finalize"
                return "tools"
            if state["step_count"] >= self.max_steps:
                return "force_finalize"
            return "replanner"

        def route_after_replanner(state: AgentState) -> str:
            if state.get("done"):
                if self._verification_gap(state):
                    return "executor"
                return END
            return "executor"

        graph = StateGraph(AgentState)
        graph.add_node("planner", planner)
        graph.add_node("executor", executor)
        graph.add_node("tools", run_tools)
        graph.add_node("replanner", replanner)
        graph.add_node("force_finalize", force_finalize)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "executor")
        graph.add_conditional_edges(
            "executor",
            route_after_executor,
            ["tools", "replanner", "force_finalize", END],
        )
        graph.add_edge("tools", "executor")
        graph.add_conditional_edges(
            "replanner",
            route_after_replanner,
            ["executor", END],
        )
        graph.add_edge("force_finalize", END)
        return graph.compile()

    def _collect_usage(self, messages: list[BaseMessage]) -> dict[str, int | None]:
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        found = False
        for message in messages:
            usage = getattr(message, "usage_metadata", None) or {}
            if not usage:
                continue
            found = True
            totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
            totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        if not found:
            return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
        return totals

    def _write_transcript(self, messages: list[BaseMessage]) -> None:
        lines: list[str] = []
        for index, message in enumerate(messages, start=1):
            role = message.type.upper()
            lines.append(f"## {index}. {role}")
            lines.append("")
            if getattr(message, "name", None):
                lines.append(f"name: {message.name}")
                lines.append("")
            lines.append(str(message.content))
            lines.append("")
        (self.logs_dir / "transcript.md").write_text("\n".join(lines))

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._emit("run.start")
        self._emit_block("run.instruction", instruction)
        graph = self._build_graph(environment, instruction)
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": instruction}],
            "step_count": 0,
            "helper_paths": [],
            "helper_roles": {},
            "failure_signals": [],
            "failure_summary": "",
            "next_actions": [],
            "evidence_log": [],
            "verification_state": "unverified",
            "verification_summary": "",
            "blocked_verifiers": [],
            "verified_failures": [],
            "verified_successes": [],
            "rejected_solution_patterns": [],
            "plan_text": "",
            "current_step": "",
            "completed_steps": [],
            "done": False,
        }

        result = await graph.ainvoke(
            initial_state,
            config=self._langsmith_config("graph-run"),
        )
        messages = result["messages"]
        for message in messages[1:]:
            self._log_message_update(message)

        self._write_transcript(messages)

        final_message = messages[-1]
        final_text = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content)
        )
        (self.logs_dir / "final_response.md").write_text(final_text)
        self._emit_block("run.final_response", final_text)

        usage = self._collect_usage(messages)
        context.n_input_tokens = usage["input_tokens"]
        context.n_output_tokens = usage["output_tokens"]
        context.metadata = {
            "model_name": getattr(llm, "model", None),
            "model_base_url": getattr(llm, "base_url", None),
            "max_steps": self.max_steps,
            "steps_used": result["step_count"],
            "capabilities": self._capabilities,
            "helper_dir": self.helper_dir,
            "helper_paths": result.get("helper_paths", []),
            "helper_roles": result.get("helper_roles", {}),
            "failure_signals": result.get("failure_signals", []),
            "failure_summary": result.get("failure_summary", ""),
            "next_actions": result.get("next_actions", []),
            "evidence_log": result.get("evidence_log", []),
            "verification_state": result.get("verification_state", "unverified"),
            "verification_summary": result.get("verification_summary", ""),
            "blocked_verifiers": result.get("blocked_verifiers", []),
            "verified_failures": result.get("verified_failures", []),
            "verified_successes": result.get("verified_successes", []),
            "rejected_solution_patterns": result.get("rejected_solution_patterns", []),
            "plan_text": result.get("plan_text", ""),
            "current_step": result.get("current_step", ""),
            "completed_steps": result.get("completed_steps", []),
            "done": result.get("done", False),
            "final_response_path": str(self.logs_dir / "final_response.md"),
            "transcript_path": str(self.logs_dir / "transcript.md"),
            "stream_log_path": str(self._stream_log_path),
        }
        self._emit("run.finish")


class LLMSingleLoopAgent(LangGraphTerminalBenchAgent):
    @staticmethod
    def name() -> str:
        return "llm-single-loop-agent"


class MultiStepTerminalAgent(LangGraphTerminalBenchAgent):
    @staticmethod
    def name() -> str:
        return "multi-step-terminal-agent"


class ActionTerminalAgent(LangGraphTerminalBenchAgent):
    @staticmethod
    def name() -> str:
        return "action-terminal-agent"


class PatchVerifyTerminalAgent(LangGraphTerminalBenchAgent):
    @staticmethod
    def name() -> str:
        return "patch-verify-terminal-agent"
