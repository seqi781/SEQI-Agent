from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class EvidenceItem(TypedDict):
    type: Literal["inspection", "environment", "verification", "artifact_change", "failure"]
    claim: str
    scope: Literal["environment", "verifier", "artifact", "strategy", "solution"]
    confidence: Literal["low", "medium", "high"]
    source: str
    detail: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    helper_paths: list[str]
    helper_roles: dict[str, str]
    failure_signals: list[str]
    failure_summary: str
    next_actions: list[str]
    evidence_log: list[EvidenceItem]
    verification_state: str
    verification_summary: str
    blocked_verifiers: list[str]
    verified_failures: list[str]
    verified_successes: list[str]
    rejected_solution_patterns: list[str]
    plan_text: str
    current_step: str
    completed_steps: list[str]
    done: bool
