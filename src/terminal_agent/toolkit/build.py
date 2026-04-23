from typing import Any

from harbor.environments.base import BaseEnvironment
from langchain_core.tools import BaseTool

from src.terminal_agent.toolkit.extension import register_extension_tools
from src.terminal_agent.toolkit.edit import register_edit_tools
from src.terminal_agent.toolkit.inspect import register_inspect_tools
from src.terminal_agent.toolkit.repo import register_repo_tools
from src.terminal_agent.toolkit.verify import register_verify_tools
from src.terminal_agent.toolkit.web import register_web_tools


def build_agent_tools(agent: Any, environment: BaseEnvironment) -> list[BaseTool]:
    tools: list[BaseTool] = []
    register_inspect_tools(tools, agent, environment)
    register_edit_tools(tools, agent, environment)
    register_verify_tools(tools, agent, environment)
    register_repo_tools(tools, agent, environment)
    register_web_tools(tools, agent, environment)
    register_extension_tools(tools, agent, environment)
    return tools
