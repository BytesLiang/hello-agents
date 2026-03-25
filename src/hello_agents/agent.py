"""Core agent primitives for the hello_agents package."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hello_agents.llm.client import LLMClient
from hello_agents.tools.base import ToolResult
from hello_agents.tools.registry import ToolRegistry


class Agent(ABC):
    """Define the top-level abstract contract for all LLM-backed agents."""

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
    ) -> None:
        """Store the common agent identity, LLM dependency, and tools."""

        self.name = name
        self.llm = llm
        self.tools = tools or ToolRegistry()

    def describe_tools(self) -> list[dict[str, object]]:
        """Return the tools available to the agent."""

        return self.tools.describe_tools()

    def execute_tool(self, name: str, payload: dict[str, object]) -> ToolResult:
        """Execute one of the agent's registered tools."""

        return self.tools.execute(name, payload)

    @abstractmethod
    def run(self, message: str) -> str:
        """Execute the agent's primary behavior."""
