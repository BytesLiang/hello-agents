"""Core agent primitives for the hello_agents package."""

from __future__ import annotations

import logging
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
        use_tools: bool = False,
    ) -> None:
        """Store the common agent identity, LLM dependency, and tools."""

        self.name = name
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.use_tools = use_tools
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.name}")

    def describe_tools(self) -> list[dict[str, object]]:
        """Return the tools available to the agent."""

        if not self.use_tools:
            return []
        return self.tools.describe_tools()

    def execute_tool(self, name: str, payload: dict[str, object]) -> ToolResult:
        """Execute one of the agent's registered tools."""

        if not self.use_tools:
            raise RuntimeError("Tools are disabled for this agent.")
        self.logger.info(
            "Executing tool '%s' with payload keys=%s",
            name,
            sorted(payload),
        )
        return self.tools.execute(name, payload)

    @abstractmethod
    def run(self, message: str) -> str:
        """Execute the agent's primary behavior."""
