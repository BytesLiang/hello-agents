"""Core agent primitives for the hello_agents package."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from hello_agents.context import ContextEngine, ContextRequest
from hello_agents.llm.client import LLMClient
from hello_agents.memory import MemoryScope
from hello_agents.memory.base import Memory
from hello_agents.rag.retriever import RagRetriever
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
        memory: Memory | None = None,
        rag: RagRetriever | None = None,
        context_engine: ContextEngine | None = None,
    ) -> None:
        """Store the common agent identity, LLM dependency, and tools."""

        self.name = name
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.use_tools = use_tools
        self.memory = memory
        self.rag = rag
        self.context_engine = context_engine or ContextEngine(memory=memory, rag=rag)
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

    def build_effective_message(
        self,
        message: str,
        *,
        memory_scope: MemoryScope | None = None,
        tool_results: Sequence[ToolResult] = (),
    ) -> str:
        """Build the prompt-ready message for the current user request."""

        envelope = self.context_engine.compose(
            ContextRequest(
                message=message,
                memory_scope=memory_scope,
                tool_results=tuple(tool_results),
            )
        )
        return envelope.rendered_message

    def persist_memory_turn(
        self,
        *,
        memory_scope: MemoryScope | None,
        message: str,
        response: str,
        tool_results: tuple[ToolResult, ...] = (),
        success: bool = True,
    ) -> None:
        """Propose and commit memory for a completed turn."""

        if self.memory is None or memory_scope is None:
            return
        try:
            proposal = self.memory.propose(
                message,
                response,
                scope=memory_scope,
                tool_results=tool_results,
                success=success,
            )
            self.memory.commit(proposal, scope=memory_scope)
        except Exception:
            self.logger.exception("Failed to persist memory for agent=%s", self.name)

    @abstractmethod
    def run(
        self,
        message: str,
        *,
        memory_scope: MemoryScope | None = None,
    ) -> str:
        """Execute the agent's primary behavior."""
