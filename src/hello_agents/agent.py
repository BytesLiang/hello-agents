"""Core agent primitives for the hello_agents package."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from hello_agents.llm.client import LLMClient
from hello_agents.memory import MemoryQueryResult, MemoryScope
from hello_agents.memory.base import Memory
from hello_agents.rag.models import RagChunk
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
    ) -> None:
        """Store the common agent identity, LLM dependency, and tools."""

        self.name = name
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.use_tools = use_tools
        self.memory = memory
        self.rag = rag
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
    ) -> str:
        """Inject queried memory context into the user message when enabled."""

        blocks: list[str] = []
        if self.rag is not None:
            rag_chunks = self.rag.query(message)
            rag_block = self._render_rag_chunks(rag_chunks)
            if rag_block:
                blocks.append(rag_block)

        if self.memory is not None and memory_scope is not None:
            query_result = self.memory.query(message, scope=memory_scope)
            memory_block = self._render_memory_query_result(query_result)
            if memory_block:
                blocks.append(memory_block)

        if not blocks:
            return message
        return "\n\n".join(blocks) + f"\n\nUser request:\n{message}"

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

    @staticmethod
    def _render_memory_query_result(query_result: MemoryQueryResult) -> str:
        """Render queried memory into a prompt block."""

        sections: list[str] = []
        plan_entries = [
            record.content
            for record in query_result.working
            if record.kind.value == "working_plan"
        ]
        if plan_entries:
            sections.append(
                "Current plan:\n" + "\n".join(f"- {plan}" for plan in plan_entries[-2:])
            )

        context_entries = [
            record.content
            for record in query_result.working
            if record.kind.value == "working_context"
        ]
        if context_entries:
            sections.append(
                "Session context:\n"
                + "\n".join(f"- {item}" for item in context_entries[-4:])
            )

        if query_result.preferences:
            sections.append(
                "User preferences:\n"
                + "\n".join(
                    f"- {record.summary}" for record in query_result.preferences
                )
            )

        if query_result.facts:
            sections.append(
                "Confirmed facts:\n"
                + "\n".join(f"- {record.summary}" for record in query_result.facts)
            )

        if query_result.episodes:
            sections.append(
                "Relevant task history:\n"
                + "\n".join(f"- {record.summary}" for record in query_result.episodes)
            )

        if query_result.procedures:
            sections.append(
                "Successful experience:\n"
                + "\n".join(f"- {record.content}" for record in query_result.procedures)
            )

        if not sections:
            return ""
        return "[MEMORY]\n" + "\n\n".join(sections) + "\n[/MEMORY]"

    @staticmethod
    def _render_rag_chunks(chunks: Sequence[RagChunk]) -> str:
        """Render retrieved RAG chunks into a prompt block."""

        if not chunks:
            return ""
        lines: list[str] = []
        for chunk in chunks:
            snippet = chunk.content.strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300].rstrip() + "..."
            lines.append(f"- {chunk.source}: {snippet}")
        return "[RAG]\n" + "\n".join(lines) + "\n[/RAG]"

    @abstractmethod
    def run(
        self,
        message: str,
        *,
        memory_scope: MemoryScope | None = None,
    ) -> str:
        """Execute the agent's primary behavior."""
