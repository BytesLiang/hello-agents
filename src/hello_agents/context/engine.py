"""Compose retrieved context into a budgeted prompt payload."""

from __future__ import annotations

from collections.abc import Sequence

from hello_agents.context.models import (
    ContextConfig,
    ContextEnvelope,
    ContextRequest,
    ContextSection,
    ContextSectionName,
)
from hello_agents.memory import MemoryQueryResult, MemoryScope
from hello_agents.memory.base import Memory
from hello_agents.rag.models import RagChunk
from hello_agents.rag.retriever import RagRetriever
from hello_agents.tools.base import ToolResult


class ContextEngine:
    """Compose memory, RAG, and tool observations into prompt context."""

    def __init__(
        self,
        *,
        memory: Memory | None = None,
        rag: RagRetriever | None = None,
        config: ContextConfig | None = None,
    ) -> None:
        """Store source adapters and lightweight budgeting rules."""

        self.memory = memory
        self.rag = rag
        self.config = config or ContextConfig()

    def compose(self, request: ContextRequest) -> ContextEnvelope:
        """Build a rendered prompt payload from the available context sources."""

        sections_by_name = {
            "rag": self._compose_rag_section(request.message),
            "memory": self._compose_memory_section(
                request.message,
                request.memory_scope,
            ),
            "tools": self._compose_tools_section(request.tool_results),
        }

        remaining_chars = self.config.max_total_chars
        selected_sections: list[ContextSection] = []
        for name in self.config.section_order:
            section = sections_by_name.get(name)
            if section is None:
                continue
            budgeted = self._budget_section(section, remaining_chars=remaining_chars)
            if budgeted is None:
                continue
            selected_sections.append(budgeted)
            remaining_chars -= len(budgeted.rendered)
            if remaining_chars <= 0:
                break

        return ContextEnvelope(
            sections=tuple(selected_sections),
            rendered_message=_render_message(
                request.message,
                tuple(selected_sections),
            ),
        )

    def _compose_rag_section(self, message: str) -> ContextSection | None:
        """Build the RAG context section when retrieval is enabled."""

        if not self.config.enable_rag or self.rag is None:
            return None
        chunks = self.rag.query(message)
        items = tuple(_format_rag_chunk(chunk) for chunk in chunks)
        return _build_section("rag", items)

    def _compose_memory_section(
        self,
        message: str,
        memory_scope: MemoryScope | None,
    ) -> ContextSection | None:
        """Build the memory context section when retrieval is enabled."""

        if not self.config.enable_memory or self.memory is None or memory_scope is None:
            return None
        query_result = self.memory.query(message, scope=memory_scope)
        items = _memory_items(query_result)
        return _build_section("memory", items)

    def _compose_tools_section(
        self,
        tool_results: Sequence[ToolResult],
    ) -> ContextSection | None:
        """Build the recent tool-observation section."""

        if not self.config.enable_tools or not tool_results:
            return None
        recent_results = list(tool_results[-self.config.max_tool_results :])
        items = tuple(
            _format_tool_result(result) for result in reversed(recent_results)
        )
        return _build_section("tools", items)

    def _budget_section(
        self,
        section: ContextSection,
        *,
        remaining_chars: int,
    ) -> ContextSection | None:
        """Trim a section so it fits the configured section and total budgets."""

        if remaining_chars <= 0:
            return None

        target_chars = min(self.config.max_section_chars, remaining_chars)
        items = [
            _truncate_text(item, self.config.max_item_chars)
            for item in section.items[: self.config.max_items_per_section]
        ]
        items = [item for item in items if item.strip()]
        if not items:
            return None

        while items:
            candidate = _build_section(section.name, tuple(items))
            if candidate is None:
                return None
            if len(candidate.rendered) <= target_chars:
                return candidate
            if len(items) == 1:
                return self._fit_single_item_section(
                    section.name,
                    items[0],
                    target_chars=target_chars,
                )
            items.pop()

        return None

    def _fit_single_item_section(
        self,
        name: ContextSectionName,
        item: str,
        *,
        target_chars: int,
    ) -> ContextSection | None:
        """Shrink a single-item section until it fits the remaining budget."""

        overhead = len(_render_section(name, ("",)))
        available_chars = target_chars - overhead
        if available_chars <= 0:
            return None
        return _build_section(name, (_truncate_text(item, available_chars),))


def _build_section(
    name: ContextSectionName,
    items: Sequence[str],
) -> ContextSection | None:
    """Create a rendered section when at least one item is present."""

    normalized_items = tuple(item for item in items if item.strip())
    if not normalized_items:
        return None
    return ContextSection(
        name=name,
        items=normalized_items,
        rendered=_render_section(name, normalized_items),
    )


def _render_section(name: ContextSectionName, items: Sequence[str]) -> str:
    """Render a structured section into the legacy prompt-block format."""

    if name == "memory":
        body = "\n\n".join(items)
    else:
        body = "\n".join(f"- {item}" for item in items)
    tag = name.upper()
    return f"[{tag}]\n{body}\n[/{tag}]"


def _render_message(message: str, sections: Sequence[ContextSection]) -> str:
    """Render the final user-facing message for the LLM request."""

    if not sections:
        return message
    rendered_sections = "\n\n".join(section.rendered for section in sections)
    return f"{rendered_sections}\n\nUser request:\n{message}"


def _memory_items(query_result: MemoryQueryResult) -> tuple[str, ...]:
    """Convert queried memory into structured prompt sub-sections."""

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
            + "\n".join(f"- {record.summary}" for record in query_result.preferences)
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

    return tuple(sections)


def _format_rag_chunk(chunk: RagChunk) -> str:
    """Format one retrieved RAG chunk for prompt inclusion."""

    snippet = _single_line_snippet(chunk.content, max_chars=300)
    return f"{chunk.source}: {snippet}"


def _format_tool_result(result: ToolResult) -> str:
    """Format one tool result for prompt inclusion."""

    status = "success" if result.success else "failure"
    snippet = _single_line_snippet(result.content, max_chars=240)
    return f"{result.tool_name} [{status}]: {snippet}"


def _single_line_snippet(content: str, *, max_chars: int) -> str:
    """Collapse multi-line content into a bounded single-line snippet."""

    return _truncate_text(" ".join(content.strip().splitlines()), max_chars)


def _truncate_text(content: str, max_chars: int) -> str:
    """Return content truncated to a maximum character count."""

    normalized = content.strip()
    if max_chars <= 0:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return normalized[: max_chars - 3].rstrip() + "..."
