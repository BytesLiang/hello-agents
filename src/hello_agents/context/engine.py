"""Compose retrieved context into a budgeted prompt payload."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from hello_agents.context.models import (
    ContextConfig,
    ContextDebugInfo,
    ContextEnvelope,
    ContextRequest,
    ContextSection,
    ContextSectionName,
    ContextSectionTrace,
    TokenEstimator,
)
from hello_agents.memory import MemoryQueryResult, MemoryScope
from hello_agents.memory.base import Memory
from hello_agents.rag.models import RagChunk
from hello_agents.rag.retriever import RagRetriever
from hello_agents.tools.base import ToolResult


@dataclass(slots=True, frozen=True)
class _BudgetedText:
    """Store text trimmed to one or more prompt budgets."""

    content: str
    reasons: tuple[str, ...] = ()


class ApproximateTokenEstimator:
    """Estimate token usage without relying on a model-specific tokenizer."""

    def estimate(self, text: str) -> int:
        """Approximate tokens from ASCII and non-ASCII character counts."""

        normalized = text.strip()
        if not normalized:
            return 0
        compact = "".join(
            character for character in normalized if not character.isspace()
        )
        if not compact:
            return 0
        ascii_chars = sum(1 for character in compact if ord(character) < 128)
        non_ascii_chars = len(compact) - ascii_chars
        return ceil(ascii_chars / 4) + non_ascii_chars


class ContextEngine:
    """Compose memory, RAG, and tool observations into prompt context."""

    def __init__(
        self,
        *,
        memory: Memory | None = None,
        rag: RagRetriever | None = None,
        config: ContextConfig | None = None,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        """Store source adapters, budgets, and token estimation policy."""

        self.memory = memory
        self.rag = rag
        self.config = config or ContextConfig()
        self._token_estimator = token_estimator or ApproximateTokenEstimator()

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
        remaining_tokens = self.config.max_total_tokens
        selected_sections: list[ContextSection] = []
        section_traces: list[ContextSectionTrace] = []
        for name in self.config.section_order:
            section = sections_by_name.get(name)
            if section is None:
                continue
            budgeted, trace = self._budget_section(
                section,
                remaining_chars=remaining_chars,
                remaining_tokens=remaining_tokens,
            )
            section_traces.append(trace)
            if budgeted is None:
                continue
            selected_sections.append(budgeted)
            remaining_chars -= len(budgeted.rendered)
            if remaining_tokens is not None:
                remaining_tokens = max(0, remaining_tokens - budgeted.estimated_tokens)
            if remaining_chars <= 0:
                break

        rendered_message = _render_message(
            request.message,
            tuple(selected_sections),
        )
        return ContextEnvelope(
            sections=tuple(selected_sections),
            rendered_message=rendered_message,
            debug=ContextDebugInfo(
                char_budget_applied=True,
                token_budget_applied=_token_budget_enabled(self.config),
                context_chars=sum(
                    len(section.rendered) for section in selected_sections
                ),
                context_tokens=sum(
                    section.estimated_tokens for section in selected_sections
                ),
                rendered_message_chars=len(rendered_message),
                rendered_message_tokens=self._token_estimator.estimate(
                    rendered_message
                ),
                section_traces=tuple(section_traces),
            ),
        )

    def _compose_rag_section(self, message: str) -> ContextSection | None:
        """Build the RAG context section when retrieval is enabled."""

        if not self.config.enable_rag or self.rag is None:
            return None
        chunks = self.rag.query(message)
        items = tuple(_format_rag_chunk(chunk) for chunk in chunks)
        return _build_section("rag", items, token_estimator=self._token_estimator)

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
        return _build_section("memory", items, token_estimator=self._token_estimator)

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
        return _build_section("tools", items, token_estimator=self._token_estimator)

    def _budget_section(
        self,
        section: ContextSection,
        *,
        remaining_chars: int,
        remaining_tokens: int | None,
    ) -> tuple[ContextSection | None, ContextSectionTrace]:
        """Trim a section so it fits the configured section and total budgets."""

        if remaining_chars <= 0:
            return None, self._build_trace(
                name=section.name,
                selected=False,
                original_item_count=len(section.items),
                final_item_count=0,
                reasons=("max_total_chars",),
            )

        reasons: set[str] = set()
        if len(section.items) > self.config.max_items_per_section:
            reasons.add("max_items_per_section")
        budgeted_items = [
            self._fit_item_budget(item)
            for item in section.items[: self.config.max_items_per_section]
        ]
        for item in budgeted_items:
            reasons.update(item.reasons)
        candidate_items = [
            item.content for item in budgeted_items if item.content.strip()
        ]
        if not candidate_items:
            return None, self._build_trace(
                name=section.name,
                selected=False,
                original_item_count=len(section.items),
                final_item_count=0,
                reasons=tuple(sorted(reasons | {"empty_after_item_budget"})),
            )

        while candidate_items:
            candidate = _build_section(
                section.name,
                tuple(candidate_items),
                token_estimator=self._token_estimator,
            )
            if candidate is None:
                return None, self._build_trace(
                    name=section.name,
                    selected=False,
                    original_item_count=len(section.items),
                    final_item_count=0,
                    reasons=tuple(sorted(reasons | {"empty_after_item_budget"})),
                )
            violations = self._budget_violations(
                candidate,
                remaining_chars=remaining_chars,
                remaining_tokens=remaining_tokens,
            )
            if not violations:
                return candidate, self._build_trace(
                    name=section.name,
                    selected=True,
                    original_item_count=len(section.items),
                    final_item_count=len(candidate.items),
                    estimated_tokens=candidate.estimated_tokens,
                    estimated_chars=len(candidate.rendered),
                    reasons=tuple(sorted(reasons)),
                )
            reasons.update(violations)
            if len(candidate_items) == 1:
                fitted = self._fit_single_item_section(
                    section.name,
                    candidate_items[0],
                    remaining_chars=remaining_chars,
                    remaining_tokens=remaining_tokens,
                )
                if fitted is None:
                    break
                return fitted, self._build_trace(
                    name=section.name,
                    selected=True,
                    original_item_count=len(section.items),
                    final_item_count=len(fitted.items),
                    estimated_tokens=fitted.estimated_tokens,
                    estimated_chars=len(fitted.rendered),
                    reasons=tuple(sorted(reasons)),
                )
            candidate_items.pop()

        return None, self._build_trace(
            name=section.name,
            selected=False,
            original_item_count=len(section.items),
            final_item_count=0,
            reasons=tuple(sorted(reasons)),
        )

    def _fit_single_item_section(
        self,
        name: ContextSectionName,
        item: str,
        *,
        remaining_chars: int,
        remaining_tokens: int | None,
    ) -> ContextSection | None:
        """Shrink a single-item section until it fits the remaining budget."""

        target_chars = min(self.config.max_section_chars, remaining_chars)
        overhead = len(_render_section(name, ("",)))
        available_chars = target_chars - overhead
        if available_chars <= 0:
            return None
        target_tokens = _min_optional(
            self.config.max_section_tokens,
            remaining_tokens,
        )
        overhead_tokens = self._token_estimator.estimate(_render_section(name, ("",)))
        available_tokens = None
        if target_tokens is not None:
            available_tokens = target_tokens - overhead_tokens
            if available_tokens <= 0:
                return None
        fitted_item = item
        fitted_item = _truncate_text(fitted_item, available_chars)
        fitted_item = _truncate_to_token_budget(
            fitted_item,
            max_tokens=available_tokens,
            token_estimator=self._token_estimator,
        )
        if not fitted_item.strip():
            return None
        section = _build_section(
            name,
            (fitted_item,),
            token_estimator=self._token_estimator,
        )
        if section is None:
            return None
        if self._budget_violations(
            section,
            remaining_chars=remaining_chars,
            remaining_tokens=remaining_tokens,
        ):
            return None
        return section

    def _fit_item_budget(self, item: str) -> _BudgetedText:
        """Apply per-item char and token budgets."""

        content = item.strip()
        reasons: set[str] = set()
        truncated_by_chars = _truncate_text(content, self.config.max_item_chars)
        if truncated_by_chars != content:
            reasons.add("max_item_chars")
        content = truncated_by_chars
        truncated_by_tokens = _truncate_to_token_budget(
            content,
            max_tokens=self.config.max_item_tokens,
            token_estimator=self._token_estimator,
        )
        if truncated_by_tokens != content:
            reasons.add("max_item_tokens")
        return _BudgetedText(
            content=truncated_by_tokens,
            reasons=tuple(sorted(reasons)),
        )

    def _budget_violations(
        self,
        section: ContextSection,
        *,
        remaining_chars: int,
        remaining_tokens: int | None,
    ) -> tuple[str, ...]:
        """Return all budget constraints violated by a rendered section."""

        reasons: set[str] = set()
        if len(section.rendered) > self.config.max_section_chars:
            reasons.add("max_section_chars")
        if len(section.rendered) > remaining_chars:
            reasons.add("max_total_chars")
        if (
            self.config.max_section_tokens is not None
            and section.estimated_tokens > self.config.max_section_tokens
        ):
            reasons.add("max_section_tokens")
        if remaining_tokens is not None and section.estimated_tokens > remaining_tokens:
            reasons.add("max_total_tokens")
        return tuple(sorted(reasons))

    @staticmethod
    def _build_trace(
        *,
        name: ContextSectionName,
        selected: bool,
        original_item_count: int,
        final_item_count: int,
        estimated_tokens: int = 0,
        estimated_chars: int = 0,
        reasons: tuple[str, ...] = (),
    ) -> ContextSectionTrace:
        """Create a stable trace payload for one section decision."""

        return ContextSectionTrace(
            name=name,
            selected=selected,
            original_item_count=original_item_count,
            final_item_count=final_item_count,
            estimated_tokens=estimated_tokens,
            estimated_chars=estimated_chars,
            dropped_reasons=reasons,
        )


def _build_section(
    name: ContextSectionName,
    items: Sequence[str],
    *,
    token_estimator: TokenEstimator,
) -> ContextSection | None:
    """Create a rendered section when at least one item is present."""

    normalized_items = tuple(item for item in items if item.strip())
    if not normalized_items:
        return None
    rendered = _render_section(name, normalized_items)
    return ContextSection(
        name=name,
        items=normalized_items,
        rendered=rendered,
        estimated_tokens=token_estimator.estimate(rendered),
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


def _truncate_to_token_budget(
    content: str,
    *,
    max_tokens: int | None,
    token_estimator: TokenEstimator,
) -> str:
    """Return content truncated to an estimated token budget."""

    normalized = content.strip()
    if max_tokens is None:
        return normalized
    if max_tokens <= 0:
        return ""
    if token_estimator.estimate(normalized) <= max_tokens:
        return normalized
    if token_estimator.estimate("...") > max_tokens:
        return ""

    low = 0
    high = len(normalized)
    best = ""
    while low <= high:
        midpoint = (low + high) // 2
        prefix = normalized[:midpoint].rstrip()
        candidate = prefix
        if midpoint < len(normalized):
            candidate = f"{prefix}..." if prefix else "..."
        if token_estimator.estimate(candidate) <= max_tokens:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


def _min_optional(first: int | None, second: int | None) -> int | None:
    """Return the minimum present value across two optional limits."""

    values = [value for value in (first, second) if value is not None]
    if not values:
        return None
    return min(values)


def _token_budget_enabled(config: ContextConfig) -> bool:
    """Return whether any token-based budget is configured."""

    return any(
        value is not None
        for value in (
            config.max_total_tokens,
            config.max_section_tokens,
            config.max_item_tokens,
        )
    )
