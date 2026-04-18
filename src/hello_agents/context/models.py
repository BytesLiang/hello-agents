"""Define typed models used by the context-engineering layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from hello_agents.memory import MemoryScope
from hello_agents.tools.base import ToolResult

ContextSectionName = Literal["rag", "memory", "tools"]


@runtime_checkable
class TokenEstimator(Protocol):
    """Estimate token usage for prompt content."""

    def estimate(self, text: str) -> int:
        """Return the estimated token count for a text payload."""


@dataclass(slots=True, frozen=True)
class ContextConfig:
    """Configure context sources, ordering, and prompt budgets."""

    enable_rag: bool = True
    enable_memory: bool = True
    enable_tools: bool = True
    section_order: tuple[ContextSectionName, ...] = ("rag", "memory", "tools")
    max_total_chars: int = 4_000
    max_section_chars: int = 1_600
    max_items_per_section: int = 5
    max_item_chars: int = 320
    max_tool_results: int = 3
    max_total_tokens: int | None = None
    max_section_tokens: int | None = None
    max_item_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class ContextRequest:
    """Describe the inputs required to compose effective model context."""

    message: str
    memory_scope: MemoryScope | None = None
    tool_results: tuple[ToolResult, ...] = ()


@dataclass(slots=True, frozen=True)
class ContextSection:
    """Represent one structured context section before final rendering."""

    name: ContextSectionName
    items: tuple[str, ...]
    rendered: str
    estimated_tokens: int = 0


@dataclass(slots=True, frozen=True)
class ContextSectionTrace:
    """Capture budgeting decisions for one context section."""

    name: ContextSectionName
    selected: bool
    original_item_count: int
    final_item_count: int
    estimated_tokens: int = 0
    estimated_chars: int = 0
    dropped_reasons: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ContextDebugInfo:
    """Expose lightweight budgeting and rendering diagnostics."""

    char_budget_applied: bool = False
    token_budget_applied: bool = False
    context_chars: int = 0
    context_tokens: int = 0
    rendered_message_chars: int = 0
    rendered_message_tokens: int = 0
    section_traces: tuple[ContextSectionTrace, ...] = ()


@dataclass(slots=True, frozen=True)
class ContextEnvelope:
    """Bundle the chosen sections with the final rendered user message."""

    sections: tuple[ContextSection, ...]
    rendered_message: str
    debug: ContextDebugInfo = field(default_factory=ContextDebugInfo)
