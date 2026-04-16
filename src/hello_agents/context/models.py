"""Define typed models used by the context-engineering layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hello_agents.memory import MemoryScope
from hello_agents.tools.base import ToolResult

ContextSectionName = Literal["rag", "memory", "tools"]


@dataclass(slots=True, frozen=True)
class ContextConfig:
    """Configure context sources, ordering, and lightweight budgets."""

    enable_rag: bool = True
    enable_memory: bool = True
    enable_tools: bool = True
    section_order: tuple[ContextSectionName, ...] = ("rag", "memory", "tools")
    max_total_chars: int = 4_000
    max_section_chars: int = 1_600
    max_items_per_section: int = 5
    max_item_chars: int = 320
    max_tool_results: int = 3


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


@dataclass(slots=True, frozen=True)
class ContextEnvelope:
    """Bundle the chosen sections with the final rendered user message."""

    sections: tuple[ContextSection, ...]
    rendered_message: str
