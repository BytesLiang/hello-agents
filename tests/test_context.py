"""Tests for the context-engineering subsystem."""

from __future__ import annotations

from hello_agents.context import ContextConfig, ContextEngine, ContextRequest
from hello_agents.memory import MemoryQueryResult, MemoryRecord, MemoryScope
from hello_agents.memory.models import MemoryKind
from hello_agents.rag.models import RagChunk
from hello_agents.tools import ToolResult


class StubMemory:
    """Provide deterministic memory query results for context tests."""

    def query(
        self,
        message: str,
        *,
        scope: MemoryScope,
        kinds=None,
        limit: int = 10,
    ) -> MemoryQueryResult:
        """Return a fixed memory payload regardless of the query."""

        del message, scope, kinds, limit
        return MemoryQueryResult(
            working=(
                MemoryRecord(
                    kind=MemoryKind.WORKING_PLAN,
                    user_id="user-1",
                    session_id="session-1",
                    agent_id="agent-1",
                    content="Ship context engineering.",
                    summary="Ship context engineering.",
                ),
            ),
            facts=(
                MemoryRecord(
                    kind=MemoryKind.SEMANTIC_FACT,
                    user_id="user-1",
                    session_id="session-1",
                    agent_id="agent-1",
                    content="The codename is Atlas.",
                    summary="Confirmed fact: The codename is Atlas.",
                    key="codename",
                    value="Atlas",
                    confirmed=True,
                ),
            ),
        )


class StubRagRetriever:
    """Provide deterministic retrieved chunks for context tests."""

    def query(self, text: str, *, top_k: int | None = None) -> list[RagChunk]:
        """Return a fixed RAG hit."""

        del text, top_k
        return [
            RagChunk(
                id="rag-1",
                source="docs/atlas.md",
                content="Atlas release notes and deployment checklist.",
            )
        ]


def test_context_engine_renders_single_enabled_source() -> None:
    """Verify one enabled source produces one prompt block."""

    engine = ContextEngine(
        rag=StubRagRetriever(),
        config=ContextConfig(enable_memory=False, enable_tools=False),
    )

    envelope = engine.compose(ContextRequest(message="Summarize Atlas."))

    assert [section.name for section in envelope.sections] == ["rag"]
    assert "[RAG]" in envelope.rendered_message
    assert "[MEMORY]" not in envelope.rendered_message
    assert "[TOOLS]" not in envelope.rendered_message


def test_context_engine_preserves_source_order() -> None:
    """Verify multiple sources render in the configured order."""

    scope = MemoryScope(user_id="user-1", session_id="session-1", agent_id="agent-1")
    engine = ContextEngine(memory=StubMemory(), rag=StubRagRetriever())

    envelope = engine.compose(
        ContextRequest(
            message="What should I do next?",
            memory_scope=scope,
            tool_results=(
                ToolResult(tool_name="echo", content="First observation"),
                ToolResult(tool_name="search", content="Latest observation"),
            ),
        )
    )

    assert [section.name for section in envelope.sections] == [
        "rag",
        "memory",
        "tools",
    ]
    assert envelope.rendered_message.index("[RAG]") < envelope.rendered_message.index(
        "[MEMORY]"
    )
    assert envelope.rendered_message.index(
        "[MEMORY]"
    ) < envelope.rendered_message.index("[TOOLS]")
    assert "search [success]: Latest observation" in envelope.rendered_message


def test_context_engine_applies_lightweight_budgets_without_empty_blocks() -> None:
    """Verify oversized sections are trimmed or omitted cleanly."""

    engine = ContextEngine(
        rag=StubRagRetriever(),
        config=ContextConfig(
            enable_memory=False,
            enable_tools=True,
            max_total_chars=40,
            max_section_chars=50,
            max_items_per_section=1,
            max_item_chars=18,
            max_tool_results=1,
        ),
    )

    envelope = engine.compose(
        ContextRequest(
            message="Summarize Atlas.",
            tool_results=(
                ToolResult(
                    tool_name="search",
                    content="This result is intentionally much longer than the budget.",
                ),
            ),
        )
    )

    assert "[RAG]" in envelope.rendered_message
    assert "[TOOLS]\n[/TOOLS]" not in envelope.rendered_message
    assert "[TOOLS]" not in envelope.rendered_message
    assert envelope.sections[0].name == "rag"


def test_context_engine_returns_raw_message_without_context() -> None:
    """Verify prompt rendering falls back to the original message."""

    engine = ContextEngine(
        config=ContextConfig(enable_rag=False, enable_memory=False, enable_tools=False)
    )

    envelope = engine.compose(ContextRequest(message="Just answer directly."))

    assert envelope.sections == ()
    assert envelope.rendered_message == "Just answer directly."
