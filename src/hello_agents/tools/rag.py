"""Expose RAG retrieval as a tool."""

from __future__ import annotations

from collections.abc import Sequence

from hello_agents.rag.models import RagChunk
from hello_agents.rag.retriever import RagRetriever
from hello_agents.tools.base import Tool, ToolParameter, ToolResult, ToolSchema


class RagSearchTool(Tool):
    """Tool wrapper that queries RAG chunks."""

    def __init__(self, retriever: RagRetriever) -> None:
        """Store the retriever and declare the tool schema."""

        super().__init__(
            name="rag_search",
            description="Search indexed documents for relevant context.",
            schema=ToolSchema(
                parameters=(
                    ToolParameter(
                        name="query",
                        description="Search query used for retrieval.",
                    ),
                    ToolParameter(
                        name="top_k",
                        description="Maximum number of chunks to return.",
                        value_type="integer",
                        required=False,
                    ),
                )
            ),
        )
        self._retriever = retriever

    def execute(self, payload: dict[str, object]) -> ToolResult:
        """Query RAG chunks and return a concise summary."""

        query = str(payload.get("query", "")).strip()
        top_k = payload.get("top_k")
        limit = int(top_k) if isinstance(top_k, int) else None
        chunks = self._retriever.query(query, top_k=limit)
        content = _format_chunks(chunks)
        return ToolResult(tool_name=self.name, content=content)


def _format_chunks(chunks: Sequence[RagChunk]) -> str:
    """Format chunks into a readable string payload."""

    if not chunks:
        return "No relevant context found."
    lines = []
    for chunk in chunks:
        snippet = chunk.content.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "..."
        lines.append(f"- {chunk.source}: {snippet}")
    return "\n".join(lines)
