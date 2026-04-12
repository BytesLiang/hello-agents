"""Tests for the lightweight RAG subsystem."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hello_agents.agent import Agent
from hello_agents.llm.client import LLMClient
from hello_agents.rag.config import RagConfig
from hello_agents.rag.indexer import RagIndexer
from hello_agents.rag.models import RagChunk
from hello_agents.rag.retriever import RagRetriever
from hello_agents.tools.rag import RagSearchTool


class StubEmbedder:
    """Return deterministic embeddings for tests."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return simple vectors based on input length."""

        return [[float(len(text)), 0.1, 0.2] for text in texts]


class StubRagStore:
    """In-memory store for RAG test coverage."""

    def __init__(self) -> None:
        """Initialize an empty index."""

        self._chunks: list[RagChunk] = []
        self._embeddings: list[list[float]] = []

    def upsert(
        self,
        chunks: Sequence[RagChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Persist provided chunks in memory."""

        self._chunks.extend(chunks)
        self._embeddings.extend([list(vector) for vector in embeddings])

    def search(self, embedding: Sequence[float], *, top_k: int) -> list[RagChunk]:
        """Return the first top-k chunks for simplicity."""

        del embedding
        return list(self._chunks[:top_k])


class DemoAgent(Agent):
    """Concrete agent used for prompt injection tests."""

    def run(self, message: str, *, memory_scope=None) -> str:
        """Return the message unchanged."""

        del memory_scope
        return message


def test_indexer_indexes_local_files(tmp_path: Path) -> None:
    """Verify indexing returns a positive chunk count."""

    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "# Intro\n\nHello RAG world.\n\n## Details\n\nThis is a test document."
    )
    config = RagConfig(
        enabled=True,
        paths=(tmp_path,),
        top_k=3,
        chunk_size=40,
        chunk_overlap=5,
        embed=None,
    )

    store = StubRagStore()

    class StubConverter:
        """Provide a minimal MarkItDown-like converter."""

        def convert(self, path: str) -> object:
            """Return a result with text_content."""

            with open(path, encoding="utf-8") as handle:
                content = handle.read()
            return type("Result", (), {"text_content": content})()

    indexer = RagIndexer.__new__(RagIndexer)
    indexer._config = config  # type: ignore[attr-defined]
    indexer._embedder = StubEmbedder()  # type: ignore[attr-defined]
    indexer._store = store  # type: ignore[attr-defined]
    indexer._converter = StubConverter()  # type: ignore[attr-defined]

    indexed = indexer.index_folder(tmp_path)
    assert indexed > 0
    assert store._chunks
    assert store._chunks[0].metadata["heading_path"] == "Intro"
    assert "Section: Intro" in store._chunks[0].content


def test_retriever_returns_chunks() -> None:
    """Verify retriever returns chunks from the store."""

    config = RagConfig(enabled=True, embed=None)
    store = StubRagStore()
    chunk = RagChunk(id="1", source="doc.txt", content="Hello retrieval")
    store.upsert([chunk], [[0.1, 0.2, 0.3]])

    retriever = RagRetriever.__new__(RagRetriever)
    retriever._config = config  # type: ignore[attr-defined]
    retriever._embedder = StubEmbedder()  # type: ignore[attr-defined]
    retriever._store = store  # type: ignore[attr-defined]

    results = retriever.query("Hello")
    assert results
    assert results[0].content == "Hello retrieval"


def test_rag_tool_returns_formatted_payload() -> None:
    """Verify tool output includes chunk content."""

    store = StubRagStore()
    chunk = RagChunk(id="1", source="doc.txt", content="Tool context")
    store.upsert([chunk], [[0.1, 0.2, 0.3]])
    retriever = RagRetriever.__new__(RagRetriever)
    retriever._config = RagConfig(enabled=True, embed=None)  # type: ignore[attr-defined]
    retriever._embedder = StubEmbedder()  # type: ignore[attr-defined]
    retriever._store = store  # type: ignore[attr-defined]

    tool = RagSearchTool(retriever)
    result = tool.execute({"query": "Tool"})
    assert "Tool context" in result.content


def test_agent_includes_rag_block() -> None:
    """Verify the agent prepends a RAG block when enabled."""

    store = StubRagStore()
    chunk = RagChunk(id="1", source="doc.txt", content="Injected context")
    store.upsert([chunk], [[0.1, 0.2, 0.3]])
    retriever = RagRetriever.__new__(RagRetriever)
    retriever._config = RagConfig(enabled=True, embed=None)  # type: ignore[attr-defined]
    retriever._embedder = StubEmbedder()  # type: ignore[attr-defined]
    retriever._store = store  # type: ignore[attr-defined]

    agent = DemoAgent(
        name="demo",
        llm=LLMClient.__new__(LLMClient),
        rag=retriever,
    )
    message = agent.build_effective_message("Hello")
    assert "[RAG]" in message
    assert "Injected context" in message
