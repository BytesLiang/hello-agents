"""Tests for the lightweight RAG subsystem."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

from hello_agents.agent import Agent
from hello_agents.llm.client import LLMClient
from hello_agents.rag import qdrant_store as qdrant_store_module
from hello_agents.rag.config import RagConfig
from hello_agents.rag.indexer import RagIndexer
from hello_agents.rag.models import RagChunk
from hello_agents.rag.qdrant_store import (
    RagQdrantStore,
    _collection_schema_mismatch_reason,
    _iter_batches,
    _text_to_sparse_vector,
)
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

    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        kb_id: str | None = None,
    ) -> list[RagChunk]:
        """Return the first top-k chunks for simplicity."""

        del embedding, kb_id
        return list(self._chunks[:top_k])

    def search_hybrid(
        self,
        text: str,
        embedding: Sequence[float],
        *,
        top_k: int,
        kb_id: str | None = None,
    ) -> list[RagChunk]:
        """Return the first top-k chunks for hybrid retrieval tests."""

        del text, embedding, kb_id
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


def test_sparse_vector_generation_is_stable() -> None:
    """Verify sparse vectors are non-empty and deterministic."""

    first = _text_to_sparse_vector("Python agent framework")
    second = _text_to_sparse_vector("Python agent framework")

    assert first.indices
    assert first.indices == second.indices
    assert first.values == second.values


def test_collection_schema_mismatch_reason_detects_legacy_collection() -> None:
    """Verify legacy unnamed-vector collections are treated as incompatible."""

    info = {
        "config": {
            "params": {
                "vectors": {"size": 3, "distance": "Cosine"},
                "sparse_vectors": {},
            }
        },
        "points_count": 2,
    }

    reason = _collection_schema_mismatch_reason(
        info,
        dense_vector_name="dense",
        sparse_vector_name="sparse",
        vector_size=3,
    )

    assert reason == "missing dense vector 'dense'"


def test_ensure_collection_raises_for_incompatible_nonempty_collection() -> None:
    """Verify incompatible collections fail with a clear error by default."""

    class StubClient:
        """Provide minimal collection APIs used by the store."""

        def collection_exists(self, name: str) -> bool:
            del name
            return True

        def get_collection(self, name: str) -> dict[str, object]:
            del name
            return {
                "config": {
                    "params": {
                        "vectors": {"size": 3, "distance": "Cosine"},
                        "sparse_vectors": {},
                    }
                },
                "points_count": 4,
            }

    store = RagQdrantStore.__new__(RagQdrantStore)
    store._config = RagConfig(  # type: ignore[attr-defined]
        enabled=True,
        qdrant_url="http://localhost:6333",
        recreate_collection_on_schema_mismatch=False,
        embed=None,
    )
    store._client = StubClient()  # type: ignore[attr-defined]
    store._vector_size = None  # type: ignore[attr-defined]

    try:
        store._ensure_collection(vector_size=3)  # type: ignore[attr-defined]
    except RuntimeError as exc:
        assert "RAG_RECREATE_COLLECTION_ON_SCHEMA_MISMATCH=true" in str(exc)
    else:
        raise AssertionError("Expected an incompatible collection error.")


def test_ensure_collection_recreates_empty_incompatible_collection() -> None:
    """Verify empty incompatible collections are recreated automatically."""

    events: list[str] = []

    class StubClient:
        """Provide minimal collection APIs used by the store."""

        def collection_exists(self, name: str) -> bool:
            del name
            return True

        def get_collection(self, name: str) -> dict[str, object]:
            del name
            return {
                "config": {
                    "params": {
                        "vectors": {"size": 3, "distance": "Cosine"},
                        "sparse_vectors": {},
                    }
                },
                "points_count": 0,
            }

        def delete_collection(self, name: str) -> None:
            events.append(f"delete:{name}")

    store = RagQdrantStore.__new__(RagQdrantStore)
    store._config = RagConfig(  # type: ignore[attr-defined]
        enabled=True,
        qdrant_url="http://localhost:6333",
        collection="hello_agents_rag",
        recreate_collection_on_schema_mismatch=False,
        embed=None,
    )
    store._client = StubClient()  # type: ignore[attr-defined]
    store._vector_size = None  # type: ignore[attr-defined]
    store._create_collection = (  # type: ignore[attr-defined]
        lambda *, vector_size: events.append(f"create:{vector_size}")
    )

    store._ensure_collection(vector_size=3)  # type: ignore[attr-defined]

    assert events == ["delete:hello_agents_rag", "create:3"]


def test_iter_batches_splits_chunks_and_embeddings_consistently() -> None:
    """Verify upsert batching preserves chunk-embedding alignment."""

    chunks = [
        RagChunk(id="1", source="a.md", content="A"),
        RagChunk(id="2", source="b.md", content="B"),
        RagChunk(id="3", source="c.md", content="C"),
    ]
    embeddings = [[0.1], [0.2], [0.3]]

    batches = _iter_batches(chunks, embeddings, batch_size=2)

    assert len(batches) == 2
    assert [chunk.id for chunk in batches[0][0]] == ["1", "2"]
    assert batches[0][1] == [[0.1], [0.2]]
    assert [chunk.id for chunk in batches[1][0]] == ["3"]
    assert batches[1][1] == [[0.3]]


def test_upsert_uses_configured_batch_size() -> None:
    """Verify Qdrant upsert sends multiple requests when batching is enabled."""

    calls: list[int] = []

    original_point_struct = qdrant_store_module.models.__dict__.get("PointStruct")
    qdrant_store_module.models.PointStruct = lambda **kwargs: kwargs  # type: ignore[attr-defined]

    class StubClient:
        """Record upsert batch sizes."""

        def upsert(
            self,
            *,
            collection_name: str,
            points: Sequence[object],
            wait: bool,
            timeout: int,
        ) -> None:
            del collection_name, wait, timeout
            calls.append(len(points))

    store = RagQdrantStore.__new__(RagQdrantStore)
    store._config = RagConfig(  # type: ignore[attr-defined]
        enabled=True,
        qdrant_url="http://localhost:6333",
        qdrant_upsert_batch_size=2,
        embed=None,
    )
    store._client = StubClient()  # type: ignore[attr-defined]
    store._vector_size = 3  # type: ignore[attr-defined]

    chunks = [
        RagChunk(id="1", source="a.md", content="A"),
        RagChunk(id="2", source="b.md", content="B"),
        RagChunk(id="3", source="c.md", content="C"),
    ]
    embeddings = [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]

    try:
        store.upsert(chunks, embeddings)
    finally:
        if original_point_struct is None:
            delattr(qdrant_store_module.models, "PointStruct")
        else:
            qdrant_store_module.models.PointStruct = original_point_struct  # type: ignore[attr-defined]

    assert calls == [2, 1]


def test_upsert_logs_underlying_qdrant_failure() -> None:
    """Verify Qdrant upsert errors are logged with contextual details."""

    original_point_struct = qdrant_store_module.models.__dict__.get("PointStruct")
    qdrant_store_module.models.PointStruct = lambda **kwargs: kwargs  # type: ignore[attr-defined]

    class StubClient:
        """Raise a deterministic Qdrant failure."""

        def upsert(
            self,
            *,
            collection_name: str,
            points: Sequence[object],
            wait: bool,
            timeout: int,
        ) -> None:
            del collection_name, points, wait, timeout
            raise ValueError("socket timed out")

    store = RagQdrantStore.__new__(RagQdrantStore)
    store._config = RagConfig(  # type: ignore[attr-defined]
        enabled=True,
        qdrant_url="http://localhost:6333",
        qdrant_upsert_batch_size=64,
        embed=None,
    )
    store._client = StubClient()  # type: ignore[attr-defined]
    store._vector_size = 3  # type: ignore[attr-defined]

    chunks = [RagChunk(id="1", source="a.md", content="A")]
    embeddings = [[0.1, 0.0, 0.0]]
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.ERROR)
    logger = logging.getLogger("hello_agents")
    logger.addHandler(handler)

    try:
        try:
            store.upsert(chunks, embeddings)
        except RuntimeError:
            pass
    finally:
        logger.removeHandler(handler)
        if original_point_struct is None:
            delattr(qdrant_store_module.models, "PointStruct")
        else:
            qdrant_store_module.models.PointStruct = original_point_struct  # type: ignore[attr-defined]

    assert "Qdrant upsert failed." in stream.getvalue()
    assert "error_type=ValueError" in stream.getvalue()
    assert "Traceback" in stream.getvalue()
