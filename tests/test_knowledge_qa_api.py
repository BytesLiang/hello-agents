"""Tests for the knowledge QA FastAPI surface."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hello_agents.apps.knowledge_qa import (
    JsonKnowledgeBaseStore,
    JsonlRunTraceStore,
    KnowledgeQAConfig,
)
from hello_agents.apps.knowledge_qa.api import create_app
from hello_agents.apps.knowledge_qa.runtime import KnowledgeQARuntime
from hello_agents.llm.types import LLMMessage, LLMResponse
from hello_agents.rag.models import RagChunk


class FakeLLM:
    """Return deterministic completions for API tests."""

    def __init__(self, response: str) -> None:
        """Store the fixed response payload."""

        self._response = response
        self.calls: list[list[LLMMessage]] = []

    def chat(
        self,
        messages,
        *,
        temperature=None,
        max_tokens=None,
        tools=None,
    ) -> LLMResponse:
        """Capture one request and return a fixed chat response."""

        del temperature, max_tokens, tools
        self.calls.append(list(messages))
        return LLMResponse(
            model="fake-model",
            content=self._response,
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
        )


class StubRagRetriever:
    """Return deterministic retrieval hits for API tests."""

    def __init__(self, chunks: list[RagChunk]) -> None:
        """Store the chunk list returned by query()."""

        self._chunks = chunks

    def query(self, text: str, *, top_k: int | None = None) -> list[RagChunk]:
        """Return the configured retrieval chunks."""

        del text
        if top_k is None:
            return list(self._chunks)
        return list(self._chunks[:top_k])


class StubRagIndexer:
    """Return deterministic indexing counts for API tests."""

    def index_folder(self, path: Path, *, glob: str = "**/*") -> int:
        """Return a fixed chunk count for indexed inputs."""

        del glob
        return 2 if path.is_file() else 4


def build_runtime(
    tmp_path: Path,
    *,
    llm: FakeLLM | None = None,
    retriever: StubRagRetriever | None = None,
    indexer: StubRagIndexer | None = None,
) -> KnowledgeQARuntime:
    """Build a temporary runtime with stubbed dependencies."""

    config = KnowledgeQAConfig(
        knowledge_base_store_path=tmp_path / "knowledge_bases.json",
        trace_store_path=tmp_path / "traces.jsonl",
    )
    return KnowledgeQARuntime(
        config=config,
        knowledge_base_store=JsonKnowledgeBaseStore(config.knowledge_base_store_path),
        trace_store=JsonlRunTraceStore(config.trace_store_path),
        llm=llm,
        rag_retriever=retriever,
        rag_indexer=indexer,
    )


def test_list_knowledge_bases_returns_stored_items(tmp_path: Path) -> None:
    """Verify the list endpoint returns knowledge bases in storage."""

    document = tmp_path / "atlas.md"
    document.write_text("# Atlas\n\nOverview", encoding="utf-8")
    runtime = build_runtime(tmp_path, indexer=StubRagIndexer())
    runtime.build_ingest_service().ingest("Atlas Docs", [document])
    client = TestClient(create_app(runtime))

    response = client.get("/api/knowledge-bases")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["name"] == "Atlas Docs"


def test_create_knowledge_base_ingests_sources(tmp_path: Path) -> None:
    """Verify the create endpoint ingests a knowledge base synchronously."""

    document = tmp_path / "guide.md"
    document.write_text("# Guide\n\nAlpha", encoding="utf-8")
    runtime = build_runtime(tmp_path, indexer=StubRagIndexer())
    client = TestClient(create_app(runtime))

    response = client.post(
        "/api/knowledge-bases",
        json={
            "name": "Guide KB",
            "description": "CLI and API guide",
            "paths": [str(document)],
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["name"] == "Guide KB"
    assert payload["document_count"] == 1
    assert payload["chunk_count"] == 2


def test_upload_knowledge_base_ingests_uploaded_documents(tmp_path: Path) -> None:
    """Verify browser-style file uploads can create a knowledge base."""

    runtime = build_runtime(tmp_path, indexer=StubRagIndexer())
    client = TestClient(create_app(runtime))

    response = client.post(
        "/api/knowledge-bases/upload",
        data={"name": "Upload KB", "description": "Uploaded docs"},
        files=[
            ("files", ("guide.md", b"# Guide\n\nAlpha", "text/markdown")),
            ("files", ("faq.txt", b"Q: Hello\nA: World", "text/plain")),
        ],
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["name"] == "Upload KB"
    assert payload["document_count"] == 2
    assert payload["chunk_count"] == 4
    assert len(payload["source_paths"]) == 2
    assert all(".hello_agents/uploads/" in path for path in payload["source_paths"])


def test_get_unknown_knowledge_base_returns_404(tmp_path: Path) -> None:
    """Verify missing knowledge bases are reported as 404."""

    runtime = build_runtime(tmp_path)
    client = TestClient(create_app(runtime))

    response = client.get("/api/knowledge-bases/missing-kb")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown knowledge base: missing-kb"


def test_create_knowledge_base_rejects_empty_paths(tmp_path: Path) -> None:
    """Verify path validation maps to HTTP 400."""

    runtime = build_runtime(tmp_path, indexer=StubRagIndexer())
    client = TestClient(create_app(runtime))

    response = client.post(
        "/api/knowledge-bases",
        json={"name": "Empty KB", "description": "", "paths": ["   ", ""]},
    )

    assert response.status_code == 400
    assert "source path" in response.json()["detail"]


def test_upload_knowledge_base_rejects_missing_files(tmp_path: Path) -> None:
    """Verify upload requests need at least one file."""

    runtime = build_runtime(tmp_path, indexer=StubRagIndexer())
    client = TestClient(create_app(runtime))

    response = client.post(
        "/api/knowledge-bases/upload",
        data={"name": "Upload KB", "description": "Uploaded docs"},
    )

    assert response.status_code == 422


def test_ask_knowledge_base_returns_answer_and_trace(tmp_path: Path) -> None:
    """Verify the ask endpoint returns citations and a trace id."""

    document = tmp_path / "atlas.md"
    document.write_text("# Atlas\n\nUses Qdrant.", encoding="utf-8")
    runtime = build_runtime(
        tmp_path,
        llm=FakeLLM(
            '{"answer":"Atlas uses Qdrant for retrieval.",'
            '"answered":true,'
            '"reason":null,'
            '"citation_indices":[1]}'
        ),
        retriever=StubRagRetriever(
            [
                RagChunk(
                    id="chunk-1",
                    source=str(document),
                    content="Atlas uses Qdrant for retrieval.",
                    score=0.9,
                    metadata={"heading_path": "Architecture"},
                )
            ]
        ),
        indexer=StubRagIndexer(),
    )
    knowledge_base = runtime.build_ingest_service().ingest("Atlas Docs", [document])
    client = TestClient(create_app(runtime))

    response = client.post(
        f"/api/knowledge-bases/{knowledge_base.kb_id}/ask",
        json={"question": "What does Atlas use for retrieval?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answered"] is True
    assert payload["trace_id"]
    assert payload["citations"][0]["source"].endswith("atlas.md")


def test_ask_knowledge_base_rejects_empty_question(tmp_path: Path) -> None:
    """Verify empty questions map to HTTP 400."""

    document = tmp_path / "atlas.md"
    document.write_text("Atlas", encoding="utf-8")
    runtime = build_runtime(
        tmp_path,
        llm=FakeLLM("unused"),
        retriever=StubRagRetriever([]),
        indexer=StubRagIndexer(),
    )
    knowledge_base = runtime.build_ingest_service().ingest("Atlas Docs", [document])
    client = TestClient(create_app(runtime))

    response = client.post(
        f"/api/knowledge-bases/{knowledge_base.kb_id}/ask",
        json={"question": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question must not be empty."


def test_list_recent_traces_returns_recent_activity(tmp_path: Path) -> None:
    """Verify traces are exposed through the HTTP API."""

    document = tmp_path / "atlas.md"
    document.write_text("# Atlas\n\nUses Qdrant.", encoding="utf-8")
    runtime = build_runtime(
        tmp_path,
        llm=FakeLLM(
            '{"answer":"Atlas uses Qdrant for retrieval.",'
            '"answered":true,'
            '"reason":null,'
            '"citation_indices":[1]}'
        ),
        retriever=StubRagRetriever(
            [
                RagChunk(
                    id="chunk-1",
                    source=str(document),
                    content="Atlas uses Qdrant for retrieval.",
                    score=0.9,
                )
            ]
        ),
        indexer=StubRagIndexer(),
    )
    knowledge_base = runtime.build_ingest_service().ingest("Atlas Docs", [document])
    runtime.build_answer_service().ask(
        "What does Atlas use for retrieval?",
        kb_id=knowledge_base.kb_id,
    )
    client = TestClient(create_app(runtime))

    response = client.get("/api/traces?limit=5")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["question"] == "What does Atlas use for retrieval?"
    assert payload[0]["token_usage"]["total_tokens"] == 20
