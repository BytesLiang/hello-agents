"""Tests for the knowledge QA FastAPI surface."""

from __future__ import annotations

import json
import logging
import re
from io import StringIO
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
        total_tokens = 20 if _is_answer_call(messages) else 0
        return LLMResponse(
            model="fake-model",
            content=self._next_response(messages),
            prompt_tokens=12 if total_tokens else 0,
            completion_tokens=8 if total_tokens else 0,
            total_tokens=total_tokens,
        )

    def _next_response(self, messages: list[LLMMessage]) -> str:
        """Return the next inferred response for the agent stage."""

        prompt = messages[-1].content
        if "Normalize the question below" in prompt:
            normalized = _extract_between(
                prompt,
                "Heuristic normalization candidate:\n",
            ) or _extract_between(prompt, "Original question:\n")
            return json.dumps(
                {
                    "normalized_question": normalized.strip(),
                    "is_valid": True,
                    "issues": [],
                    "reason": None,
                }
            )
        if "Classify the question for a knowledge QA agent." in prompt:
            question = _extract_between(prompt, "Question:\n")
            return json.dumps(_classify_question(question))
        if "Build a retrieval plan for the question below." in prompt:
            question = _extract_between(prompt, "Question:\n")
            return json.dumps(_plan_question(question))
        if "Assess the evidence for the question below." in prompt:
            return json.dumps(
                {
                    "status": "sufficient",
                    "score": 0.9,
                    "reason": (
                        "Retrieved chunks are sufficient for direct answer generation."
                    ),
                    "failure_mode": None,
                    "rewritten_query": "",
                }
            )
        if "Validate whether the answer is fully supported" in prompt:
            return json.dumps(
                {
                    "is_valid": True,
                    "reason": None,
                    "citation_indices": [1],
                    "answered": True,
                }
            )
        return self._response


class StubRagRetriever:
    """Return deterministic retrieval hits for API tests."""

    def __init__(self, chunks: list[RagChunk]) -> None:
        """Store the chunk list returned by query()."""

        self._chunks = chunks

    def query(
        self,
        text: str,
        *,
        top_k: int | None = None,
        kb_id: str | None = None,
    ) -> list[RagChunk]:
        """Return the configured retrieval chunks."""

        del text, kb_id
        if top_k is None:
            return list(self._chunks)
        return list(self._chunks[:top_k])


class StubRagIndexer:
    """Return deterministic indexing counts for API tests."""

    def __init__(self) -> None:
        """Track deletion requests for assertions."""

        self.deleted_document_ids: list[tuple[str, str]] = []
        self.deleted_kb_ids: list[str] = []

    def index_file(self, path: Path, *, kb_id: str, document_id: str) -> int:
        """Return a fixed chunk count for indexed inputs."""

        del kb_id, document_id
        return 2 if path.is_file() else 4

    def delete_document(self, *, kb_id: str, document_id: str) -> None:
        """Accept document deletion requests during tests."""

        self.deleted_document_ids.append((kb_id, document_id))

    def delete_knowledge_base(self, *, kb_id: str) -> None:
        """Accept knowledge-base deletion requests during tests."""

        self.deleted_kb_ids.append(kb_id)


class FailingRagIndexer:
    """Raise a deterministic indexing failure for API logging tests."""

    def index_file(self, path: Path, *, kb_id: str, document_id: str) -> int:
        """Raise one runtime error when indexing is attempted."""

        del path, kb_id, document_id
        raise RuntimeError("boom")

    def delete_document(self, *, kb_id: str, document_id: str) -> None:
        """Accept document deletion requests during tests."""

        del kb_id, document_id

    def delete_knowledge_base(self, *, kb_id: str) -> None:
        """Accept knowledge-base deletion requests during tests."""

        del kb_id


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
    assert len(payload["documents"]) == 1
    assert payload["documents"][0]["name"] == "guide.md"


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
    assert len(payload["documents"]) == 2
    assert len(payload["source_paths"]) == 2
    assert all(".hello_agents/uploads/" in path for path in payload["source_paths"])


def test_add_documents_endpoint_updates_existing_knowledge_base(tmp_path: Path) -> None:
    """Verify an existing knowledge base can ingest additional documents."""

    first = tmp_path / "guide.md"
    second = tmp_path / "faq.md"
    first.write_text("# Guide\n\nAlpha", encoding="utf-8")
    second.write_text("# FAQ\n\nBeta", encoding="utf-8")
    runtime = build_runtime(tmp_path, indexer=StubRagIndexer())
    service = runtime.build_ingest_service()
    knowledge_base = service.ingest("Guide KB", [first])
    client = TestClient(create_app(runtime))

    response = client.post(
        f"/api/knowledge-bases/{knowledge_base.kb_id}/documents",
        json={"paths": [str(second)]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_count"] == 2
    assert {document["name"] for document in payload["documents"]} == {
        "guide.md",
        "faq.md",
    }


def test_delete_document_endpoint_removes_one_document(tmp_path: Path) -> None:
    """Verify one managed document can be removed from a knowledge base."""

    first = tmp_path / "guide.md"
    second = tmp_path / "faq.md"
    first.write_text("# Guide\n\nAlpha", encoding="utf-8")
    second.write_text("# FAQ\n\nBeta", encoding="utf-8")
    runtime = build_runtime(tmp_path, indexer=StubRagIndexer())
    service = runtime.build_ingest_service()
    knowledge_base = service.ingest("Guide KB", [first, second])
    client = TestClient(create_app(runtime))
    document_id = knowledge_base.documents[0].document_id

    response = client.delete(
        f"/api/knowledge-bases/{knowledge_base.kb_id}/documents/{document_id}"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_count"] == 1
    assert len(payload["documents"]) == 1
    assert payload["documents"][0]["document_id"] != document_id


def test_delete_knowledge_base_endpoint_removes_one_knowledge_base(
    tmp_path: Path,
) -> None:
    """Verify the delete endpoint removes metadata and returns 204."""

    document = tmp_path / "guide.md"
    document.write_text("# Guide\n\nAlpha", encoding="utf-8")
    indexer = StubRagIndexer()
    runtime = build_runtime(tmp_path, indexer=indexer)
    knowledge_base = runtime.build_ingest_service().ingest("Guide KB", [document])
    client = TestClient(create_app(runtime))

    response = client.delete(f"/api/knowledge-bases/{knowledge_base.kb_id}")

    assert response.status_code == 204
    assert runtime.build_read_service().get_knowledge_base(knowledge_base.kb_id) is None
    assert indexer.deleted_kb_ids == [knowledge_base.kb_id]


def test_create_knowledge_base_logs_unhandled_exceptions(
    tmp_path: Path,
) -> None:
    """Verify create failures are emitted through the application logger."""

    document = tmp_path / "guide.md"
    document.write_text("# Guide\n\nAlpha", encoding="utf-8")
    runtime = build_runtime(tmp_path, indexer=FailingRagIndexer())
    client = TestClient(create_app(runtime))
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("hello_agents")
    logger.addHandler(handler)

    try:
        response = client.post(
            "/api/knowledge-bases",
            json={
                "name": "Guide KB",
                "description": "CLI and API guide",
                "paths": [str(document)],
            },
        )
    finally:
        logger.removeHandler(handler)

    assert response.status_code == 500
    assert response.json()["detail"] == "boom"
    assert "Create knowledge base failed." in stream.getvalue()
    assert "Traceback" in stream.getvalue()


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
    assert payload[0]["normalized_question"] == "What does Atlas use for retrieval?"
    assert payload[0]["input_check"]["is_valid"] is True
    assert payload[0]["citation_validation"]["is_valid"] is True
    assert payload[0]["retrieved_chunks"][0]["rerank_score"] is not None
    assert payload[0]["token_usage"]["total_tokens"] == 20


def _extract_between(prompt: str, marker: str) -> str:
    """Extract the text after a marker up to the next blank line."""

    if marker not in prompt:
        return ""
    section = prompt.split(marker, maxsplit=1)[1]
    return section.split("\n\n", maxsplit=1)[0].strip()


def _classify_question(question: str) -> dict[str, object]:
    """Return a simple semantic classification for API tests."""

    normalized = question.strip()
    target_files = tuple(
        match.lower()
        for match in re.findall(r"([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,16})", normalized)
    )
    return {
        "question_type": "fact_lookup",
        "target_files": list(target_files),
        "needs_document_inspection": False,
        "needs_multi_step": bool(target_files),
        "reason": "The question can be handled as grounded retrieval.",
    }


def _plan_question(question: str) -> dict[str, object]:
    """Return a simple retrieval plan for API tests."""

    return {
        "primary_queries": [question.strip()],
        "fallback_queries": [],
        "max_rounds": 1,
        "use_document_inspection_on_failure": False,
        "summary": "Use direct retrieval for grounded answer generation.",
    }


def _is_answer_call(messages: list[LLMMessage]) -> bool:
    """Return whether the current prompt is the final answer generation step."""

    return "Answer the question using the context below." in messages[-1].content
