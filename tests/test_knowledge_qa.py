"""Tests for the knowledge QA application layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from hello_agents.apps.knowledge_qa import (
    JsonKnowledgeBaseStore,
    JsonlRunTraceStore,
    KnowledgeQAConfig,
    KnowledgeQAService,
)
from hello_agents.apps.knowledge_qa.retrieve import KnowledgeRetriever
from hello_agents.llm.types import LLMMessage, LLMResponse
from hello_agents.rag.models import RagChunk


class FakeLLM:
    """Return deterministic completions for knowledge QA tests."""

    def __init__(self, response: str) -> None:
        """Store the response and captured requests."""

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
        """Capture the request and return a fixed answer."""

        del temperature, max_tokens, tools
        self.calls.append(list(messages))
        return LLMResponse(
            model="fake-model",
            content=self._response,
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        )


class StubRagRetriever:
    """Return deterministic retrieval hits for tests."""

    def __init__(self, chunks: list[RagChunk]) -> None:
        """Store the chunk list."""

        self._chunks = chunks
        self.top_k_calls: list[int | None] = []

    def query(self, text: str, *, top_k: int | None = None) -> list[RagChunk]:
        """Return the first top-k chunks."""

        del text
        self.top_k_calls.append(top_k)
        if top_k is None:
            return list(self._chunks)
        return list(self._chunks[:top_k])


class StubRagIndexer:
    """Return deterministic indexing counts for tests."""

    def index_folder(self, path: Path, *, glob: str = "**/*") -> int:
        """Return one chunk per provided file for simplicity."""

        del glob
        return 3 if path.is_file() else 5


def build_service(
    tmp_path: Path,
    *,
    llm: FakeLLM | None = None,
    retriever: StubRagRetriever | None = None,
    indexer: StubRagIndexer | None = None,
) -> KnowledgeQAService:
    """Build a test service with local metadata and trace stores."""

    config = KnowledgeQAConfig(
        knowledge_base_store_path=tmp_path / "knowledge_bases.json",
        trace_store_path=tmp_path / "traces.jsonl",
    )
    return KnowledgeQAService(
        config=config,
        llm=llm,
        rag_retriever=retriever,
        rag_indexer=indexer,
        knowledge_base_store=JsonKnowledgeBaseStore(config.knowledge_base_store_path),
        trace_store=JsonlRunTraceStore(config.trace_store_path),
    )


def test_ingest_persists_knowledge_base_metadata(tmp_path: Path) -> None:
    """Verify ingestion stores indexed knowledge base metadata."""

    document = tmp_path / "guide.md"
    document.write_text("# Guide\n\nAlpha", encoding="utf-8")
    service = build_service(tmp_path, indexer=StubRagIndexer())

    knowledge_base = service.ingest("Atlas Docs", [document], description="Atlas KB")

    assert knowledge_base.name == "Atlas Docs"
    assert knowledge_base.description == "Atlas KB"
    assert knowledge_base.document_count == 1
    assert knowledge_base.chunk_count == 3
    assert service.get_knowledge_base(knowledge_base.kb_id) is not None


def test_ask_returns_answer_and_citations(tmp_path: Path) -> None:
    """Verify the service answers questions and records a trace."""

    llm = FakeLLM(
        '{"answer":"Atlas uses Qdrant for retrieval.",'
        '"answered":true,'
        '"reason":null,'
        '"citation_indices":[1]}'
    )
    retriever = StubRagRetriever(
        [
            RagChunk(
                id="chunk-1",
                source=str(tmp_path / "atlas.md"),
                content="Atlas uses Qdrant for retrieval.",
                score=0.8,
                metadata={"heading_path": "Architecture"},
            )
        ]
    )
    service = build_service(tmp_path, llm=llm, retriever=retriever)

    result = service.ask("What vector store does Atlas use?")

    assert result.answered is True
    assert result.trace_id is not None
    assert "Qdrant" in result.answer
    assert result.citations[0].source.endswith("atlas.md")
    traces = service.list_recent_traces(limit=5)
    assert len(traces) == 1
    assert traces[0].question == "What vector store does Atlas use?"
    assert traces[0].token_usage.total_tokens == 18


def test_ask_falls_back_to_plain_text_when_json_parse_fails(tmp_path: Path) -> None:
    """Verify raw text answers still work when structured parsing fails."""

    llm = FakeLLM("Atlas uses SQLite for long-term memory.")
    retriever = StubRagRetriever(
        [
            RagChunk(
                id="chunk-1",
                source=str(tmp_path / "atlas.md"),
                content="Atlas uses SQLite for long-term memory.",
                score=0.7,
            )
        ]
    )
    service = build_service(tmp_path, llm=llm, retriever=retriever)

    result = service.ask("What store is used for long-term memory?")

    assert result.answered is True
    assert result.answer == "Atlas uses SQLite for long-term memory."
    assert len(result.citations) == 1


def test_ask_uses_structured_refusal_and_drops_default_citations(
    tmp_path: Path,
) -> None:
    """Verify structured refusals return no citations by default."""

    llm = FakeLLM(
        '{"answer":"I do not know based on the current knowledge base.",'
        '"answered":false,'
        '"reason":"insufficient_evidence",'
        '"citation_indices":[]}'
    )
    retriever = StubRagRetriever(
        [
            RagChunk(
                id="chunk-1",
                source=str(tmp_path / "atlas.md"),
                content="This file does not contain the answer.",
                score=0.2,
            )
        ]
    )
    service = build_service(tmp_path, llm=llm, retriever=retriever)

    result = service.ask("What is the exact GA date?")

    assert result.answered is False
    assert result.reason == "insufficient_evidence"
    assert result.citations == ()


def test_ask_rejects_when_no_chunks_are_found(tmp_path: Path) -> None:
    """Verify the service refuses to answer without retrieval evidence."""

    llm = FakeLLM("This should not be used.")
    retriever = StubRagRetriever([])
    service = build_service(tmp_path, llm=llm, retriever=retriever)

    result = service.ask("What is the release date?")

    assert result.answered is False
    assert result.reason == "no_relevant_context"
    assert "do not know" in result.answer
    assert not llm.calls


def test_retriever_deduplicates_and_prioritizes_named_file_chunks(
    tmp_path: Path,
) -> None:
    """Verify file-specific questions prefer matching sources and drop duplicates."""

    readme_path = tmp_path / "README.md"
    overview_path = tmp_path / "01_overview.md"
    retriever = StubRagRetriever(
        [
            RagChunk(
                id="dup-1",
                source=str(readme_path),
                content="Section: Files\n\n- `01_overview.md`: product scope and goals",
                score=1.0,
                metadata={"heading_path": "Files"},
            ),
            RagChunk(
                id="dup-2",
                source=str(readme_path),
                content="Section: Files\n\n- `01_overview.md`: product scope and goals",
                score=0.9,
                metadata={"heading_path": "Files"},
            ),
            RagChunk(
                id="overview-1",
                source=str(overview_path),
                content=(
                    "Section: Product Summary\n\n"
                    "Atlas Assistant is an internal QA product."
                ),
                score=0.3,
                metadata={"heading_path": "Product Summary"},
            ),
            RagChunk(
                id="overview-2",
                source=str(overview_path),
                content="Section: Goals\n\n- Provide grounded answers with citations.",
                score=0.2,
                metadata={"heading_path": "Goals"},
            ),
            RagChunk(
                id="overview-3",
                source=str(overview_path),
                content=(
                    "Section: Non-Goals\n\n"
                    "- Atlas Assistant does not browse the public internet."
                ),
                score=0.1,
                metadata={"heading_path": "Non-Goals"},
            ),
        ]
    )
    knowledge_retriever = KnowledgeRetriever(retriever, top_k=4)

    result = knowledge_retriever.retrieve("01_overview.md 分几段")

    assert retriever.top_k_calls == [12]
    assert [Path(chunk.source).name for chunk in result.chunks[:3]] == [
        "01_overview.md",
        "01_overview.md",
        "01_overview.md",
    ]
    assert len(result.chunks) == 4
    assert (
        sum(1 for chunk in result.chunks if Path(chunk.source).name == "README.md") == 1
    )


def test_ask_raises_for_unknown_knowledge_base(tmp_path: Path) -> None:
    """Verify unknown knowledge base identifiers are rejected explicitly."""

    llm = FakeLLM("Unused")
    retriever = StubRagRetriever([])
    service = build_service(tmp_path, llm=llm, retriever=retriever)

    with pytest.raises(ValueError, match="Unknown knowledge base"):
        service.ask("Where is the deployment guide?", kb_id="missing-kb")
