"""Tests for the knowledge QA application layer."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from hello_agents.apps.knowledge_qa import (
    JsonKnowledgeBaseStore,
    JsonlRunTraceStore,
    KnowledgeQAConfig,
    KnowledgeQAService,
)
from hello_agents.apps.knowledge_qa.classifier import QuestionClassifier
from hello_agents.apps.knowledge_qa.retrieve import (
    DashScopeChunkReranker,
    KnowledgeRetriever,
)
from hello_agents.llm.types import LLMMessage, LLMResponse
from hello_agents.rag.models import RagChunk


class FakeLLM:
    """Return deterministic completions for knowledge QA tests."""

    def __init__(
        self,
        response: str | None = None,
        *,
        responses: list[str] | None = None,
    ) -> None:
        """Store optional staged responses and captured requests."""

        self._response = response
        self._responses = list(responses or [])
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
        content = self._next_response(messages)
        total_tokens = 18 if _is_answer_call(messages) else 0
        return LLMResponse(
            model="fake-model",
            content=content,
            prompt_tokens=11 if total_tokens else 0,
            completion_tokens=7 if total_tokens else 0,
            total_tokens=total_tokens,
        )

    def _next_response(self, messages: list[LLMMessage]) -> str:
        """Return the next staged or inferred response."""

        if self._responses:
            return self._responses.pop(0)

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
            return json.dumps(_assess_evidence(prompt))
        if "Validate whether the answer is fully supported" in prompt:
            return json.dumps(
                {
                    "is_valid": True,
                    "reason": None,
                    "citation_indices": _extract_citation_indices(prompt),
                    "answered": True,
                }
            )
        if self._response is not None:
            return self._response
        return json.dumps(
            {
                "answer": "I do not know based on the current knowledge base.",
                "answered": False,
                "reason": "insufficient_evidence",
                "citation_indices": [],
            }
        )


class StubRagRetriever:
    """Return deterministic retrieval hits for tests."""

    def __init__(self, chunks: list[RagChunk]) -> None:
        """Store the chunk list."""

        self._chunks = chunks
        self.top_k_calls: list[int | None] = []

    def query(
        self, text: str, *, top_k: int | None = None, kb_id: str | None = None
    ) -> list[RagChunk]:
        """Return the first top-k chunks."""

        del text, kb_id
        self.top_k_calls.append(top_k)
        if top_k is None:
            return list(self._chunks)
        return list(self._chunks[:top_k])


class StubRagIndexer:
    """Return deterministic indexing counts for tests."""

    def __init__(self) -> None:
        """Track deletion requests for assertions."""

        self.deleted_document_ids: list[tuple[str, str]] = []
        self.deleted_kb_ids: list[str] = []

    def index_file(self, path: Path, *, kb_id: str, document_id: str) -> int:
        """Return one chunk per provided file for simplicity."""

        del kb_id, document_id
        return 3 if path.is_file() else 5

    def delete_document(self, *, kb_id: str, document_id: str) -> None:
        """Accept document deletion requests during tests."""

        self.deleted_document_ids.append((kb_id, document_id))

    def delete_knowledge_base(self, *, kb_id: str) -> None:
        """Accept knowledge-base deletion requests during tests."""

        self.deleted_kb_ids.append(kb_id)


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
    assert len(knowledge_base.documents) == 1
    assert service.get_knowledge_base(knowledge_base.kb_id) is not None


def test_add_documents_updates_knowledge_base_metadata(tmp_path: Path) -> None:
    """Verify documents can be appended to an existing knowledge base."""

    first = tmp_path / "guide.md"
    second = tmp_path / "faq.md"
    first.write_text("# Guide\n\nAlpha", encoding="utf-8")
    second.write_text("# FAQ\n\nBeta", encoding="utf-8")
    service = build_service(tmp_path, indexer=StubRagIndexer())
    knowledge_base = service.ingest("Atlas Docs", [first], description="Atlas KB")

    updated = service.add_documents(knowledge_base.kb_id, [second])

    assert updated.document_count == 2
    assert updated.chunk_count == 6
    assert {document.name for document in updated.documents} == {"guide.md", "faq.md"}


def test_remove_document_updates_knowledge_base_metadata(tmp_path: Path) -> None:
    """Verify one managed document can be removed from a knowledge base."""

    first = tmp_path / "guide.md"
    second = tmp_path / "faq.md"
    first.write_text("# Guide\n\nAlpha", encoding="utf-8")
    second.write_text("# FAQ\n\nBeta", encoding="utf-8")
    service = build_service(tmp_path, indexer=StubRagIndexer())
    knowledge_base = service.ingest(
        "Atlas Docs",
        [first, second],
        description="Atlas KB",
    )

    updated = service.remove_document(
        knowledge_base.kb_id,
        knowledge_base.documents[0].document_id,
    )

    assert updated.document_count == 1
    assert updated.chunk_count == 3
    assert len(updated.documents) == 1


def test_delete_knowledge_base_removes_metadata_and_index(tmp_path: Path) -> None:
    """Verify deleting a knowledge base clears metadata and vector ownership."""

    document = tmp_path / "guide.md"
    document.write_text("# Guide\n\nAlpha", encoding="utf-8")
    indexer = StubRagIndexer()
    service = build_service(tmp_path, indexer=indexer)
    knowledge_base = service.ingest("Atlas Docs", [document], description="Atlas KB")

    service.delete_knowledge_base(knowledge_base.kb_id)

    assert service.get_knowledge_base(knowledge_base.kb_id) is None
    assert indexer.deleted_kb_ids == [knowledge_base.kb_id]


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
    service = build_service(
        tmp_path,
        llm=llm,
        retriever=retriever,
        indexer=StubRagIndexer(),
    )

    result = service.ask("What vector store does Atlas use?")

    assert result.answered is True
    assert result.trace_id is not None
    assert "Qdrant" in result.answer
    assert result.citations[0].source.endswith("atlas.md")
    traces = service.list_recent_traces(limit=5)
    assert len(traces) == 1
    assert traces[0].question == "What vector store does Atlas use?"
    assert traces[0].normalized_question == "What vector store does Atlas use?"
    assert traces[0].input_check is not None
    assert traces[0].citation_validation is not None
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
    assert llm.calls
    assert not any(_is_answer_call(call) for call in llm.calls)


def test_ask_uses_document_inspection_for_structure_questions(tmp_path: Path) -> None:
    """Verify structure questions can be answered from direct document analysis."""

    document = tmp_path / "01_overview.md"
    document.write_text(
        "# Overview\n\n"
        "## Product Summary\n\nAlpha\n\n"
        "## Goals\n\nBeta\n\n"
        "## Non-Goals\n\nGamma\n\n"
        "## Supported Content\n\nDelta\n\n"
        "## Demo Notes\n\nEpsilon\n",
        encoding="utf-8",
    )
    llm = FakeLLM(
        '{"answer":"01_overview.md has 5 sections.",'
        '"answered":true,'
        '"reason":null,'
        '"citation_indices":[]}'
    )
    retriever = StubRagRetriever(
        [
            RagChunk(
                id="chunk-1",
                source=str(document),
                content="Section: Supported Content\n\nDelta",
                score=0.3,
                metadata={"heading_path": "Supported Content"},
            )
        ]
    )
    service = build_service(
        tmp_path,
        llm=llm,
        retriever=retriever,
        indexer=StubRagIndexer(),
    )
    knowledge_base = service.ingest("Atlas Docs", [document], description="Atlas KB")

    result = service.ask("01_overview.md 分几段", kb_id=knowledge_base.kb_id)

    assert result.answered is True
    assert "5 sections" in result.answer
    assert result.citations == ()
    trace = service.list_recent_traces(limit=1)[0]
    assert trace.question_type == "document_structure"
    assert trace.inspection_result is not None
    assert trace.inspection_result["metadata"]["section_count"] == 5
    assert any("Document analysis" in call[-1].content for call in llm.calls)


def test_ask_uses_document_inspection_when_retrieval_misses_stat_question(
    tmp_path: Path,
) -> None:
    """Verify statistic questions can fall back to direct file inspection."""

    document = tmp_path / "01_overview.md"
    document.write_text("# Overview\n\nAlpha beta gamma.", encoding="utf-8")
    llm = FakeLLM(
        '{"answer":"01_overview.md contains 4 words.",'
        '"answered":true,'
        '"reason":null,'
        '"citation_indices":[]}'
    )
    retriever = StubRagRetriever([])
    service = build_service(
        tmp_path,
        llm=llm,
        retriever=retriever,
        indexer=StubRagIndexer(),
    )
    knowledge_base = service.ingest("Atlas Docs", [document], description="Atlas KB")

    result = service.ask("01_overview.md 有多少词", kb_id=knowledge_base.kb_id)

    assert result.answered is True
    assert "4 words" in result.answer
    trace = service.list_recent_traces(limit=1)[0]
    assert trace.question_type == "document_statistic"
    assert trace.inspection_result is not None
    assert trace.inspection_result["metadata"]["word_count"] == 4
    assert trace.retrieval_rounds


def test_classifier_extracts_ascii_filename_from_chinese_prefix() -> None:
    """Verify Chinese prose around a filename does not pollute target extraction."""

    classification = QuestionClassifier().classify("文档01_overview.md分几段")

    assert classification.question_type == "document_structure"
    assert classification.target_files == ("01_overview.md",)


def test_classifier_treats_singular_word_count_as_document_statistic() -> None:
    """Verify singular 'word' phrasing routes to document statistics."""

    classification = QuestionClassifier().classify("how many word 01_overview.md has")

    assert classification.question_type == "document_statistic"
    assert classification.target_files == ("01_overview.md",)


def test_classifier_drops_invented_target_files_from_llm_output() -> None:
    """Verify semantic classification cannot invent target files."""

    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "question_type": "fact_lookup",
                    "target_files": ["01_overview.md"],
                    "needs_document_inspection": False,
                    "needs_multi_step": False,
                    "reason": "This concept is likely defined in 01_overview.md.",
                }
            )
        ]
    )

    decision = QuestionClassifier(llm).classify_with_trace("skills是什么")

    assert decision.classification.question_type == "fact_lookup"
    assert decision.classification.target_files == ()
    assert decision.classification.reason == (
        "The question can be handled as grounded retrieval."
    )


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


def test_retriever_reranks_chunks_beyond_qdrant_score(tmp_path: Path) -> None:
    """Verify local reranking can promote a more relevant lower-score chunk."""

    atlas_path = tmp_path / "atlas.md"
    retriever = StubRagRetriever(
        [
            RagChunk(
                id="high-score-low-match",
                source=str(atlas_path),
                content="Atlas roadmap and release planning notes.",
                score=0.95,
                metadata={"heading_path": "Planning"},
            ),
            RagChunk(
                id="lower-score-better-match",
                source=str(atlas_path),
                content="Atlas uses Qdrant for retrieval and citations.",
                score=0.62,
                metadata={"heading_path": "Architecture"},
            ),
        ]
    )
    knowledge_retriever = KnowledgeRetriever(retriever, top_k=2)

    result = knowledge_retriever.retrieve("What does Atlas use for retrieval?")

    assert [chunk.chunk_id for chunk in result.chunks] == [
        "lower-score-better-match",
        "high-score-low-match",
    ]
    assert result.chunks[0].rerank_score is not None
    assert result.chunks[1].rerank_score is not None
    assert result.chunks[0].rerank_score > result.chunks[1].rerank_score


def test_retriever_can_use_dashscope_reranker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the retriever can use DashScope rerank scores."""

    atlas_path = tmp_path / "atlas.md"
    retriever = StubRagRetriever(
        [
            RagChunk(
                id="candidate-1",
                source=str(atlas_path),
                content="Atlas roadmap and planning notes.",
                score=0.95,
                metadata={"heading_path": "Planning"},
            ),
            RagChunk(
                id="candidate-2",
                source=str(atlas_path),
                content="Atlas uses Qdrant for retrieval and citations.",
                score=0.55,
                metadata={"heading_path": "Architecture"},
            ),
        ]
    )
    calls: list[dict[str, object]] = []

    class _FakeTextReRank:
        @staticmethod
        def call(**kwargs: object) -> object:
            calls.append(dict(kwargs))
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "output": {
                        "results": [
                            {"index": 1, "relevance_score": 0.98},
                            {"index": 0, "relevance_score": 0.2},
                        ]
                    },
                },
            )()

    fake_dashscope = type(
        "FakeDashScope",
        (),
        {"TextReRank": _FakeTextReRank, "api_key": None},
    )()
    monkeypatch.setattr(
        "hello_agents.apps.knowledge_qa.retrieve.dashscope",
        fake_dashscope,
    )
    knowledge_retriever = KnowledgeRetriever(
        retriever,
        top_k=2,
        reranker=DashScopeChunkReranker(api_key="test-key"),
    )

    result = knowledge_retriever.retrieve("What does Atlas use for retrieval?")

    assert [chunk.chunk_id for chunk in result.chunks] == [
        "candidate-1",
        "candidate-2",
    ]
    assert result.chunks[0].rerank_score == 0.98
    assert result.chunks[1].rerank_score == 0.2
    assert calls[0]["model"] == "qwen3-rerank"


def test_ask_raises_for_unknown_knowledge_base(tmp_path: Path) -> None:
    """Verify unknown knowledge base identifiers are rejected explicitly."""

    llm = FakeLLM("Unused")
    retriever = StubRagRetriever([])
    service = build_service(tmp_path, llm=llm, retriever=retriever)

    with pytest.raises(ValueError, match="Unknown knowledge base"):
        service.ask("Where is the deployment guide?", kb_id="missing-kb")


def test_input_check_normalizes_pasted_trace_fragments(tmp_path: Path) -> None:
    """Verify pasted trace fields are removed before planning and answering."""

    document = tmp_path / "01_overview.md"
    document.write_text("# Overview\n\nAlpha beta gamma.", encoding="utf-8")
    llm = FakeLLM(
        '{"answer":"01_overview.md contains 4 words.",'
        '"answered":true,'
        '"reason":null,'
        '"citation_indices":[]}'
    )
    retriever = StubRagRetriever([])
    service = build_service(
        tmp_path,
        llm=llm,
        retriever=retriever,
        indexer=StubRagIndexer(),
    )
    knowledge_base = service.ingest("Atlas Docs", [document], description="Atlas KB")

    result = service.ask(
        'how many word 01_overview.md has", "rewritten_query": '
        '"how many word 01_overview.md has',
        kb_id=knowledge_base.kb_id,
    )

    assert result.answered is True
    trace = service.list_recent_traces(limit=1)[0]
    assert trace.normalized_question == "how many word 01_overview.md has"
    assert trace.input_check is not None
    assert trace.input_check["issues"] == [
        "Removed pasted trace metadata from the question."
    ]


def _extract_between(prompt: str, marker: str) -> str:
    """Extract the text after a marker up to the next blank line."""

    if marker not in prompt:
        return ""
    section = prompt.split(marker, maxsplit=1)[1]
    return section.split("\n\n", maxsplit=1)[0].strip()


def _classify_question(question: str) -> dict[str, object]:
    """Return a simple semantic classification for the fake LLM."""

    normalized = question.strip()
    target_files = tuple(
        match.lower()
        for match in re.findall(r"([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,16})", normalized)
    )
    lowered = normalized.lower()
    if any(token in lowered for token in ("多少词", "词数", "word count", "many word")):
        return {
            "question_type": "document_statistic",
            "target_files": list(target_files),
            "needs_document_inspection": True,
            "needs_multi_step": True,
            "reason": "The question asks for document statistics.",
        }
    if any(token in lowered for token in ("分几段", "几段", "how many sections")):
        return {
            "question_type": "document_structure",
            "target_files": list(target_files),
            "needs_document_inspection": True,
            "needs_multi_step": True,
            "reason": "The question asks about document structure.",
        }
    return {
        "question_type": "fact_lookup",
        "target_files": list(target_files),
        "needs_document_inspection": False,
        "needs_multi_step": bool(target_files),
        "reason": "The question can be handled as grounded retrieval.",
    }


def _plan_question(question: str) -> dict[str, object]:
    """Return a simple retrieval plan for the fake LLM."""

    normalized = question.strip()
    target_files = re.findall(r"([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,16})", normalized)
    target_file = target_files[0] if target_files else ""
    lowered = normalized.lower()
    if any(token in lowered for token in ("分几段", "几段", "how many sections")):
        return {
            "primary_queries": [normalized, target_file],
            "fallback_queries": [
                f"sections in {target_file}" if target_file else "",
                f"headings in {target_file}" if target_file else "",
            ],
            "max_rounds": 2,
            "use_document_inspection_on_failure": True,
            "summary": "Locate the target document, then inspect its structure.",
        }
    if any(token in lowered for token in ("多少词", "词数", "word count", "many word")):
        return {
            "primary_queries": [target_file, normalized],
            "fallback_queries": [f"word count {target_file}" if target_file else ""],
            "max_rounds": 2,
            "use_document_inspection_on_failure": True,
            "summary": "Locate the target document, then inspect it for statistics.",
        }
    return {
        "primary_queries": [normalized],
        "fallback_queries": [],
        "max_rounds": 1,
        "use_document_inspection_on_failure": False,
        "summary": "Use direct retrieval for grounded answer generation.",
    }


def _assess_evidence(prompt: str) -> dict[str, object]:
    """Return a simple evidence assessment for the fake LLM."""

    question = _extract_between(prompt, "Question:\n")
    lowered = question.lower()
    round_match = re.search(r"Round:\s*(\d+)\s*/\s*(\d+)", prompt)
    round_index = int(round_match.group(1)) if round_match else 1
    max_rounds = int(round_match.group(2)) if round_match else 1
    has_chunks = "No chunks retrieved." not in prompt
    matched_target = "01_overview.md" in prompt

    if any(
        token in lowered
        for token in (
            "分几段",
            "几段",
            "how many sections",
            "多少词",
            "word count",
            "many word",
        )
    ):
        if matched_target:
            return {
                "status": "needs_document_inspection",
                "score": 0.8,
                "reason": (
                    "Target document was located and can now be inspected directly."
                ),
                "failure_mode": "needs_document_inspection",
                "rewritten_query": "",
            }
        status = "needs_rewrite" if round_index < max_rounds else "insufficient"
        return {
            "status": status,
            "score": 0.1,
            "reason": "Retrieved evidence does not yet include the target document.",
            "failure_mode": "wrong_document",
            "rewritten_query": "01_overview.md",
        }
    if has_chunks:
        return {
            "status": "sufficient",
            "score": 0.8,
            "reason": "Retrieved chunks are sufficient for direct answer generation.",
            "failure_mode": None,
            "rewritten_query": "",
        }
    status = "needs_rewrite" if round_index < max_rounds else "insufficient"
    return {
        "status": status,
        "score": 0.0,
        "reason": "No retrieval hits were found for the current query.",
        "failure_mode": "no_hits",
        "rewritten_query": "",
    }


def _extract_citation_indices(prompt: str) -> list[int]:
    """Extract citation indices from the validation prompt."""

    match = re.search(r"Citation indices from the answer:\n(\[[^\n]*\])", prompt)
    if match is None:
        return []
    parsed = json.loads(match.group(1))
    return parsed if isinstance(parsed, list) else []


def _is_answer_call(messages: list[LLMMessage]) -> bool:
    """Return whether the current prompt is the final answer generation step."""

    return "Answer the question using the context below." in messages[-1].content
