"""Service layer for the knowledge QA application."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from hello_agents.apps.knowledge_qa.answer import (
    build_answer_messages,
    build_citations,
    parse_answer_response,
)
from hello_agents.apps.knowledge_qa.config import KnowledgeQAConfig
from hello_agents.apps.knowledge_qa.ingest import (
    KnowledgeBaseIngester,
    SupportsIndexFolder,
)
from hello_agents.apps.knowledge_qa.models import (
    AnswerResult,
    KnowledgeBase,
    KnowledgeBaseStatus,
    RetrievedChunk,
    RunTrace,
    TokenUsage,
)
from hello_agents.apps.knowledge_qa.retrieve import (
    KnowledgeRetriever,
    QueryRewriter,
    SupportsRagQuery,
)
from hello_agents.apps.knowledge_qa.store import JsonKnowledgeBaseStore
from hello_agents.apps.knowledge_qa.trace import JsonlRunTraceStore
from hello_agents.llm.types import LLMMessage, LLMResponse


class SupportsChat(Protocol):
    """Represent the LLM interface required by the app layer."""

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> LLMResponse:
        """Return one chat completion."""


class KnowledgeQAService:
    """Coordinate indexing, retrieval, answering, and trace persistence."""

    def __init__(
        self,
        *,
        config: KnowledgeQAConfig | None = None,
        llm: SupportsChat | None = None,
        rag_retriever: SupportsRagQuery | None = None,
        rag_indexer: SupportsIndexFolder | None = None,
        knowledge_base_store: JsonKnowledgeBaseStore | None = None,
        trace_store: JsonlRunTraceStore | None = None,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        """Store the runtime dependencies for the knowledge QA application."""

        self.config = config or KnowledgeQAConfig()
        self._llm = llm
        self._knowledge_base_store = knowledge_base_store or JsonKnowledgeBaseStore(
            self.config.knowledge_base_store_path
        )
        self._trace_store = trace_store or JsonlRunTraceStore(
            self.config.trace_store_path
        )
        self._ingester = (
            KnowledgeBaseIngester(rag_indexer) if rag_indexer is not None else None
        )
        self._retriever = (
            KnowledgeRetriever(
                rag_retriever,
                top_k=self.config.retrieval_top_k,
                query_rewriter=query_rewriter,
            )
            if rag_retriever is not None
            else None
        )

    def ingest(
        self,
        name: str,
        paths: Sequence[Path],
        *,
        description: str = "",
    ) -> KnowledgeBase:
        """Index source paths and persist the resulting knowledge base metadata."""

        if self._ingester is None:
            raise RuntimeError("Knowledge QA ingestion requires a configured indexer.")

        normalized_paths = tuple(
            path.expanduser() for path in paths if str(path).strip()
        )
        if not normalized_paths:
            raise ValueError("Please provide at least one source path to ingest.")

        knowledge_base = KnowledgeBase(
            kb_id=uuid4().hex,
            name=name.strip() or "knowledge-base",
            description=description.strip(),
            source_paths=tuple(str(path) for path in normalized_paths),
            status=KnowledgeBaseStatus.INDEXING,
        )
        knowledge_base = self._knowledge_base_store.save(knowledge_base)

        try:
            result = self._ingester.ingest(normalized_paths)
        except Exception:
            self._knowledge_base_store.mark_failed(knowledge_base.kb_id)
            raise

        ready = replace(
            knowledge_base,
            status=KnowledgeBaseStatus.READY,
            document_count=result.indexed_documents,
            chunk_count=result.indexed_chunks,
        )
        return self._knowledge_base_store.save(ready)

    def ask(
        self,
        question: str,
        *,
        kb_id: str | None = None,
    ) -> AnswerResult:
        """Answer one question against the configured knowledge base runtime."""

        if self._retriever is None:
            raise RuntimeError(
                "Knowledge QA answering requires a configured retriever."
            )
        if self._llm is None:
            raise RuntimeError("Knowledge QA answering requires a configured LLM.")

        prompt = question.strip()
        if not prompt:
            raise ValueError("Question must not be empty.")

        knowledge_base = self.get_knowledge_base(kb_id) if kb_id is not None else None
        if kb_id is not None and knowledge_base is None:
            raise ValueError(f"Unknown knowledge base: {kb_id}")
        source_paths = () if knowledge_base is None else knowledge_base.source_paths

        started = perf_counter()
        retrieval = self._retriever.retrieve(prompt, source_paths=source_paths)
        selected_chunks = retrieval.chunks[: self.config.answer_context_top_k]

        if len(selected_chunks) < self.config.min_retrieved_chunks:
            result = AnswerResult(
                answer="I do not know based on the current knowledge base.",
                answered=False,
                reason="no_relevant_context",
            )
            self._record_trace(
                question=prompt,
                rewritten_query=retrieval.query,
                retrieved_chunks=retrieval.chunks,
                selected_chunks=selected_chunks,
                rendered_prompt="",
                result=result,
                latency_ms=_elapsed_ms(started),
                token_usage=TokenUsage(),
            )
            return result

        messages = build_answer_messages(prompt, selected_chunks)
        rendered_prompt = messages[-1].content[: self.config.trace_prompt_chars]
        response = self._llm.chat(
            messages,
            max_tokens=self.config.answer_max_tokens,
        )
        parsed_answer = parse_answer_response(
            response.content,
            max_citation_index=len(selected_chunks),
        )
        citations = build_citations(
            selected_chunks,
            limit=min(self.config.max_citations, len(selected_chunks)),
            citation_indices=parsed_answer.citation_indices,
        )
        if not parsed_answer.answered and not parsed_answer.citation_indices:
            citations = ()

        result = AnswerResult(
            answer=parsed_answer.answer,
            citations=citations,
            confidence=_estimate_confidence(selected_chunks),
            answered=parsed_answer.answered,
            reason=parsed_answer.reason,
        )
        trace_id = self._record_trace(
            question=prompt,
            rewritten_query=retrieval.query,
            retrieved_chunks=retrieval.chunks,
            selected_chunks=selected_chunks,
            rendered_prompt=rendered_prompt,
            result=result,
            latency_ms=_elapsed_ms(started),
            token_usage=TokenUsage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
            ),
        )
        return replace(result, trace_id=trace_id)

    def list_knowledge_bases(self) -> tuple[KnowledgeBase, ...]:
        """Return all persisted knowledge bases."""

        return self._knowledge_base_store.list_all()

    def get_knowledge_base(self, kb_id: str | None) -> KnowledgeBase | None:
        """Return one persisted knowledge base when the identifier is present."""

        if kb_id is None:
            return None
        return self._knowledge_base_store.get(kb_id)

    def list_recent_traces(self, *, limit: int = 20) -> tuple[RunTrace, ...]:
        """Return recent answer traces."""

        return self._trace_store.list_recent(limit=limit)

    def _record_trace(
        self,
        *,
        question: str,
        rewritten_query: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        selected_chunks: Sequence[RetrievedChunk],
        rendered_prompt: str,
        result: AnswerResult,
        latency_ms: int,
        token_usage: TokenUsage,
    ) -> str:
        """Persist one run trace and return its identifier."""

        trace_id = uuid4().hex
        trace = RunTrace(
            trace_id=trace_id,
            question=question,
            rewritten_query=rewritten_query,
            retrieved_chunks=tuple(retrieved_chunks),
            selected_chunks=tuple(selected_chunks),
            rendered_prompt=rendered_prompt,
            answer=result.answer,
            citations=result.citations,
            answered=result.answered,
            reason=result.reason,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
        self._trace_store.append(trace)
        return trace_id


def _elapsed_ms(started: float) -> int:
    """Return elapsed wall-clock time in milliseconds."""

    return int((perf_counter() - started) * 1000)


def _estimate_confidence(chunks: Sequence[RetrievedChunk]) -> float:
    """Estimate a lightweight confidence score from selected chunks."""

    if not chunks:
        return 0.0
    positive_scores = [chunk.score for chunk in chunks if chunk.score > 0]
    if positive_scores:
        return min(1.0, sum(positive_scores) / len(positive_scores))
    return min(1.0, len(chunks) / 4)
