"""Service layer for the knowledge QA application."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from hello_agents.apps.knowledge_qa.agent import AgentRunResult, QuestionAgent
from hello_agents.apps.knowledge_qa.answer import (
    build_answer_messages,
    build_citations,
    parse_answer_response,
    validate_citations,
)
from hello_agents.apps.knowledge_qa.config import KnowledgeQAConfig
from hello_agents.apps.knowledge_qa.evidence import EvidenceStatus
from hello_agents.apps.knowledge_qa.ingest import (
    IndexedDocument,
    KnowledgeBaseIngester,
    SupportsDocumentIndex,
)
from hello_agents.apps.knowledge_qa.llm_utils import SupportsChat, merge_token_usage
from hello_agents.apps.knowledge_qa.models import (
    AnswerResult,
    Citation,
    KnowledgeBase,
    KnowledgeBaseStatus,
    KnowledgeDocument,
    RetrievedChunk,
    RunTrace,
    TokenUsage,
)
from hello_agents.apps.knowledge_qa.retrieve import (
    DashScopeChunkReranker,
    KnowledgeRetriever,
    QueryRewriter,
    SupportsRagQuery,
)
from hello_agents.apps.knowledge_qa.store import JsonKnowledgeBaseStore
from hello_agents.apps.knowledge_qa.trace import JsonlRunTraceStore


@dataclass(slots=True, frozen=True)
class ValidatedAnswerState:
    """Capture the final answer after citation validation."""

    answer: str
    citations: tuple[Citation, ...]
    answered: bool
    reason: str | None
    rendered_prompt: str
    token_usage: TokenUsage
    citation_validation: dict[str, object] | None = None


class KnowledgeQAService:
    """Coordinate indexing, retrieval, answering, and trace persistence."""

    def __init__(
        self,
        *,
        config: KnowledgeQAConfig | None = None,
        llm: SupportsChat | None = None,
        rag_retriever: SupportsRagQuery | None = None,
        rag_indexer: SupportsDocumentIndex | None = None,
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
                reranker=DashScopeChunkReranker.from_env(),
            )
            if rag_retriever is not None
            else None
        )
        self._agent = (
            QuestionAgent(self._retriever, llm=self._llm)
            if self._retriever is not None
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
            status=KnowledgeBaseStatus.INDEXING,
            uses_scoped_index=True,
        )
        knowledge_base = self._knowledge_base_store.save(knowledge_base)

        try:
            result = self._ingester.ingest(
                knowledge_base.kb_id,
                normalized_paths,
            )
        except Exception:
            self._knowledge_base_store.mark_failed(knowledge_base.kb_id)
            raise

        ready = _apply_ingestion_result(
            knowledge_base,
            result.documents,
            status=KnowledgeBaseStatus.READY,
        )
        return self._knowledge_base_store.save(ready)

    def add_documents(self, kb_id: str, paths: Sequence[Path]) -> KnowledgeBase:
        """Add one or more documents to an existing knowledge base."""

        if self._ingester is None:
            raise RuntimeError("Knowledge QA ingestion requires a configured indexer.")

        knowledge_base = self.get_knowledge_base(kb_id)
        if knowledge_base is None:
            raise ValueError(f"Unknown knowledge base: {kb_id}")

        normalized_paths = tuple(
            path.expanduser() for path in paths if str(path).strip()
        )
        if not normalized_paths:
            raise ValueError("Please provide at least one source path to ingest.")

        indexing = self._knowledge_base_store.save(
            replace(knowledge_base, status=KnowledgeBaseStatus.INDEXING)
        )
        try:
            result = self._ingester.ingest(
                kb_id,
                normalized_paths,
                existing_source_paths=indexing.source_paths,
            )
        except Exception:
            self._knowledge_base_store.mark_failed(kb_id)
            raise

        ready = _apply_ingestion_result(
            indexing,
            result.documents,
            status=KnowledgeBaseStatus.READY,
        )
        return self._knowledge_base_store.save(ready)

    def remove_document(self, kb_id: str, document_id: str) -> KnowledgeBase:
        """Remove one document from an existing knowledge base."""

        if self._ingester is None:
            raise RuntimeError("Knowledge QA ingestion requires a configured indexer.")

        knowledge_base = self.get_knowledge_base(kb_id)
        if knowledge_base is None:
            raise ValueError(f"Unknown knowledge base: {kb_id}")

        remaining_documents = tuple(
            document
            for document in knowledge_base.documents
            if document.document_id != document_id
        )
        if len(remaining_documents) == len(knowledge_base.documents):
            raise ValueError(f"Unknown document: {document_id}")

        self._ingester.delete_document(kb_id=kb_id, document_id=document_id)
        updated = replace(
            knowledge_base,
            documents=remaining_documents,
            source_paths=tuple(
                document.source_path for document in remaining_documents
            ),
            document_count=len(remaining_documents),
            chunk_count=sum(document.chunk_count for document in remaining_documents),
            status=KnowledgeBaseStatus.READY,
        )
        return self._knowledge_base_store.save(updated)

    def ask(
        self,
        question: str,
        *,
        kb_id: str | None = None,
    ) -> AnswerResult:
        """Answer one question against the configured knowledge base runtime."""

        if self._retriever is None or self._agent is None:
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
        query_kb_id = (
            knowledge_base.kb_id
            if knowledge_base is not None and knowledge_base.uses_scoped_index
            else None
        )

        started = perf_counter()
        agent_result = self._agent.run(
            question=prompt,
            source_paths=source_paths,
            kb_id=query_kb_id,
            answer_context_top_k=self.config.answer_context_top_k,
        )
        selected_chunks = agent_result.selected_chunks

        if (
            len(selected_chunks) < self.config.min_retrieved_chunks
            and agent_result.inspection_result is None
        ):
            result = AnswerResult(
                answer="I do not know based on the current knowledge base.",
                answered=False,
                reason=_normalize_failure_reason(agent_result.final_assessment),
            )
            self._record_trace(
                question=prompt,
                agent_result=agent_result,
                selected_chunks=selected_chunks,
                rendered_prompt="",
                result=result,
                latency_ms=_elapsed_ms(started),
                token_usage=agent_result.token_usage,
                citation_validation=None,
            )
            return result

        if (
            agent_result.final_assessment.status is not EvidenceStatus.SUFFICIENT
            and agent_result.inspection_result is None
        ):
            result = AnswerResult(
                answer="I do not know based on the current knowledge base.",
                answered=False,
                reason=agent_result.final_assessment.failure_mode
                or "insufficient_evidence",
            )
            self._record_trace(
                question=prompt,
                agent_result=agent_result,
                selected_chunks=selected_chunks,
                rendered_prompt="",
                result=result,
                latency_ms=_elapsed_ms(started),
                token_usage=agent_result.token_usage,
                citation_validation=None,
            )
            return result

        inspection_summary = (
            agent_result.inspection_result.output
            if agent_result.inspection_result is not None
            else None
        )
        answer_state = self._generate_validated_answer(
            question=agent_result.input_check.normalized_question,
            selected_chunks=selected_chunks,
            inspection_summary=inspection_summary,
        )

        result = AnswerResult(
            answer=answer_state.answer,
            citations=answer_state.citations,
            confidence=max(
                agent_result.final_assessment.score,
                _estimate_confidence(selected_chunks),
            ),
            answered=answer_state.answered,
            reason=answer_state.reason,
        )
        trace_id = self._record_trace(
            question=prompt,
            agent_result=agent_result,
            selected_chunks=selected_chunks,
            rendered_prompt=answer_state.rendered_prompt,
            result=result,
            latency_ms=_elapsed_ms(started),
            token_usage=merge_token_usage(
                agent_result.token_usage,
                answer_state.token_usage,
            ),
            citation_validation=answer_state.citation_validation,
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
        agent_result: AgentRunResult,
        selected_chunks: Sequence[RetrievedChunk],
        rendered_prompt: str,
        result: AnswerResult,
        latency_ms: int,
        token_usage: TokenUsage,
        citation_validation: dict[str, object] | None,
    ) -> str:
        """Persist one run trace and return its identifier."""

        trace_id = uuid4().hex
        trace = RunTrace(
            trace_id=trace_id,
            question=question,
            normalized_question=agent_result.input_check.normalized_question,
            rewritten_query=agent_result.rewritten_query,
            input_check={
                "is_valid": agent_result.input_check.is_valid,
                "issues": list(agent_result.input_check.issues),
                "reason": agent_result.input_check.reason,
                "used_llm": agent_result.input_check.used_llm,
            },
            question_type=agent_result.classification.question_type.value,
            classification_reason=agent_result.classification.reason,
            plan_summary=agent_result.plan.summary,
            plan={
                "primary_queries": list(agent_result.plan.primary_queries),
                "fallback_queries": list(agent_result.plan.fallback_queries),
                "max_rounds": agent_result.plan.max_rounds,
                "use_document_inspection_on_failure": (
                    agent_result.plan.use_document_inspection_on_failure
                ),
            },
            retrieved_chunks=tuple(agent_result.retrieved_chunks),
            selected_chunks=tuple(selected_chunks),
            retrieval_rounds=tuple(
                {
                    "round_index": round_item.round_index,
                    "query": round_item.query,
                    "rewritten_query": round_item.rewritten_query,
                    "assessment_status": round_item.assessment.status.value,
                    "assessment_score": round_item.assessment.score,
                    "assessment_reason": round_item.assessment.reason,
                    "failure_mode": round_item.assessment.failure_mode,
                    "retrieved_chunk_count": len(round_item.retrieved_chunks),
                    "suggested_query": round_item.suggested_query,
                }
                for round_item in agent_result.rounds
            ),
            inspection_result=(
                {
                    "source": agent_result.inspection_result.source,
                    "operation": agent_result.inspection_result.operation,
                    "output": agent_result.inspection_result.output,
                    "metadata": agent_result.inspection_result.metadata,
                }
                if agent_result.inspection_result is not None
                else None
            ),
            citation_validation=citation_validation,
            evidence_score=agent_result.final_assessment.score,
            failure_mode=agent_result.final_assessment.failure_mode,
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

    def _generate_validated_answer(
        self,
        *,
        question: str,
        selected_chunks: Sequence[RetrievedChunk],
        inspection_summary: str | None,
    ) -> ValidatedAnswerState:
        """Generate an answer and validate its citations before returning it."""

        assert self._llm is not None
        validation_feedback: str | None = None
        total_token_usage = TokenUsage()
        rendered_prompt = ""
        latest_validation: dict[str, object] | None = None

        for _ in range(2):
            messages = build_answer_messages(
                question,
                selected_chunks,
                inspection_summary=inspection_summary,
                validation_feedback=validation_feedback,
            )
            rendered_prompt = messages[-1].content[: self.config.trace_prompt_chars]
            response = self._llm.chat(
                messages,
                max_tokens=self.config.answer_max_tokens,
            )
            parsed_answer = parse_answer_response(
                response.content,
                max_citation_index=len(selected_chunks),
            )
            if (
                not parsed_answer.used_structured_output
                and parsed_answer.answered
                and not inspection_summary
                and selected_chunks
            ):
                parsed_answer = replace(parsed_answer, citation_indices=(1,))
            total_token_usage = merge_token_usage(
                total_token_usage,
                TokenUsage(
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                ),
            )
            validation = validate_citations(
                question=question,
                parsed_answer=parsed_answer,
                chunks=selected_chunks,
                inspection_summary=inspection_summary,
                llm=self._llm,
            )
            total_token_usage = merge_token_usage(
                total_token_usage,
                validation.token_usage,
            )
            latest_validation = {
                "is_valid": validation.is_valid,
                "reason": validation.reason,
                "citation_indices": list(validation.citation_indices),
                "answered": validation.answered,
                "used_llm": validation.used_llm,
            }
            citation_indices = (
                validation.citation_indices
                if validation.citation_indices
                else parsed_answer.citation_indices
            )
            citations = build_citations(
                selected_chunks,
                limit=min(self.config.max_citations, len(selected_chunks)),
                citation_indices=citation_indices,
            )
            if agent_result_inspection_only(
                inspection_summary=inspection_summary,
                citation_indices=citation_indices,
            ):
                citations = ()
            if not parsed_answer.answered and not citation_indices:
                citations = ()
            if validation.is_valid:
                answered = (
                    validation.answered
                    if isinstance(validation.answered, bool)
                    else parsed_answer.answered
                )
                reason = parsed_answer.reason
                if not answered and validation.reason:
                    reason = validation.reason
                return ValidatedAnswerState(
                    answer=parsed_answer.answer,
                    citations=citations,
                    answered=answered,
                    reason=reason,
                    rendered_prompt=rendered_prompt,
                    token_usage=total_token_usage,
                    citation_validation=latest_validation,
                )
            validation_feedback = (
                validation.reason or "Fix unsupported claims and citations."
            )

        return ValidatedAnswerState(
            answer="I do not know based on the current knowledge base.",
            citations=(),
            answered=False,
            reason="citation_validation_failed",
            rendered_prompt=rendered_prompt,
            token_usage=total_token_usage,
            citation_validation=latest_validation,
        )


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


def _apply_ingestion_result(
    knowledge_base: KnowledgeBase,
    documents: Sequence[IndexedDocument],
    *,
    status: KnowledgeBaseStatus,
) -> KnowledgeBase:
    """Merge newly indexed documents into one persisted knowledge base."""

    merged_documents = tuple(knowledge_base.documents) + tuple(
        KnowledgeDocument(
            document_id=document.document_id,
            name=document.name,
            source_path=document.source_path,
            chunk_count=document.chunk_count,
            size_bytes=document.size_bytes,
        )
        for document in documents
    )
    return replace(
        knowledge_base,
        documents=merged_documents,
        source_paths=tuple(document.source_path for document in merged_documents),
        document_count=len(merged_documents),
        chunk_count=sum(document.chunk_count for document in merged_documents),
        status=status,
    )


def _normalize_failure_reason(final_assessment: object) -> str:
    """Map internal assessment failures to user-facing refusal reasons."""

    failure_mode = getattr(final_assessment, "failure_mode", None)
    if failure_mode in {"", None, "no_hits", "no_plan"}:
        return "no_relevant_context"
    if isinstance(failure_mode, str):
        return failure_mode
    return "no_relevant_context"


def agent_result_inspection_only(
    *,
    inspection_summary: str | None,
    citation_indices: Sequence[int],
) -> bool:
    """Return whether the answer relies only on direct document inspection."""

    return bool(
        inspection_summary and inspection_summary.strip() and not citation_indices
    )
