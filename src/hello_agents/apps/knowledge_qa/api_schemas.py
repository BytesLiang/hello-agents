"""Pydantic schemas for the knowledge QA HTTP API."""

from __future__ import annotations

from pydantic import BaseModel

from hello_agents.apps.knowledge_qa.models import (
    AnswerResult,
    Citation,
    KnowledgeBase,
    RetrievedChunk,
    RunTrace,
    TokenUsage,
)
from hello_agents.apps.knowledge_qa.runtime import KnowledgeQAHealthStatus


class HealthResponse(BaseModel):
    """Represent the service health payload."""

    status: str
    llm_model: str
    llm_provider: str
    qdrant_configured: bool
    embedding_configured: bool
    knowledge_base_store_path: str
    trace_store_path: str
    upload_root_path: str

    @classmethod
    def from_runtime(cls, payload: KnowledgeQAHealthStatus) -> HealthResponse:
        """Build a response schema from the runtime health snapshot."""

        return cls(
            status=payload.status,
            llm_model=payload.llm_model,
            llm_provider=payload.llm_provider,
            qdrant_configured=payload.qdrant_configured,
            embedding_configured=payload.embedding_configured,
            knowledge_base_store_path=payload.knowledge_base_store_path,
            trace_store_path=payload.trace_store_path,
            upload_root_path=payload.upload_root_path,
        )


class CreateKnowledgeBaseRequest(BaseModel):
    """Represent a request to ingest one knowledge base."""

    name: str
    description: str = ""
    paths: list[str]


class AskKnowledgeBaseRequest(BaseModel):
    """Represent one knowledge-base question."""

    question: str


class CitationResponse(BaseModel):
    """Represent one answer citation."""

    index: int
    source: str
    snippet: str
    chunk_id: str

    @classmethod
    def from_domain(cls, citation: Citation) -> CitationResponse:
        """Build a response schema from the domain citation."""

        return cls(
            index=citation.index,
            source=citation.source,
            snippet=citation.snippet,
            chunk_id=citation.chunk_id,
        )


class TokenUsageResponse(BaseModel):
    """Represent normalized token counts."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_domain(cls, token_usage: TokenUsage) -> TokenUsageResponse:
        """Build a response schema from the domain token usage."""

        return cls(
            prompt_tokens=token_usage.prompt_tokens,
            completion_tokens=token_usage.completion_tokens,
            total_tokens=token_usage.total_tokens,
        )


class RetrievedChunkResponse(BaseModel):
    """Represent one retrieved chunk in API responses."""

    chunk_id: str
    source: str
    heading_path: str
    content: str
    score: float
    rerank_score: float | None
    metadata: dict[str, object]

    @classmethod
    def from_domain(cls, chunk: RetrievedChunk) -> RetrievedChunkResponse:
        """Build a response schema from the domain retrieval chunk."""

        return cls(
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            heading_path=chunk.heading_path,
            content=chunk.content,
            score=chunk.score,
            rerank_score=chunk.rerank_score,
            metadata=chunk.metadata,
        )


class KnowledgeBaseResponse(BaseModel):
    """Represent one knowledge base for HTTP responses."""

    kb_id: str
    name: str
    description: str
    source_paths: list[str]
    status: str
    document_count: int
    chunk_count: int
    created_at: str
    updated_at: str

    @classmethod
    def from_domain(cls, knowledge_base: KnowledgeBase) -> KnowledgeBaseResponse:
        """Build a response schema from the domain knowledge base."""

        return cls(
            kb_id=knowledge_base.kb_id,
            name=knowledge_base.name,
            description=knowledge_base.description,
            source_paths=list(knowledge_base.source_paths),
            status=knowledge_base.status.value,
            document_count=knowledge_base.document_count,
            chunk_count=knowledge_base.chunk_count,
            created_at=knowledge_base.created_at,
            updated_at=knowledge_base.updated_at,
        )


class AnswerResultResponse(BaseModel):
    """Represent one answer payload."""

    answer: str
    citations: list[CitationResponse]
    confidence: float
    answered: bool
    reason: str | None = None
    trace_id: str | None = None

    @classmethod
    def from_domain(cls, result: AnswerResult) -> AnswerResultResponse:
        """Build a response schema from the domain answer result."""

        return cls(
            answer=result.answer,
            citations=[
                CitationResponse.from_domain(citation) for citation in result.citations
            ],
            confidence=result.confidence,
            answered=result.answered,
            reason=result.reason,
            trace_id=result.trace_id,
        )


class RunTraceResponse(BaseModel):
    """Represent one recent trace payload."""

    trace_id: str
    question: str
    rewritten_query: str
    retrieved_chunks: list[RetrievedChunkResponse]
    selected_chunks: list[RetrievedChunkResponse]
    rendered_prompt: str
    answer: str
    citations: list[CitationResponse]
    answered: bool
    reason: str | None = None
    latency_ms: int
    token_usage: TokenUsageResponse
    created_at: str

    @classmethod
    def from_domain(cls, trace: RunTrace) -> RunTraceResponse:
        """Build a response schema from the domain trace."""

        return cls(
            trace_id=trace.trace_id,
            question=trace.question,
            rewritten_query=trace.rewritten_query,
            retrieved_chunks=[
                RetrievedChunkResponse.from_domain(chunk)
                for chunk in trace.retrieved_chunks
            ],
            selected_chunks=[
                RetrievedChunkResponse.from_domain(chunk)
                for chunk in trace.selected_chunks
            ],
            rendered_prompt=trace.rendered_prompt,
            answer=trace.answer,
            citations=[
                CitationResponse.from_domain(citation) for citation in trace.citations
            ],
            answered=trace.answered,
            reason=trace.reason,
            latency_ms=trace.latency_ms,
            token_usage=TokenUsageResponse.from_domain(trace.token_usage),
            created_at=trace.created_at,
        )
