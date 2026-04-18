"""Typed models used by the knowledge QA application."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


def utc_now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""

    return datetime.now(UTC).isoformat()


class KnowledgeBaseStatus(StrEnum):
    """Represent the lifecycle state of a knowledge base."""

    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class KnowledgeBase:
    """Describe one knowledge base and its indexed sources."""

    kb_id: str
    name: str
    description: str = ""
    source_paths: tuple[str, ...] = ()
    status: KnowledgeBaseStatus = KnowledgeBaseStatus.READY
    document_count: int = 0
    chunk_count: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """Represent one normalized retrieval hit used for answering."""

    chunk_id: str
    source: str
    heading_path: str = ""
    content: str = ""
    score: float = 0.0
    rerank_score: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Citation:
    """Represent one answer citation."""

    index: int
    source: str
    snippet: str
    chunk_id: str


@dataclass(slots=True, frozen=True)
class TokenUsage:
    """Represent normalized token usage for one LLM interaction."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True, frozen=True)
class AnswerResult:
    """Represent the answer returned to the knowledge QA caller."""

    answer: str
    citations: tuple[Citation, ...] = ()
    confidence: float = 0.0
    answered: bool = True
    reason: str | None = None
    trace_id: str | None = None


@dataclass(slots=True, frozen=True)
class RunTrace:
    """Capture the execution trace for one knowledge QA question."""

    trace_id: str
    question: str
    rewritten_query: str
    retrieved_chunks: tuple[RetrievedChunk, ...] = ()
    selected_chunks: tuple[RetrievedChunk, ...] = ()
    rendered_prompt: str = ""
    answer: str = ""
    citations: tuple[Citation, ...] = ()
    answered: bool = False
    reason: str | None = None
    latency_ms: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    created_at: str = field(default_factory=utc_now_iso)
