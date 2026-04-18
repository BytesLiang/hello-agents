"""Retrieval helpers for the knowledge QA application."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hello_agents.apps.knowledge_qa.models import RetrievedChunk
from hello_agents.rag.models import RagChunk


class QueryRewriter(Protocol):
    """Rewrite user questions before retrieval when desired."""

    def rewrite(self, question: str) -> str:
        """Return a rewritten retrieval query."""


class SupportsRagQuery(Protocol):
    """Represent the retrieval interface required by the app layer."""

    def query(self, text: str, *, top_k: int | None = None) -> list[RagChunk]:
        """Return the top retrieval hits for the provided text."""


class IdentityQueryRewriter:
    """Return the original question unchanged."""

    def rewrite(self, question: str) -> str:
        """Return the question unchanged."""

        return question


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """Bundle the rewritten query with normalized retrieval hits."""

    query: str
    chunks: tuple[RetrievedChunk, ...]


class KnowledgeRetriever:
    """Normalize and optionally filter RAG retrieval results."""

    def __init__(
        self,
        retriever: SupportsRagQuery,
        *,
        top_k: int,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        """Store the retrieval dependency and retrieval policy."""

        self._retriever = retriever
        self._top_k = top_k
        self._query_rewriter = query_rewriter or IdentityQueryRewriter()

    def retrieve(
        self,
        question: str,
        *,
        source_paths: Sequence[str] = (),
    ) -> RetrievalResult:
        """Retrieve normalized chunks for one question."""

        query = self._query_rewriter.rewrite(question.strip())
        if not query:
            return RetrievalResult(query="", chunks=())

        chunks = [
            _normalize_chunk(chunk)
            for chunk in self._retriever.query(query, top_k=self._top_k)
        ]
        if source_paths:
            chunks = [
                chunk
                for chunk in chunks
                if _matches_any_source(chunk.source, source_paths)
            ]
        return RetrievalResult(query=query, chunks=tuple(chunks))


def _normalize_chunk(chunk: RagChunk) -> RetrievedChunk:
    """Normalize a raw RAG chunk into an application-facing shape."""

    heading_path = chunk.metadata.get("heading_path", "")
    return RetrievedChunk(
        chunk_id=chunk.id,
        source=chunk.source,
        heading_path=heading_path if isinstance(heading_path, str) else "",
        content=chunk.content,
        score=chunk.score,
        metadata=dict(chunk.metadata),
    )


def _matches_any_source(source: str, source_paths: Sequence[str]) -> bool:
    """Return whether a source path belongs to one of the supplied roots."""

    for raw_path in source_paths:
        if _path_matches(source, raw_path):
            return True
    return False


def _path_matches(source: str, base: str) -> bool:
    """Return whether a source path matches or is nested under a base path."""

    source_path = Path(source).expanduser()
    base_path = Path(base).expanduser()
    try:
        resolved_source = source_path.resolve(strict=False)
        resolved_base = base_path.resolve(strict=False)
        return (
            resolved_source == resolved_base or resolved_base in resolved_source.parents
        )
    except OSError:
        return source.startswith(base)
