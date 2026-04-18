"""Retrieval helpers for the knowledge QA application."""

from __future__ import annotations

import re
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

    _FILENAME_PATTERN = re.compile(r"[\w.-]+\.[A-Za-z][A-Za-z0-9]{0,15}")

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

        normalized_question = question.strip()
        query = self._query_rewriter.rewrite(normalized_question)
        if not query:
            return RetrievalResult(query="", chunks=())

        referenced_filenames = _extract_referenced_filenames(
            normalized_question,
            rewritten_query=query,
        )
        chunks = [
            _normalize_chunk(chunk)
            for chunk in self._retriever.query(
                query,
                top_k=_expanded_top_k(
                    self._top_k,
                    has_filename_hint=bool(referenced_filenames),
                ),
            )
        ]
        if source_paths:
            chunks = [
                chunk
                for chunk in chunks
                if _matches_any_source(chunk.source, source_paths)
            ]
        deduplicated_chunks = _deduplicate_chunks(chunks)
        ranked_chunks = _prioritize_referenced_sources(
            deduplicated_chunks,
            referenced_filenames=referenced_filenames,
        )
        return RetrievalResult(query=query, chunks=tuple(ranked_chunks[: self._top_k]))


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


def _extract_referenced_filenames(
    question: str,
    *,
    rewritten_query: str,
) -> tuple[str, ...]:
    """Return normalized filenames explicitly mentioned in the question."""

    candidates = f"{question}\n{rewritten_query}"
    filenames: list[str] = []
    seen: set[str] = set()
    for match in KnowledgeRetriever._FILENAME_PATTERN.findall(candidates):
        normalized = Path(match).name.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        filenames.append(normalized)
    return tuple(filenames)


def _expanded_top_k(top_k: int, *, has_filename_hint: bool) -> int:
    """Fetch a wider candidate pool when the question names a source file."""

    if not has_filename_hint:
        return top_k
    return max(top_k * 3, top_k + 6)


def _deduplicate_chunks(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop exact duplicate retrieval hits while preserving first occurrence."""

    unique_chunks: list[RetrievedChunk] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        key = (
            chunk.source,
            chunk.heading_path,
            " ".join(chunk.content.split()),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_chunks.append(chunk)
    return unique_chunks


def _prioritize_referenced_sources(
    chunks: Sequence[RetrievedChunk],
    *,
    referenced_filenames: Sequence[str],
) -> list[RetrievedChunk]:
    """Prefer chunks from explicitly named files before other retrieval hits."""

    if not referenced_filenames:
        return list(chunks)

    prioritized = sorted(
        enumerate(chunks),
        key=lambda item: (_source_match_rank(item[1], referenced_filenames), item[0]),
    )
    return [chunk for _, chunk in prioritized]


def _source_match_rank(
    chunk: RetrievedChunk,
    referenced_filenames: Sequence[str],
) -> int:
    """Return a sort rank for how directly a chunk source matches the query."""

    source_name = Path(chunk.source).name.lower()
    return 0 if source_name in referenced_filenames else 1
