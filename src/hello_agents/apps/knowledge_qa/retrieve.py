"""Retrieval helpers for the knowledge QA application."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from pathlib import Path
from typing import Any, Protocol

from hello_agents.apps.knowledge_qa.models import RetrievedChunk, TokenUsage
from hello_agents.rag.models import RagChunk

try:
    import dashscope as _dashscope  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    _dashscope = None

dashscope: Any = _dashscope

DEFAULT_RERANK_MODEL = "qwen3-rerank"
DEFAULT_RERANK_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query."
)


class QueryRewriter(Protocol):
    """Rewrite user questions before retrieval when desired."""

    def rewrite(self, question: str) -> str:
        """Return a rewritten retrieval query."""


class SupportsRagQuery(Protocol):
    """Represent the retrieval interface required by the app layer."""

    def query(
        self,
        text: str,
        *,
        top_k: int | None = None,
        kb_id: str | None = None,
    ) -> list[RagChunk]:
        """Return the top retrieval hits for the provided text."""


class Reranker(Protocol):
    """Describe the reranking contract used by the retriever."""

    def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
        referenced_filenames: Sequence[str],
    ) -> RerankResult:
        """Return the reranked chunks plus any model token usage."""


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
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(slots=True, frozen=True)
class RerankResult:
    """Capture the reranked chunks and any model token usage."""

    chunks: tuple[RetrievedChunk, ...]
    token_usage: TokenUsage = field(default_factory=TokenUsage)


class HeuristicChunkReranker:
    """Apply a local lexical reranking pass over retrieved chunks."""

    def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
        referenced_filenames: Sequence[str],
    ) -> RerankResult:
        """Return the locally reranked chunks."""

        reranked_chunks = _rerank_chunks(
            chunks,
            query=query,
            referenced_filenames=referenced_filenames,
        )
        return RerankResult(chunks=tuple(reranked_chunks))


class DashScopeChunkReranker:
    """Use DashScope qwen rerank models to rerank retrieved chunks."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_RERANK_MODEL,
        instruct: str = DEFAULT_RERANK_INSTRUCT,
        fallback: Reranker | None = None,
        return_documents: bool = False,
    ) -> None:
        """Store the DashScope rerank configuration."""

        self._api_key = api_key
        self._model = model
        self._instruct = instruct
        self._fallback = fallback or HeuristicChunkReranker()
        self._return_documents = return_documents

    @classmethod
    def from_env(cls) -> DashScopeChunkReranker | None:
        """Build a reranker from environment variables when configured."""

        api_key = os.getenv("RERANK_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=os.getenv("RERANK_MODEL_NAME", DEFAULT_RERANK_MODEL),
            instruct=os.getenv("KNOWLEDGE_QA_RERANK_INSTRUCT", DEFAULT_RERANK_INSTRUCT),
        )

    def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
        referenced_filenames: Sequence[str],
    ) -> RerankResult:
        """Return DashScope-reranked chunks, falling back when unavailable."""

        baseline = self._fallback.rerank(
            query=query,
            chunks=chunks,
            referenced_filenames=referenced_filenames,
        )
        if len(baseline.chunks) <= 1 or dashscope is None:
            return baseline

        dashscope.api_key = self._api_key
        response = dashscope.TextReRank.call(
            model=self._model,
            query=query,
            documents=[_chunk_to_rerank_document(chunk) for chunk in baseline.chunks],
            top_n=len(baseline.chunks),
            return_documents=self._return_documents,
            instruct=self._instruct,
        )
        if getattr(response, "status_code", None) != HTTPStatus.OK:
            return baseline

        reranked_chunks = _apply_dashscope_ranking(baseline.chunks, response=response)
        if reranked_chunks is None:
            return baseline
        token_usage = _extract_dashscope_usage(response)
        return RerankResult(chunks=tuple(reranked_chunks), token_usage=token_usage)


class KnowledgeRetriever:
    """Normalize and optionally filter RAG retrieval results."""

    _ENGLISH_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
    _CJK_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
    _FILENAME_PATTERN = re.compile(
        r"(?<![A-Za-z0-9_.-])"
        r"([A-Za-z0-9_.-]+\.[A-Za-z][A-Za-z0-9]{0,15})"
        r"(?![A-Za-z0-9_.-])"
    )
    _STOPWORDS = frozenset(
        {
            "a",
            "an",
            "are",
            "as",
            "at",
            "be",
            "by",
            "does",
            "for",
            "from",
            "how",
            "in",
            "into",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "this",
            "to",
            "use",
            "uses",
            "used",
            "using",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }
    )

    def __init__(
        self,
        retriever: SupportsRagQuery,
        *,
        top_k: int,
        query_rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """Store the retrieval dependency and retrieval policy."""

        self._retriever = retriever
        self._top_k = top_k
        self._query_rewriter = query_rewriter or IdentityQueryRewriter()
        self._reranker = reranker or HeuristicChunkReranker()

    def retrieve(
        self,
        question: str,
        *,
        kb_id: str | None = None,
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
                kb_id=kb_id,
            )
        ]
        deduplicated_chunks = _deduplicate_chunks(chunks)
        rerank_result = self._reranker.rerank(
            query=query,
            chunks=deduplicated_chunks,
            referenced_filenames=referenced_filenames,
        )
        ranked_chunks = _prioritize_referenced_sources(
            rerank_result.chunks,
            referenced_filenames=referenced_filenames,
        )
        return RetrievalResult(
            query=query,
            chunks=tuple(ranked_chunks[: self._top_k]),
            token_usage=rerank_result.token_usage,
        )


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

def _extract_referenced_filenames(
    question: str,
    *,
    rewritten_query: str,
) -> tuple[str, ...]:
    """Return normalized filenames explicitly mentioned in the question."""

    candidates = f"{question}\n{rewritten_query}"
    filenames: list[str] = []
    seen: set[str] = set()
    for match in KnowledgeRetriever._FILENAME_PATTERN.finditer(candidates):
        normalized = Path(match.group(1)).name.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        filenames.append(normalized)
    return tuple(filenames)


def _expanded_top_k(top_k: int, *, has_filename_hint: bool) -> int:
    """Fetch a wider candidate pool before reranking."""

    expanded = max(top_k * 3, top_k + 6)
    if has_filename_hint:
        return expanded
    return expanded


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
        key=lambda item: (
            _source_match_rank(item[1], referenced_filenames),
            -_ranking_score(item[1]),
            item[0],
        ),
    )
    return [chunk for _, chunk in prioritized]


def _source_match_rank(
    chunk: RetrievedChunk,
    referenced_filenames: Sequence[str],
) -> int:
    """Return a sort rank for how directly a chunk source matches the query."""

    source_name = Path(chunk.source).name.lower()
    return 0 if source_name in referenced_filenames else 1


def _chunk_to_rerank_document(chunk: RetrievedChunk) -> str:
    """Render one retrieved chunk into a DashScope rerank document string."""

    lines = [f"Source: {chunk.source}"]
    if chunk.heading_path:
        lines.append(f"Heading: {chunk.heading_path}")
    lines.append(f"Content: {chunk.content}")
    return "\n".join(lines)


def _apply_dashscope_ranking(
    chunks: Sequence[RetrievedChunk],
    *,
    response: object,
) -> list[RetrievedChunk] | None:
    """Apply DashScope rerank results to the current chunk list."""

    results = _extract_dashscope_results(response)
    if not results:
        return None

    reranked: list[tuple[float, float, int, RetrievedChunk]] = []
    for index, chunk in enumerate(chunks, start=1):
        baseline_score = _ranking_score(chunk)
        rerank_score = baseline_score
        for result in results:
            result_index = _result_index(result)
            result_score = _result_score(result)
            if result_index == index and result_score is not None:
                rerank_score = result_score
                break
        reranked.append(
            (
                rerank_score,
                baseline_score,
                index,
                replace(chunk, rerank_score=round(rerank_score, 6)),
            )
        )
    reranked.sort(key=lambda item: (-item[0], -item[1], -item[3].score, item[2]))
    return [chunk for _, _, _, chunk in reranked]


def _extract_dashscope_results(response: object) -> list[object]:
    """Extract raw DashScope rerank results from a response payload."""

    output = getattr(response, "output", None)
    if isinstance(output, dict):
        raw_results = output.get("results")
    else:
        raw_results = getattr(output, "results", None)
    return list(raw_results) if isinstance(raw_results, list) else []


def _extract_dashscope_usage(response: object) -> TokenUsage:
    """Extract token usage from a DashScope rerank response."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()

    if isinstance(usage, dict):
        return TokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        )

    return TokenUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
    )


def _result_index(result: object) -> int | None:
    """Extract one-based result indices from DashScope output."""

    raw_index: object
    if isinstance(result, dict):
        raw_index = result.get("index")
    else:
        raw_index = getattr(result, "index", None)
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        return None
    return raw_index + 1


def _result_score(result: object) -> float | None:
    """Extract normalized relevance scores from DashScope output."""

    raw_score: object
    if isinstance(result, dict):
        raw_score = result.get("relevance_score")
    else:
        raw_score = getattr(result, "relevance_score", None)
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return None
    return min(1.0, max(0.0, float(raw_score)))


def _rerank_chunks(
    chunks: Sequence[RetrievedChunk],
    *,
    query: str,
    referenced_filenames: Sequence[str],
) -> list[RetrievedChunk]:
    """Apply a lightweight lexical reranker over retrieved chunks."""

    if not chunks:
        return []

    query_terms = _tokenize(query)
    query_phrases = _query_phrases(query)
    score_bounds = _score_bounds(chunks)
    reranked: list[tuple[float, int, RetrievedChunk]] = []

    for index, chunk in enumerate(chunks):
        content_terms = _tokenize(chunk.content)
        heading_terms = _tokenize(chunk.heading_path)
        source_terms = _tokenize(Path(chunk.source).name)
        lexical_overlap = _term_overlap(query_terms, content_terms | heading_terms)
        heading_overlap = _term_overlap(query_terms, heading_terms)
        source_overlap = max(
            _term_overlap(query_terms, source_terms),
            1.0 if Path(chunk.source).name.lower() in referenced_filenames else 0.0,
        )
        phrase_overlap = _phrase_overlap(
            query_phrases,
            _normalized_text(
                " ".join(
                    part
                    for part in (chunk.source, chunk.heading_path, chunk.content)
                    if part
                )
            ),
        )
        dense_rank = _dense_rank_score(index=index, total=len(chunks))
        dense_score = _normalized_dense_score(chunk.score, bounds=score_bounds)
        rerank_score = min(
            1.0,
            (
                0.15 * dense_rank
                + 0.05 * dense_score
                + 0.45 * lexical_overlap
                + 0.10 * heading_overlap
                + 0.10 * phrase_overlap
                + 0.15 * source_overlap
            ),
        )
        reranked.append(
            (
                rerank_score,
                index,
                replace(chunk, rerank_score=round(rerank_score, 6)),
            )
        )

    reranked.sort(key=lambda item: (-item[0], -item[2].score, item[1]))
    return [chunk for _, _, chunk in reranked]


def _ranking_score(chunk: RetrievedChunk) -> float:
    """Return the score used for final ordering after reranking."""

    if chunk.rerank_score is not None:
        return float(chunk.rerank_score)
    return float(chunk.score)


def _score_bounds(chunks: Sequence[RetrievedChunk]) -> tuple[float, float]:
    """Return the min/max bounds for dense retrieval scores."""

    scores = [chunk.score for chunk in chunks]
    return (min(scores), max(scores))


def _normalized_dense_score(
    score: float,
    *,
    bounds: tuple[float, float],
) -> float:
    """Normalize one dense retrieval score into the `[0, 1]` range."""

    lower, upper = bounds
    if upper <= lower:
        return 1.0 if upper > 0 else 0.0
    return min(1.0, max(0.0, (score - lower) / (upper - lower)))


def _dense_rank_score(*, index: int, total: int) -> float:
    """Convert the original retrieval order into a soft prior score."""

    if total <= 1:
        return 1.0
    return max(0.0, 1.0 - (index / (total - 1)))


def _term_overlap(query_terms: set[str], chunk_terms: set[str]) -> float:
    """Return the fraction of query terms covered by the chunk."""

    if not query_terms or not chunk_terms:
        return 0.0
    return len(query_terms & chunk_terms) / len(query_terms)


def _query_phrases(query: str) -> tuple[str, ...]:
    """Return a small set of query phrases for phrase matching."""

    ordered_terms = _ordered_tokens(query)
    if len(ordered_terms) < 2:
        return ()
    phrases = [
        " ".join(ordered_terms[index : index + 2])
        for index in range(len(ordered_terms) - 1)
    ]
    return tuple(dict.fromkeys(phrases))


def _phrase_overlap(phrases: Sequence[str], normalized_chunk_text: str) -> float:
    """Return phrase coverage for the query against one chunk."""

    if not phrases or not normalized_chunk_text:
        return 0.0
    matched = sum(1 for phrase in phrases if phrase in normalized_chunk_text)
    return matched / len(phrases)


def _tokenize(text: str) -> set[str]:
    """Extract normalized lexical tokens from mixed-language text."""

    if not text.strip():
        return set()

    tokens: set[str] = set()
    lowered = text.lower()
    for raw_token in KnowledgeRetriever._ENGLISH_TOKEN_PATTERN.findall(lowered):
        token = _normalize_ascii_token(raw_token)
        if token:
            tokens.add(token)
        parts = re.split(r"[^a-z0-9]+", raw_token)
        for part in parts:
            normalized = _normalize_ascii_token(part)
            if normalized:
                tokens.add(normalized)
    for raw_token in KnowledgeRetriever._CJK_TOKEN_PATTERN.findall(text):
        tokens.update(_expand_cjk_token(raw_token))
    return tokens


def _ordered_tokens(text: str) -> list[str]:
    """Extract normalized tokens while preserving first-seen order."""

    ordered: list[str] = []
    seen: set[str] = set()
    lowered = text.lower()
    for raw_token in KnowledgeRetriever._ENGLISH_TOKEN_PATTERN.findall(lowered):
        for candidate in (raw_token, *re.split(r"[^a-z0-9]+", raw_token)):
            normalized = _normalize_ascii_token(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    for raw_token in KnowledgeRetriever._CJK_TOKEN_PATTERN.findall(text):
        for normalized in sorted(_expand_cjk_token(raw_token)):
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _normalize_ascii_token(token: str) -> str:
    """Normalize one ASCII token for lexical scoring."""

    normalized = token.strip("._-").lower()
    if len(normalized) <= 1:
        return ""
    if normalized in KnowledgeRetriever._STOPWORDS:
        return ""
    if normalized.endswith("ies") and len(normalized) > 4:
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("ing") and len(normalized) > 5:
        normalized = normalized[:-3]
    elif normalized.endswith("es") and len(normalized) > 4:
        normalized = normalized[:-2]
    elif normalized.endswith("s") and len(normalized) > 3:
        normalized = normalized[:-1]
    if normalized in KnowledgeRetriever._STOPWORDS or len(normalized) <= 1:
        return ""
    return normalized


def _expand_cjk_token(token: str) -> set[str]:
    """Expand one contiguous CJK span into single chars and bigrams."""

    items = {character for character in token if character.strip()}
    items.update(
        token[index : index + 2]
        for index in range(len(token) - 1)
        if token[index : index + 2].strip()
    )
    return items


def _normalized_text(text: str) -> str:
    """Return a normalized token string for phrase checks."""

    return " ".join(sorted(_tokenize(text)))
