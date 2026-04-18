"""Execution trace stores for the knowledge QA application."""

from __future__ import annotations

import json
from pathlib import Path

from hello_agents.apps.knowledge_qa.models import (
    Citation,
    RetrievedChunk,
    RunTrace,
    TokenUsage,
)


class JsonlRunTraceStore:
    """Persist knowledge QA traces as local JSON Lines."""

    def __init__(self, path: Path) -> None:
        """Store the trace file location."""

        self._path = path

    def append(self, trace: RunTrace) -> None:
        """Append one trace record to the JSONL file."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_trace_to_dict(trace), ensure_ascii=False) + "\n")

    def list_recent(self, *, limit: int = 20) -> tuple[RunTrace, ...]:
        """Return the most recent trace records up to the requested limit."""

        if limit <= 0 or not self._path.exists():
            return ()
        lines = self._path.read_text(encoding="utf-8").splitlines()
        traces: list[RunTrace] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            traces.append(_trace_from_dict(raw))
            if len(traces) >= limit:
                break
        return tuple(traces)


def _trace_to_dict(trace: RunTrace) -> dict[str, object]:
    """Serialize one trace to a JSON-compatible dictionary."""

    return {
        "trace_id": trace.trace_id,
        "question": trace.question,
        "rewritten_query": trace.rewritten_query,
        "retrieved_chunks": [_chunk_to_dict(chunk) for chunk in trace.retrieved_chunks],
        "selected_chunks": [_chunk_to_dict(chunk) for chunk in trace.selected_chunks],
        "rendered_prompt": trace.rendered_prompt,
        "answer": trace.answer,
        "citations": [_citation_to_dict(citation) for citation in trace.citations],
        "answered": trace.answered,
        "reason": trace.reason,
        "latency_ms": trace.latency_ms,
        "token_usage": _token_usage_to_dict(trace.token_usage),
        "created_at": trace.created_at,
    }


def _trace_from_dict(payload: dict[str, object]) -> RunTrace:
    """Deserialize one trace record from a JSON payload."""

    raw_retrieved_chunks = payload.get("retrieved_chunks", ())
    raw_selected_chunks = payload.get("selected_chunks", ())
    raw_citations = payload.get("citations", ())
    raw_reason = payload.get("reason")

    return RunTrace(
        trace_id=str(payload["trace_id"]),
        question=str(payload["question"]),
        rewritten_query=str(payload.get("rewritten_query", "")),
        retrieved_chunks=tuple(
            _chunk_from_dict(item)
            for item in raw_retrieved_chunks
            if isinstance(item, dict)
        )
        if isinstance(raw_retrieved_chunks, list)
        else (),
        selected_chunks=tuple(
            _chunk_from_dict(item)
            for item in raw_selected_chunks
            if isinstance(item, dict)
        )
        if isinstance(raw_selected_chunks, list)
        else (),
        rendered_prompt=str(payload.get("rendered_prompt", "")),
        answer=str(payload.get("answer", "")),
        citations=tuple(
            _citation_from_dict(item)
            for item in raw_citations
            if isinstance(item, dict)
        )
        if isinstance(raw_citations, list)
        else (),
        answered=bool(payload.get("answered", False)),
        reason=raw_reason if isinstance(raw_reason, str) else None,
        latency_ms=_as_int(payload.get("latency_ms", 0)),
        token_usage=_token_usage_from_dict(payload.get("token_usage", {})),
        created_at=str(payload.get("created_at", "")),
    )


def _chunk_to_dict(chunk: RetrievedChunk) -> dict[str, object]:
    """Serialize one retrieved chunk."""

    return {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "heading_path": chunk.heading_path,
        "content": chunk.content,
        "score": chunk.score,
        "rerank_score": chunk.rerank_score,
        "metadata": chunk.metadata,
    }


def _chunk_from_dict(payload: dict[str, object]) -> RetrievedChunk:
    """Deserialize one retrieved chunk."""

    metadata = payload.get("metadata", {})
    raw_rerank_score = payload.get("rerank_score")
    return RetrievedChunk(
        chunk_id=str(payload.get("chunk_id", "")),
        source=str(payload.get("source", "")),
        heading_path=str(payload.get("heading_path", "")),
        content=str(payload.get("content", "")),
        score=_as_float(payload.get("score", 0.0)),
        rerank_score=(
            _as_float(raw_rerank_score)
            if isinstance(raw_rerank_score, (int, float, str))
            else None
        ),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _citation_to_dict(citation: Citation) -> dict[str, object]:
    """Serialize one citation."""

    return {
        "index": citation.index,
        "source": citation.source,
        "snippet": citation.snippet,
        "chunk_id": citation.chunk_id,
    }


def _citation_from_dict(payload: dict[str, object]) -> Citation:
    """Deserialize one citation."""

    return Citation(
        index=_as_int(payload.get("index", 0)),
        source=str(payload.get("source", "")),
        snippet=str(payload.get("snippet", "")),
        chunk_id=str(payload.get("chunk_id", "")),
    )


def _token_usage_to_dict(token_usage: TokenUsage) -> dict[str, int]:
    """Serialize token usage."""

    return {
        "prompt_tokens": token_usage.prompt_tokens,
        "completion_tokens": token_usage.completion_tokens,
        "total_tokens": token_usage.total_tokens,
    }


def _token_usage_from_dict(payload: object) -> TokenUsage:
    """Deserialize token usage."""

    if not isinstance(payload, dict):
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=_as_int(payload.get("prompt_tokens", 0)),
        completion_tokens=_as_int(payload.get("completion_tokens", 0)),
        total_tokens=_as_int(payload.get("total_tokens", 0)),
    )


def _as_int(value: object) -> int:
    """Convert a JSON payload value to an integer when possible."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return 0


def _as_float(value: object) -> float:
    """Convert a JSON payload value to a float when possible."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return 0.0
