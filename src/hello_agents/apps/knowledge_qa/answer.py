"""Prompt construction helpers for the knowledge QA application."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from hello_agents.apps.knowledge_qa.models import Citation, RetrievedChunk
from hello_agents.llm.types import LLMMessage

KNOWLEDGE_QA_SYSTEM_PROMPT = (
    "You answer questions using only the supplied knowledge base excerpts. "
    "Return valid JSON only. "
    "Do not wrap the JSON in Markdown or add extra text."
)


@dataclass(slots=True, frozen=True)
class ParsedAnswer:
    """Represent a normalized answer parsed from model output."""

    answer: str
    answered: bool
    reason: str | None = None
    citation_indices: tuple[int, ...] = ()
    used_structured_output: bool = False


def build_answer_messages(
    question: str,
    chunks: Sequence[RetrievedChunk],
) -> list[LLMMessage]:
    """Build the LLM messages used for knowledge QA answering."""

    return [
        LLMMessage(role="system", content=KNOWLEDGE_QA_SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=_build_user_prompt(question, chunks),
        ),
    ]


def build_citations(
    chunks: Sequence[RetrievedChunk],
    *,
    limit: int,
    citation_indices: Sequence[int] | None = None,
) -> tuple[Citation, ...]:
    """Build answer citations from selected retrieval chunks."""

    chosen_chunks = _select_citation_chunks(
        chunks,
        limit=limit,
        citation_indices=citation_indices,
    )
    citations: list[Citation] = []
    for index, chunk in enumerate(chosen_chunks, start=1):
        citations.append(
            Citation(
                index=index,
                source=chunk.source,
                snippet=_truncate_snippet(chunk.content),
                chunk_id=chunk.chunk_id,
            )
        )
    return tuple(citations)


def parse_answer_response(
    content: str,
    *,
    max_citation_index: int,
) -> ParsedAnswer:
    """Parse a model answer from structured JSON or fall back to raw text."""

    normalized = content.strip()
    if not normalized:
        return ParsedAnswer(
            answer="I do not know based on the current knowledge base.",
            answered=False,
            reason="empty_answer",
        )

    payload = _load_json_object(normalized)
    if payload is None:
        return ParsedAnswer(
            answer=normalized,
            answered=True,
            used_structured_output=False,
        )

    answer = payload.get("answer", "")
    answered = payload.get("answered")
    reason = payload.get("reason")
    citation_indices = _normalize_citation_indices(
        payload.get("citation_indices"),
        max_citation_index=max_citation_index,
    )

    parsed_answer = answer if isinstance(answer, str) else ""
    parsed_answered = answered if isinstance(answered, bool) else bool(parsed_answer)
    parsed_reason = reason if isinstance(reason, str) else None

    if not parsed_answer and not parsed_answered:
        parsed_answer = "I do not know based on the current knowledge base."
    if not parsed_answer and parsed_answered:
        parsed_answer = normalized

    return ParsedAnswer(
        answer=parsed_answer,
        answered=parsed_answered,
        reason=parsed_reason,
        citation_indices=citation_indices,
        used_structured_output=True,
    )


def _build_user_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Render the final user prompt for the knowledge QA generation step."""

    context_lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        heading = f" ({chunk.heading_path})" if chunk.heading_path else ""
        context_lines.append(
            f"[{index}] {chunk.source}{heading}\n{chunk.content.strip()}"
        )

    rendered_context = "\n\n".join(context_lines) if context_lines else "No context."
    return (
        "Answer the question using the context below.\n"
        "If the context is insufficient, explicitly say so.\n"
        "Return JSON with the exact shape:\n"
        "{\n"
        '  "answer": "string",\n'
        '  "answered": true,\n'
        '  "reason": null,\n'
        '  "citation_indices": [1]\n'
        "}\n"
        "Rules:\n"
        "- citation_indices must reference the numbered context items.\n"
        "- If evidence is insufficient, set answered to false.\n"
        '- If answered is false, reason should be "insufficient_evidence" or a '
        "short explanation.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Context:\n{rendered_context}"
    )


def _select_citation_chunks(
    chunks: Sequence[RetrievedChunk],
    *,
    limit: int,
    citation_indices: Sequence[int] | None,
) -> tuple[RetrievedChunk, ...]:
    """Select citation chunks by explicit indices or by default ordering."""

    if not citation_indices:
        return tuple(chunks[:limit])

    selected_chunks: list[RetrievedChunk] = []
    seen_indices: set[int] = set()
    for raw_index in citation_indices:
        if raw_index in seen_indices:
            continue
        seen_indices.add(raw_index)
        if raw_index < 1 or raw_index > len(chunks):
            continue
        selected_chunks.append(chunks[raw_index - 1])
        if len(selected_chunks) >= limit:
            break
    return tuple(selected_chunks)


def _load_json_object(content: str) -> dict[str, object] | None:
    """Load a JSON object from plain or fenced model output."""

    candidate = content.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _normalize_citation_indices(
    value: object,
    *,
    max_citation_index: int,
) -> tuple[int, ...]:
    """Normalize citation indices from parsed JSON output."""

    if not isinstance(value, list):
        return ()

    normalized: list[int] = []
    seen: set[int] = set()
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            continue
        if item < 1 or item > max_citation_index or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _truncate_snippet(content: str, *, limit: int = 240) -> str:
    """Trim a chunk to a citation-friendly snippet."""

    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."
