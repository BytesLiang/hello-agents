"""Prompt construction helpers for the knowledge QA application."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hello_agents.apps.knowledge_qa.llm_utils import (
    SupportsChat,
    load_json_object,
    token_usage_from_response,
)
from hello_agents.apps.knowledge_qa.models import Citation, RetrievedChunk, TokenUsage
from hello_agents.llm.types import LLMMessage

KNOWLEDGE_QA_SYSTEM_PROMPT = (
    "You answer questions using only the supplied knowledge base excerpts. "
    "Return valid JSON only. "
    "Do not wrap the JSON in Markdown or add extra text."
)
_CITATION_VALIDATION_SYSTEM_PROMPT = (
    "You verify whether an answer is fully supported by the provided evidence and "
    "whether the cited items are appropriate. Return valid JSON only."
)


@dataclass(slots=True, frozen=True)
class ParsedAnswer:
    """Represent a normalized answer parsed from model output."""

    answer: str
    answered: bool
    reason: str | None = None
    citation_indices: tuple[int, ...] = ()
    used_structured_output: bool = False


@dataclass(slots=True, frozen=True)
class CitationValidationResult:
    """Represent the outcome of answer and citation validation."""

    is_valid: bool
    reason: str | None = None
    citation_indices: tuple[int, ...] = ()
    answered: bool | None = None
    used_llm: bool = False
    token_usage: TokenUsage = TokenUsage()


def build_answer_messages(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    inspection_summary: str | None = None,
    validation_feedback: str | None = None,
) -> list[LLMMessage]:
    """Build the LLM messages used for knowledge QA answering."""

    return [
        LLMMessage(role="system", content=KNOWLEDGE_QA_SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=_build_user_prompt(
                question,
                chunks,
                inspection_summary=inspection_summary,
                validation_feedback=validation_feedback,
            ),
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

    payload = load_json_object(normalized)
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


def validate_citations(
    *,
    question: str,
    parsed_answer: ParsedAnswer,
    chunks: Sequence[RetrievedChunk],
    inspection_summary: str | None = None,
    llm: SupportsChat | None = None,
) -> CitationValidationResult:
    """Validate whether the answer and citations are grounded in evidence."""

    deterministic = _validate_citations_deterministically(
        parsed_answer=parsed_answer,
        chunks=chunks,
        inspection_summary=inspection_summary,
    )
    if not deterministic.is_valid or llm is None or not parsed_answer.answered:
        return deterministic

    response = llm.chat(
        [
            LLMMessage(role="system", content=_CITATION_VALIDATION_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=_build_validation_prompt(
                    question=question,
                    parsed_answer=parsed_answer,
                    chunks=chunks,
                    inspection_summary=inspection_summary,
                ),
            ),
        ],
        temperature=0,
        max_tokens=180,
    )
    payload = load_json_object(response.content)
    if payload is None:
        return deterministic

    raw_is_valid = payload.get("is_valid")
    raw_reason = payload.get("reason")
    raw_answered = payload.get("answered")
    citation_indices = _normalize_citation_indices(
        payload.get("citation_indices"),
        max_citation_index=len(chunks),
    )
    return CitationValidationResult(
        is_valid=(
            raw_is_valid if isinstance(raw_is_valid, bool) else deterministic.is_valid
        ),
        reason=(
            raw_reason
            if isinstance(raw_reason, str) and raw_reason
            else deterministic.reason
        ),
        citation_indices=citation_indices or deterministic.citation_indices,
        answered=(
            raw_answered if isinstance(raw_answered, bool) else deterministic.answered
        ),
        used_llm=True,
        token_usage=token_usage_from_response(response),
    )


def _build_user_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    inspection_summary: str | None,
    validation_feedback: str | None,
) -> str:
    """Render the final user prompt for the knowledge QA generation step."""

    context_lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        heading = f" ({chunk.heading_path})" if chunk.heading_path else ""
        context_lines.append(
            f"[{index}] {chunk.source}{heading}\n{chunk.content.strip()}"
        )

    rendered_context = "\n\n".join(context_lines) if context_lines else "No context."
    rendered_analysis = (
        f"\n\nDocument analysis:\n{inspection_summary.strip()}"
        if inspection_summary and inspection_summary.strip()
        else ""
    )
    rendered_feedback = (
        f"\n\nValidation feedback:\n{validation_feedback.strip()}"
        if validation_feedback and validation_feedback.strip()
        else ""
    )
    return (
        "Answer the question using the context below.\n"
        "Use the retrieved excerpts and any document analysis provided below.\n"
        "If the evidence is insufficient, explicitly say so.\n"
        "Return JSON with the exact shape:\n"
        "{\n"
        '  "answer": "string",\n'
        '  "answered": true,\n'
        '  "reason": null,\n'
        '  "citation_indices": [1]\n'
        "}\n"
        "Rules:\n"
        "- citation_indices must reference the numbered context items.\n"
        "- citation_indices may be [] when the answer relies only on "
        "document analysis.\n"
        "- If evidence is insufficient, set answered to false.\n"
        '- If answered is false, reason should be "insufficient_evidence" or a '
        "short explanation.\n"
        "- Keep the answer concise and grounded in the cited evidence.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Context:\n{rendered_context}{rendered_analysis}{rendered_feedback}"
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


def _validate_citations_deterministically(
    *,
    parsed_answer: ParsedAnswer,
    chunks: Sequence[RetrievedChunk],
    inspection_summary: str | None,
) -> CitationValidationResult:
    """Apply deterministic grounding checks before optional LLM review."""

    if not parsed_answer.answered:
        return CitationValidationResult(
            is_valid=True,
            reason=parsed_answer.reason,
            citation_indices=parsed_answer.citation_indices,
            answered=False,
        )

    if parsed_answer.citation_indices:
        return CitationValidationResult(
            is_valid=True,
            citation_indices=parsed_answer.citation_indices,
            answered=True,
        )

    if inspection_summary and inspection_summary.strip():
        return CitationValidationResult(
            is_valid=True,
            reason="Answer relies on document inspection output.",
            citation_indices=(),
            answered=True,
        )

    if not chunks:
        return CitationValidationResult(
            is_valid=False,
            reason="missing_evidence",
            citation_indices=(),
            answered=False,
        )

    return CitationValidationResult(
        is_valid=False,
        reason="missing_citations",
        citation_indices=(),
        answered=True,
    )


def _render_validation_context(
    chunks: Sequence[RetrievedChunk],
    *,
    inspection_summary: str | None,
) -> str:
    """Render the evidence bundle for citation validation."""

    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        heading = f" ({chunk.heading_path})" if chunk.heading_path else ""
        parts.append(f"[{index}] {chunk.source}{heading}\n{chunk.content.strip()}")
    if inspection_summary and inspection_summary.strip():
        parts.append(f"Document analysis:\n{inspection_summary.strip()}")
    return "\n\n".join(parts) if parts else "No evidence."


def _build_validation_prompt(
    *,
    question: str,
    parsed_answer: ParsedAnswer,
    chunks: Sequence[RetrievedChunk],
    inspection_summary: str | None,
) -> str:
    """Render the citation validation prompt."""

    evidence = _render_validation_context(
        chunks,
        inspection_summary=inspection_summary,
    )
    return (
        "Validate whether the answer is fully supported by the cited evidence.\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "is_valid": true,\n'
        '  "reason": null,\n'
        '  "citation_indices": [1],\n'
        '  "answered": true\n'
        "}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Answer:\n{parsed_answer.answer}\n\n"
        "Citation indices from the answer:\n"
        f"{list(parsed_answer.citation_indices)}\n\n"
        f"Evidence:\n{evidence}"
    )
