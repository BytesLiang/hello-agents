"""Evidence scoring helpers for the knowledge QA workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from hello_agents.apps.knowledge_qa.classifier import (
    QuestionClassification,
    QuestionType,
)
from hello_agents.apps.knowledge_qa.llm_utils import (
    SupportsChat,
    load_json_object,
    token_usage_from_response,
)
from hello_agents.apps.knowledge_qa.models import RetrievedChunk, TokenUsage
from hello_agents.llm.types import LLMMessage

_SCORER_SYSTEM_PROMPT = (
    "You assess whether retrieved evidence is strong enough for a grounded "
    "knowledge QA agent. You may request a rewritten retrieval query when the "
    "current evidence is weak or off-target. Return valid JSON only."
)


class EvidenceStatus(StrEnum):
    """Represent the next action chosen after retrieval."""

    SUFFICIENT = "sufficient"
    NEEDS_REWRITE = "needs_rewrite"
    NEEDS_DOCUMENT_INSPECTION = "needs_document_inspection"
    INSUFFICIENT = "insufficient"


@dataclass(slots=True, frozen=True)
class EvidenceAssessment:
    """Capture the evidence quality and the next workflow action."""

    status: EvidenceStatus
    score: float
    reason: str
    failure_mode: str | None = None


@dataclass(slots=True, frozen=True)
class EvidenceDecision:
    """Bundle the assessment with optional rewrite guidance."""

    assessment: EvidenceAssessment
    rewritten_query: str = ""
    used_llm: bool = False
    token_usage: TokenUsage = TokenUsage()


class EvidenceScorer:
    """Assess whether retrieved evidence is enough for the next step."""

    def __init__(self, llm: SupportsChat | None = None) -> None:
        """Store the optional LLM used for evidence assessment."""

        self._llm = llm

    def assess(
        self,
        *,
        question: str = "",
        classification: QuestionClassification,
        chunks: tuple[RetrievedChunk, ...],
        round_index: int,
        max_rounds: int,
    ) -> EvidenceAssessment:
        """Return the next retrieval decision for the provided evidence."""

        return self.assess_with_trace(
            question=question,
            classification=classification,
            chunks=chunks,
            round_index=round_index,
            max_rounds=max_rounds,
        ).assessment

    def assess_with_trace(
        self,
        *,
        question: str,
        classification: QuestionClassification,
        chunks: tuple[RetrievedChunk, ...],
        round_index: int,
        max_rounds: int,
    ) -> EvidenceDecision:
        """Return the evidence assessment plus token usage metadata."""

        heuristic = _assess_heuristically(
            classification=classification,
            chunks=chunks,
            round_index=round_index,
            max_rounds=max_rounds,
        )
        if self._llm is None or not question.strip():
            return EvidenceDecision(assessment=heuristic)

        response = self._llm.chat(
            [
                LLMMessage(role="system", content=_SCORER_SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=_build_scoring_prompt(
                        question=question,
                        classification=classification,
                        chunks=chunks,
                        round_index=round_index,
                        max_rounds=max_rounds,
                        heuristic=heuristic,
                    ),
                ),
            ],
            temperature=0,
            max_tokens=260,
        )
        payload = load_json_object(response.content)
        if payload is None:
            return EvidenceDecision(assessment=heuristic)

        assessment = _assessment_from_payload(payload, fallback=heuristic)
        raw_rewritten_query = payload.get("rewritten_query", "")
        return EvidenceDecision(
            assessment=assessment,
            rewritten_query=(
                raw_rewritten_query.strip()
                if isinstance(raw_rewritten_query, str)
                else ""
            ),
            used_llm=True,
            token_usage=token_usage_from_response(response),
        )


def _assess_heuristically(
    *,
    classification: QuestionClassification,
    chunks: tuple[RetrievedChunk, ...],
    round_index: int,
    max_rounds: int,
) -> EvidenceAssessment:
    """Return the deterministic fallback evidence assessment."""

    if not chunks:
        return EvidenceAssessment(
            status=(
                EvidenceStatus.NEEDS_REWRITE
                if round_index < max_rounds
                else EvidenceStatus.INSUFFICIENT
            ),
            score=0.0,
            reason="No retrieval hits were found for the current query.",
            failure_mode="no_hits",
        )

    average_score = sum(max(chunk.score, 0.0) for chunk in chunks) / len(chunks)
    if classification.question_type is QuestionType.FACT_LOOKUP:
        matched_target = _has_target_file(chunks, classification.target_files)
        if classification.target_files and not matched_target:
            return EvidenceAssessment(
                status=(
                    EvidenceStatus.NEEDS_REWRITE
                    if round_index < max_rounds
                    else EvidenceStatus.INSUFFICIENT
                ),
                score=min(0.49, average_score),
                reason="Retrieved evidence does not yet include the target document.",
                failure_mode="wrong_document",
            )
        return EvidenceAssessment(
            status=EvidenceStatus.SUFFICIENT,
            score=min(1.0, average_score or len(chunks) / 4),
            reason="Retrieved chunks are sufficient for direct answer generation.",
        )

    matched_target = _has_target_file(chunks, classification.target_files)
    if matched_target:
        return EvidenceAssessment(
            status=EvidenceStatus.NEEDS_DOCUMENT_INSPECTION,
            score=min(1.0, max(average_score, 0.6)),
            reason="Target document was located and can now be inspected directly.",
            failure_mode="needs_document_inspection",
        )

    return EvidenceAssessment(
        status=(
            EvidenceStatus.NEEDS_REWRITE
            if round_index < max_rounds
            else EvidenceStatus.INSUFFICIENT
        ),
        score=min(0.49, average_score),
        reason="Retrieved evidence does not yet include the target document.",
        failure_mode="wrong_document",
    )


def _classification_payload(
    classification: QuestionClassification,
) -> dict[str, object]:
    """Serialize a classification for evidence prompts."""

    return {
        "question_type": classification.question_type.value,
        "target_files": list(classification.target_files),
        "needs_document_inspection": classification.needs_document_inspection,
        "needs_multi_step": classification.needs_multi_step,
        "reason": classification.reason,
    }


def _assessment_payload(assessment: EvidenceAssessment) -> dict[str, object]:
    """Serialize one evidence assessment."""

    return {
        "status": assessment.status.value,
        "score": assessment.score,
        "reason": assessment.reason,
        "failure_mode": assessment.failure_mode,
    }


def _assessment_from_payload(
    payload: dict[str, object],
    *,
    fallback: EvidenceAssessment,
) -> EvidenceAssessment:
    """Parse one evidence assessment from model output."""

    raw_status = payload.get("status", fallback.status.value)
    try:
        status = EvidenceStatus(str(raw_status))
    except ValueError:
        status = fallback.status

    raw_score = payload.get("score", fallback.score)
    raw_reason = payload.get("reason")
    raw_failure_mode = payload.get("failure_mode")

    return EvidenceAssessment(
        status=status,
        score=_normalize_score(raw_score, fallback=fallback.score),
        reason=(
            raw_reason
            if isinstance(raw_reason, str) and raw_reason
            else fallback.reason
        ),
        failure_mode=(
            raw_failure_mode
            if isinstance(raw_failure_mode, str) and raw_failure_mode
            else fallback.failure_mode
        ),
    )


def _normalize_score(raw_score: object, *, fallback: float) -> float:
    """Clamp score values returned by the model."""

    if isinstance(raw_score, bool):
        return fallback
    if isinstance(raw_score, (int, float)):
        return min(1.0, max(0.0, float(raw_score)))
    return fallback


def _render_chunks(chunks: tuple[RetrievedChunk, ...]) -> str:
    """Render retrieval evidence for the scoring prompt."""

    if not chunks:
        return "No chunks retrieved."

    lines: list[str] = []
    for index, chunk in enumerate(chunks[:5], start=1):
        heading = f" ({chunk.heading_path})" if chunk.heading_path else ""
        snippet = " ".join(chunk.content.split())
        if len(snippet) > 180:
            snippet = snippet[:180].rstrip() + "..."
        score_text = f"score={chunk.score:.3f}"
        if chunk.rerank_score is not None:
            score_text += f" rerank={chunk.rerank_score:.3f}"
        lines.append(f"[{index}] {chunk.source}{heading} {score_text} :: {snippet}")
    return "\n".join(lines)


def _build_scoring_prompt(
    *,
    question: str,
    classification: QuestionClassification,
    chunks: tuple[RetrievedChunk, ...],
    round_index: int,
    max_rounds: int,
    heuristic: EvidenceAssessment,
) -> str:
    """Render the LLM prompt used for evidence assessment."""

    classification_payload = json.dumps(
        _classification_payload(classification),
        ensure_ascii=False,
    )
    heuristic_payload = json.dumps(
        _assessment_payload(heuristic),
        ensure_ascii=False,
    )
    return (
        "Assess the evidence for the question below.\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "status": "sufficient",\n'
        '  "score": 0.8,\n'
        '  "reason": "string",\n'
        '  "failure_mode": null,\n'
        '  "rewritten_query": ""\n'
        "}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Classification:\n"
        f"{classification_payload}\n\n"
        f"Round: {round_index} / {max_rounds}\n\n"
        "Retrieved chunks:\n"
        f"{_render_chunks(chunks)}\n\n"
        "Heuristic assessment:\n"
        f"{heuristic_payload}"
    )


def _has_target_file(
    chunks: tuple[RetrievedChunk, ...],
    target_files: tuple[str, ...],
) -> bool:
    """Return whether retrieved chunks include any explicitly targeted file."""

    if not target_files:
        return False
    target_names = {item.lower() for item in target_files}
    return any(Path(chunk.source).name.lower() in target_names for chunk in chunks)
