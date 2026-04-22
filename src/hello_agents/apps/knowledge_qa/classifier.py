"""Question classification helpers for the knowledge QA workflow."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from hello_agents.apps.knowledge_qa.llm_utils import (
    SupportsChat,
    load_json_object,
    token_usage_from_response,
)
from hello_agents.apps.knowledge_qa.models import TokenUsage
from hello_agents.llm.types import LLMMessage

_CLASSIFIER_SYSTEM_PROMPT = (
    "You classify document-grounded questions for a knowledge QA agent. "
    "Choose the execution strategy that will best answer the question. "
    "Return valid JSON only."
)


class QuestionType(StrEnum):
    """Describe the execution strategy required for one question."""

    FACT_LOOKUP = "fact_lookup"
    DOCUMENT_STRUCTURE = "document_structure"
    DOCUMENT_STATISTIC = "document_statistic"
    AMBIGUOUS = "ambiguous"


@dataclass(slots=True, frozen=True)
class QuestionClassification:
    """Capture the routing decision for one knowledge QA question."""

    question_type: QuestionType
    target_files: tuple[str, ...] = ()
    needs_document_inspection: bool = False
    needs_multi_step: bool = False
    reason: str = ""


@dataclass(slots=True, frozen=True)
class ClassificationDecision:
    """Bundle the classification with execution metadata."""

    classification: QuestionClassification
    used_llm: bool = False
    token_usage: TokenUsage = TokenUsage()


class QuestionClassifier:
    """Classify user questions into retrieval and analysis workflows."""

    _FILENAME_PATTERN = re.compile(
        r"(?<![A-Za-z0-9_.-])"
        r"([A-Za-z0-9_.-]+\.[A-Za-z][A-Za-z0-9]{0,15})"
        r"(?![A-Za-z0-9_.-])"
    )
    _STRUCTURE_PATTERN = re.compile(
        r"(分几段|几段|几节|多少节|多少段|多少部分|多少个部分|"
        r"how many sections|how many headings|headings|section count)",
        re.IGNORECASE,
    )
    _STATISTIC_PATTERN = re.compile(
        r"(多少词|多少字|字数|词数|how many words?|word count|count words?)",
        re.IGNORECASE,
    )

    def __init__(self, llm: SupportsChat | None = None) -> None:
        """Store the optional LLM used for semantic classification."""

        self._llm = llm

    def classify(self, question: str) -> QuestionClassification:
        """Return the best-effort workflow classification for the question."""

        return self.classify_with_trace(question).classification

    def classify_with_trace(self, question: str) -> ClassificationDecision:
        """Return the routing decision plus token usage metadata."""

        heuristic = _classify_heuristically(question)
        normalized = question.strip()
        if self._llm is None or not normalized:
            return ClassificationDecision(classification=heuristic)

        response = self._llm.chat(
            [
                LLMMessage(role="system", content=_CLASSIFIER_SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=_build_classification_prompt(
                        question=normalized,
                        heuristic=heuristic,
                    ),
                ),
            ],
            temperature=0,
            max_tokens=220,
        )
        payload = load_json_object(response.content)
        if payload is None:
            return ClassificationDecision(classification=heuristic)

        classification = _classification_from_payload(payload, fallback=heuristic)
        return ClassificationDecision(
            classification=classification,
            used_llm=True,
            token_usage=token_usage_from_response(response),
        )


def _classify_heuristically(question: str) -> QuestionClassification:
    """Return the deterministic fallback classification."""

    normalized = question.strip()
    target_files = _extract_target_files(
        normalized, QuestionClassifier._FILENAME_PATTERN
    )
    if QuestionClassifier._STATISTIC_PATTERN.search(normalized):
        return QuestionClassification(
            question_type=QuestionType.DOCUMENT_STATISTIC,
            target_files=target_files,
            needs_document_inspection=True,
            needs_multi_step=True,
            reason="The question asks for document statistics.",
        )
    if QuestionClassifier._STRUCTURE_PATTERN.search(normalized):
        return QuestionClassification(
            question_type=QuestionType.DOCUMENT_STRUCTURE,
            target_files=target_files,
            needs_document_inspection=True,
            needs_multi_step=True,
            reason="The question asks about document structure.",
        )
    if not normalized:
        return QuestionClassification(
            question_type=QuestionType.AMBIGUOUS,
            target_files=(),
            needs_document_inspection=False,
            needs_multi_step=False,
            reason="The question is empty after normalization.",
        )
    return QuestionClassification(
        question_type=QuestionType.FACT_LOOKUP,
        target_files=target_files,
        needs_document_inspection=False,
        needs_multi_step=bool(target_files),
        reason="The question can be handled as grounded retrieval.",
    )


def _classification_to_payload(
    classification: QuestionClassification,
) -> dict[str, object]:
    """Serialize a classification for prompts or traces."""

    return {
        "question_type": classification.question_type.value,
        "target_files": list(classification.target_files),
        "needs_document_inspection": classification.needs_document_inspection,
        "needs_multi_step": classification.needs_multi_step,
        "reason": classification.reason,
    }


def _classification_from_payload(
    payload: dict[str, object],
    *,
    fallback: QuestionClassification,
) -> QuestionClassification:
    """Parse one classification payload from model output."""

    raw_type = payload.get("question_type", fallback.question_type.value)
    try:
        question_type = QuestionType(str(raw_type))
    except ValueError:
        question_type = fallback.question_type

    raw_target_files = payload.get("target_files", list(fallback.target_files))
    target_files = _normalize_target_files(raw_target_files)
    if not target_files:
        target_files = fallback.target_files

    raw_reason = payload.get("reason")
    raw_needs_document_inspection = payload.get(
        "needs_document_inspection",
        fallback.needs_document_inspection,
    )
    raw_needs_multi_step = payload.get(
        "needs_multi_step",
        fallback.needs_multi_step,
    )

    return QuestionClassification(
        question_type=question_type,
        target_files=target_files,
        needs_document_inspection=(
            raw_needs_document_inspection
            if isinstance(raw_needs_document_inspection, bool)
            else fallback.needs_document_inspection
        ),
        needs_multi_step=(
            raw_needs_multi_step
            if isinstance(raw_needs_multi_step, bool)
            else fallback.needs_multi_step
        ),
        reason=(
            raw_reason
            if isinstance(raw_reason, str) and raw_reason
            else fallback.reason
        ),
    )


def _extract_target_files(
    question: str,
    filename_pattern: re.Pattern[str],
) -> tuple[str, ...]:
    """Extract normalized filenames explicitly referenced in the question."""

    seen: set[str] = set()
    target_files: list[str] = []
    for match in filename_pattern.finditer(question):
        normalized = Path(match.group(1)).name.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        target_files.append(normalized)
    return tuple(target_files)


def _build_classification_prompt(
    *,
    question: str,
    heuristic: QuestionClassification,
) -> str:
    """Render the LLM prompt used for question classification."""

    heuristic_payload = json.dumps(
        _classification_to_payload(heuristic),
        ensure_ascii=False,
    )
    return (
        "Classify the question for a knowledge QA agent.\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "question_type": "fact_lookup",\n'
        '  "target_files": ["01_overview.md"],\n'
        '  "needs_document_inspection": false,\n'
        '  "needs_multi_step": false,\n'
        '  "reason": "string"\n'
        "}\n\n"
        f"Question:\n{question}\n\n"
        "Heuristic classification:\n"
        f"{heuristic_payload}"
    )


def _normalize_target_files(raw_target_files: object) -> tuple[str, ...]:
    """Normalize target filenames returned by the model."""

    if not isinstance(raw_target_files, list):
        return ()

    normalized_files: list[str] = []
    for item in raw_target_files:
        if not isinstance(item, str):
            continue
        normalized = Path(item).name.lower().strip()
        if normalized:
            normalized_files.append(normalized)
    return tuple(normalized_files)
