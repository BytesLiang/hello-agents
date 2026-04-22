"""Input validation and normalization for knowledge QA questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from hello_agents.apps.knowledge_qa.llm_utils import (
    SupportsChat,
    load_json_object,
    token_usage_from_response,
)
from hello_agents.apps.knowledge_qa.models import TokenUsage
from hello_agents.llm.types import LLMMessage

_INPUT_CHECK_SYSTEM_PROMPT = (
    "You normalize user questions for a document-grounded QA agent. "
    "Preserve the original intent, filenames, and language. "
    "Remove accidental pasted trace fields, duplicated fragments, dangling quotes, "
    "and formatting noise. "
    "Return valid JSON only."
)

_TRACE_FRAGMENT_PATTERN = re.compile(
    r'["\']?\s*,?\s*["\']?(rewritten_query|question_type|plan_summary)["\']?\s*:\s*["\'].*$',
    re.IGNORECASE,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True, frozen=True)
class InputCheckResult:
    """Capture the normalized user question before planning starts."""

    original_question: str
    normalized_question: str
    is_valid: bool
    issues: tuple[str, ...] = ()
    reason: str | None = None
    used_llm: bool = False
    token_usage: TokenUsage = TokenUsage()


class QuestionInputChecker:
    """Normalize raw user questions before classification and retrieval."""

    def __init__(self, llm: SupportsChat | None = None) -> None:
        """Store the optional LLM used for cleanup and validation."""

        self._llm = llm

    def check(self, question: str) -> InputCheckResult:
        """Return a normalized question and any detected quality issues."""

        heuristic_result = _heuristic_check(question)
        if self._llm is None or not heuristic_result.normalized_question:
            return heuristic_result

        response = self._llm.chat(
            [
                LLMMessage(role="system", content=_INPUT_CHECK_SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=(
                        "Normalize the question below for a knowledge QA agent.\n"
                        "Return JSON with this exact shape:\n"
                        "{\n"
                        '  "normalized_question": "string",\n'
                        '  "is_valid": true,\n'
                        '  "issues": ["string"],\n'
                        '  "reason": null\n'
                        "}\n\n"
                        f"Original question:\n{question.strip()}\n\n"
                        "Heuristic normalization candidate:\n"
                        f"{heuristic_result.normalized_question}"
                    ),
                ),
            ],
            temperature=0,
            max_tokens=200,
        )
        payload = load_json_object(response.content)
        if payload is None:
            return heuristic_result

        normalized_question = str(
            payload.get("normalized_question", heuristic_result.normalized_question)
        ).strip()
        is_valid = payload.get("is_valid")
        raw_issues = payload.get("issues", ())
        reason = payload.get("reason")

        issues = _normalize_issues(raw_issues)
        if not normalized_question:
            normalized_question = heuristic_result.normalized_question
        return InputCheckResult(
            original_question=question,
            normalized_question=normalized_question,
            is_valid=bool(is_valid) if isinstance(is_valid, bool) else True,
            issues=issues or heuristic_result.issues,
            reason=reason if isinstance(reason, str) and reason.strip() else None,
            used_llm=True,
            token_usage=token_usage_from_response(response),
        )


def _heuristic_check(question: str) -> InputCheckResult:
    """Apply lightweight deterministic cleanup before optional LLM review."""

    normalized = question.strip()
    issues: list[str] = []

    cleaned = _TRACE_FRAGMENT_PATTERN.sub("", normalized)
    if cleaned != normalized:
        issues.append("Removed pasted trace metadata from the question.")
        normalized = cleaned

    normalized = normalized.strip(" \t\r\n\"'")
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()

    if not normalized:
        return InputCheckResult(
            original_question=question,
            normalized_question="",
            is_valid=False,
            issues=tuple(issues),
            reason="empty_question",
        )

    return InputCheckResult(
        original_question=question,
        normalized_question=normalized,
        is_valid=True,
        issues=tuple(issues),
    )


def _normalize_issues(raw_issues: object) -> tuple[str, ...]:
    """Normalize issue strings returned by the model."""

    if not isinstance(raw_issues, list):
        return ()
    normalized: list[str] = []
    for item in raw_issues:
        if not isinstance(item, str):
            continue
        issue = item.strip()
        if issue:
            normalized.append(issue)
    return tuple(normalized)
