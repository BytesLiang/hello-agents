"""Query planning helpers for the knowledge QA workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass

from hello_agents.apps.knowledge_qa.classifier import (
    QuestionClassification,
    QuestionType,
)
from hello_agents.apps.knowledge_qa.llm_utils import (
    SupportsChat,
    load_json_object,
    token_usage_from_response,
)
from hello_agents.apps.knowledge_qa.models import TokenUsage
from hello_agents.llm.types import LLMMessage

_PLANNER_SYSTEM_PROMPT = (
    "You plan a bounded retrieval strategy for a grounded knowledge QA agent. "
    "Return valid JSON only."
)


@dataclass(slots=True, frozen=True)
class RetrievalPlan:
    """Describe the retrieval and fallback strategy for one question."""

    primary_queries: tuple[str, ...]
    fallback_queries: tuple[str, ...]
    max_rounds: int
    use_document_inspection_on_failure: bool
    summary: str


@dataclass(slots=True, frozen=True)
class PlanDecision:
    """Bundle the retrieval plan with execution metadata."""

    plan: RetrievalPlan
    used_llm: bool = False
    token_usage: TokenUsage = TokenUsage()


class QueryPlanner:
    """Build a bounded multi-step retrieval plan from a classification."""

    def __init__(self, llm: SupportsChat | None = None) -> None:
        """Store the optional LLM used for semantic query planning."""

        self._llm = llm

    def plan(
        self,
        question: str,
        classification: QuestionClassification,
    ) -> RetrievalPlan:
        """Return a deterministic retrieval plan for one classified question."""

        return self.plan_with_trace(question, classification).plan

    def plan_with_trace(
        self,
        question: str,
        classification: QuestionClassification,
    ) -> PlanDecision:
        """Return the retrieval plan plus token usage metadata."""

        heuristic = _plan_heuristically(question, classification)
        normalized = question.strip()
        if self._llm is None or not normalized:
            return PlanDecision(plan=heuristic)

        response = self._llm.chat(
            [
                LLMMessage(role="system", content=_PLANNER_SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=_build_plan_prompt(
                        question=normalized,
                        classification=classification,
                        heuristic=heuristic,
                    ),
                ),
            ],
            temperature=0,
            max_tokens=280,
        )
        payload = load_json_object(response.content)
        if payload is None:
            return PlanDecision(plan=heuristic)

        plan = _plan_from_payload(payload, fallback=heuristic)
        return PlanDecision(
            plan=plan,
            used_llm=True,
            token_usage=token_usage_from_response(response),
        )


def _plan_heuristically(
    question: str,
    classification: QuestionClassification,
) -> RetrievalPlan:
    """Return the deterministic fallback retrieval plan."""

    normalized = question.strip()
    target_file = classification.target_files[0] if classification.target_files else ""
    if classification.question_type is QuestionType.DOCUMENT_STRUCTURE:
        return RetrievalPlan(
            primary_queries=_compact_queries((normalized, target_file)),
            fallback_queries=_compact_queries(
                (
                    f"sections in {target_file}" if target_file else "",
                    f"headings in {target_file}" if target_file else "",
                )
            ),
            max_rounds=2,
            use_document_inspection_on_failure=True,
            summary="Locate the target document, then inspect its structure.",
        )
    if classification.question_type is QuestionType.DOCUMENT_STATISTIC:
        return RetrievalPlan(
            primary_queries=_compact_queries((target_file, normalized)),
            fallback_queries=_compact_queries(
                (f"word count {target_file}" if target_file else "",)
            ),
            max_rounds=2,
            use_document_inspection_on_failure=True,
            summary="Locate the target document, then inspect it for statistics.",
        )
    return RetrievalPlan(
        primary_queries=_compact_queries((normalized,)),
        fallback_queries=(),
        max_rounds=1,
        use_document_inspection_on_failure=False,
        summary="Use direct retrieval for grounded answer generation.",
    )


def _classification_payload(
    classification: QuestionClassification,
) -> dict[str, object]:
    """Serialize a classification for planning prompts."""

    return {
        "question_type": classification.question_type.value,
        "target_files": list(classification.target_files),
        "needs_document_inspection": classification.needs_document_inspection,
        "needs_multi_step": classification.needs_multi_step,
        "reason": classification.reason,
    }


def _plan_payload(plan: RetrievalPlan) -> dict[str, object]:
    """Serialize a retrieval plan for prompts or traces."""

    return {
        "primary_queries": list(plan.primary_queries),
        "fallback_queries": list(plan.fallback_queries),
        "max_rounds": plan.max_rounds,
        "use_document_inspection_on_failure": (plan.use_document_inspection_on_failure),
        "summary": plan.summary,
    }


def _plan_from_payload(
    payload: dict[str, object],
    *,
    fallback: RetrievalPlan,
) -> RetrievalPlan:
    """Parse one retrieval plan from model output."""

    primary_queries = _normalize_queries(
        payload.get("primary_queries"),
        fallback=fallback.primary_queries,
    )
    fallback_queries = _normalize_queries(
        payload.get("fallback_queries"),
        fallback=fallback.fallback_queries,
    )
    raw_max_rounds = payload.get("max_rounds", fallback.max_rounds)
    raw_use_inspection = payload.get(
        "use_document_inspection_on_failure",
        fallback.use_document_inspection_on_failure,
    )
    raw_summary = payload.get("summary")

    return RetrievalPlan(
        primary_queries=primary_queries,
        fallback_queries=fallback_queries,
        max_rounds=max(1, min(4, raw_max_rounds))
        if isinstance(raw_max_rounds, int)
        else fallback.max_rounds,
        use_document_inspection_on_failure=(
            raw_use_inspection
            if isinstance(raw_use_inspection, bool)
            else fallback.use_document_inspection_on_failure
        ),
        summary=(
            raw_summary
            if isinstance(raw_summary, str) and raw_summary
            else fallback.summary
        ),
    )


def _normalize_queries(
    raw_queries: object,
    *,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    """Normalize candidate query strings from model output."""

    if not isinstance(raw_queries, list):
        return fallback
    normalized = _compact_queries(
        tuple(
            item.strip()
            for item in raw_queries
            if isinstance(item, str) and item.strip()
        )
    )
    return normalized or fallback


def _compact_queries(queries: tuple[str, ...]) -> tuple[str, ...]:
    """Drop empty or duplicate query strings while preserving order."""

    seen: set[str] = set()
    compacted: list[str] = []
    for query in queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        compacted.append(normalized)
    return tuple(compacted)


def _build_plan_prompt(
    *,
    question: str,
    classification: QuestionClassification,
    heuristic: RetrievalPlan,
) -> str:
    """Render the LLM prompt used for retrieval planning."""

    classification_payload = json.dumps(
        _classification_payload(classification),
        ensure_ascii=False,
    )
    heuristic_payload = json.dumps(
        _plan_payload(heuristic),
        ensure_ascii=False,
    )
    return (
        "Build a retrieval plan for the question below.\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "primary_queries": ["string"],\n'
        '  "fallback_queries": ["string"],\n'
        '  "max_rounds": 2,\n'
        '  "use_document_inspection_on_failure": false,\n'
        '  "summary": "string"\n'
        "}\n\n"
        f"Question:\n{question}\n\n"
        "Classification:\n"
        f"{classification_payload}\n\n"
        "Heuristic plan:\n"
        f"{heuristic_payload}"
    )
