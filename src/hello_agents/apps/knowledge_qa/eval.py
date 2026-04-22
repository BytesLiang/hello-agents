"""Offline evaluation runner for the knowledge QA application."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from hello_agents.apps.knowledge_qa.models import (
    AnswerResult,
    RunTrace,
    TokenUsage,
)
from hello_agents.apps.knowledge_qa.service import KnowledgeQAService


@dataclass(slots=True, frozen=True)
class EvalCase:
    """Represent one evaluation test case."""

    question: str
    expected_answer: str = ""
    expected_sources: tuple[str, ...] = ()
    kb_id: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class EvalCaseResult:
    """Capture the evaluation result for one test case."""

    question: str
    answer: str
    answered: bool
    reason: str | None
    expected_answer: str
    expected_sources: tuple[str, ...]
    citation_sources: tuple[str, ...]
    source_hit: bool
    latency_ms: int
    token_usage: TokenUsage
    trace_id: str | None
    tags: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class EvalMetrics:
    """Aggregate evaluation metrics across all test cases."""

    total: int = 0
    answered: int = 0
    unanswered: int = 0
    answer_rate: float = 0.0
    source_hit_count: int = 0
    source_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    avg_tokens_per_case: float = 0.0


@dataclass(slots=True, frozen=True)
class EvalReport:
    """Capture the full evaluation report."""

    metrics: EvalMetrics
    results: tuple[EvalCaseResult, ...]
    dataset_path: str = ""


class SupportsAsking(Protocol):
    """Protocol for services that can answer questions."""

    def ask(self, question: str, *, kb_id: str | None = None) -> AnswerResult: ...


def load_dataset(path: Path) -> tuple[EvalCase, ...]:
    """Load evaluation cases from a JSON or JSONL file."""

    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        items = json.loads(text)

    cases: list[EvalCase] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = item.get("question", "")
        if not question:
            continue
        raw_sources = item.get("expected_sources", [])
        raw_tags = item.get("tags", [])
        cases.append(
            EvalCase(
                question=str(question),
                expected_answer=str(item.get("expected_answer", "")),
                expected_sources=tuple(
                    str(source) for source in raw_sources if isinstance(source, str)
                ),
                kb_id=item.get("kb_id") if isinstance(item.get("kb_id"), str) else None,
                tags=tuple(str(tag) for tag in raw_tags if isinstance(tag, str)),
            )
        )
    return tuple(cases)


def compute_metrics(results: tuple[EvalCaseResult, ...]) -> EvalMetrics:
    """Compute aggregate metrics from individual case results."""

    total = len(results)
    if total == 0:
        return EvalMetrics()

    answered = sum(1 for result in results if result.answered)
    source_hit_count = sum(1 for result in results if result.source_hit)
    total_latency = sum(result.latency_ms for result in results)
    total_prompt_tokens = sum(result.token_usage.prompt_tokens for result in results)
    total_completion_tokens = sum(
        result.token_usage.completion_tokens for result in results
    )
    total_tokens = sum(result.token_usage.total_tokens for result in results)

    return EvalMetrics(
        total=total,
        answered=answered,
        unanswered=total - answered,
        answer_rate=round(answered / total, 4),
        source_hit_count=source_hit_count,
        source_hit_rate=round(source_hit_count / total, 4),
        avg_latency_ms=round(total_latency / total, 1),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        avg_tokens_per_case=round(total_tokens / total, 1),
    )


def _check_source_hit(
    citation_sources: tuple[str, ...],
    expected_sources: tuple[str, ...],
) -> bool:
    """Return whether any expected source appears in the citation sources."""

    if not expected_sources:
        return True
    if not citation_sources:
        return False
    expected_lower = {source.lower() for source in expected_sources}
    for source in citation_sources:
        source_lower = source.lower()
        for expected in expected_lower:
            if expected in source_lower or source_lower in expected:
                return True
    return False


class EvalRunner:
    """Run evaluation cases against a knowledge QA service."""

    def __init__(self, service: SupportsAsking) -> None:
        """Store the service used for answering evaluation questions."""

        self._service = service

    def run(
        self,
        cases: tuple[EvalCase, ...],
        *,
        dataset_path: str = "",
    ) -> EvalReport:
        """Execute all evaluation cases and return a full report."""

        results: list[EvalCaseResult] = []
        for case in cases:
            result = self._run_case(case)
            results.append(result)
        metrics = compute_metrics(tuple(results))
        return EvalReport(
            metrics=metrics,
            results=tuple(results),
            dataset_path=dataset_path,
        )

    def _run_case(self, case: EvalCase) -> EvalCaseResult:
        """Execute one evaluation case and capture the result."""

        try:
            answer_result = self._service.ask(
                case.question,
                kb_id=case.kb_id,
            )
            citation_sources = tuple(
                citation.source for citation in answer_result.citations
            )
            source_hit = _check_source_hit(
                citation_sources, case.expected_sources
            )
            return EvalCaseResult(
                question=case.question,
                answer=answer_result.answer,
                answered=answer_result.answered,
                reason=answer_result.reason,
                expected_answer=case.expected_answer,
                expected_sources=case.expected_sources,
                citation_sources=citation_sources,
                source_hit=source_hit,
                latency_ms=0,
                token_usage=TokenUsage(),
                trace_id=answer_result.trace_id,
                tags=case.tags,
            )
        except Exception as exc:
            return EvalCaseResult(
                question=case.question,
                answer="",
                answered=False,
                reason=f"eval_error: {exc}",
                expected_answer=case.expected_answer,
                expected_sources=case.expected_sources,
                citation_sources=(),
                source_hit=False,
                latency_ms=0,
                token_usage=TokenUsage(),
                trace_id=None,
                tags=case.tags,
            )


class TraceEvalRunner:
    """Evaluate answers using existing RunTrace records."""

    def evaluate_traces(
        self,
        cases: tuple[EvalCase, ...],
        traces: tuple[RunTrace, ...],
    ) -> EvalReport:
        """Match evaluation cases to traces by question text and compute metrics."""

        trace_by_question: dict[str, RunTrace] = {}
        for trace in traces:
            normalized = trace.question.strip().lower()
            trace_by_question[normalized] = trace

        results: list[EvalCaseResult] = []
        for case in cases:
            normalized = case.question.strip().lower()
            trace = trace_by_question.get(normalized)
            if trace is None:
                results.append(
                    EvalCaseResult(
                        question=case.question,
                        answer="",
                        answered=False,
                        reason="no_matching_trace",
                        expected_answer=case.expected_answer,
                        expected_sources=case.expected_sources,
                        citation_sources=(),
                        source_hit=False,
                        latency_ms=0,
                        token_usage=TokenUsage(),
                        trace_id=None,
                        tags=case.tags,
                    )
                )
                continue

            citation_sources = tuple(
                citation.source for citation in trace.citations
            )
            source_hit = _check_source_hit(
                citation_sources, case.expected_sources
            )
            results.append(
                EvalCaseResult(
                    question=case.question,
                    answer=trace.answer,
                    answered=trace.answered,
                    reason=trace.reason,
                    expected_answer=case.expected_answer,
                    expected_sources=case.expected_sources,
                    citation_sources=citation_sources,
                    source_hit=source_hit,
                    latency_ms=trace.latency_ms,
                    token_usage=trace.token_usage,
                    trace_id=trace.trace_id,
                    tags=case.tags,
                )
            )

        metrics = compute_metrics(tuple(results))
        return EvalReport(metrics=metrics, results=tuple(results))


def format_report(report: EvalReport) -> str:
    """Format an evaluation report as a human-readable string."""

    lines: list[str] = []
    metrics = report.metrics

    lines.append("=" * 60)
    lines.append("Knowledge QA Evaluation Report")
    if report.dataset_path:
        lines.append(f"Dataset: {report.dataset_path}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Summary Metrics")
    lines.append("-" * 40)
    lines.append(f"  Total cases:        {metrics.total}")
    lines.append(f"  Answered:           {metrics.answered}")
    lines.append(f"  Unanswered:         {metrics.unanswered}")
    lines.append(f"  Answer rate:        {metrics.answer_rate:.1%}")
    lines.append(f"  Source hit count:   {metrics.source_hit_count}")
    lines.append(f"  Source hit rate:    {metrics.source_hit_rate:.1%}")
    lines.append(f"  Avg latency:        {metrics.avg_latency_ms:.0f} ms")
    lines.append(f"  Total tokens:       {metrics.total_tokens}")
    lines.append(f"  Avg tokens/case:    {metrics.avg_tokens_per_case:.0f}")
    lines.append("")
    lines.append("Per-Case Results")
    lines.append("-" * 40)

    for index, result in enumerate(report.results, start=1):
        status = "ANSWERED" if result.answered else "REFUSED"
        hit = "HIT" if result.source_hit else "MISS"
        lines.append(f"  [{index}] {status} | Source: {hit} | {result.question[:60]}")
        if result.reason and not result.answered:
            lines.append(f"       Reason: {result.reason}")
        if result.citation_sources:
            sources = ", ".join(result.citation_sources[:3])
            lines.append(f"       Sources: {sources}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def save_report_json(report: EvalReport, path: Path) -> None:
    """Save an evaluation report as a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset_path": report.dataset_path,
        "metrics": {
            "total": report.metrics.total,
            "answered": report.metrics.answered,
            "unanswered": report.metrics.unanswered,
            "answer_rate": report.metrics.answer_rate,
            "source_hit_count": report.metrics.source_hit_count,
            "source_hit_rate": report.metrics.source_hit_rate,
            "avg_latency_ms": report.metrics.avg_latency_ms,
            "total_prompt_tokens": report.metrics.total_prompt_tokens,
            "total_completion_tokens": report.metrics.total_completion_tokens,
            "total_tokens": report.metrics.total_tokens,
            "avg_tokens_per_case": report.metrics.avg_tokens_per_case,
        },
        "results": [
            {
                "question": result.question,
                "answer": result.answer,
                "answered": result.answered,
                "reason": result.reason,
                "expected_answer": result.expected_answer,
                "expected_sources": list(result.expected_sources),
                "citation_sources": list(result.citation_sources),
                "source_hit": result.source_hit,
                "latency_ms": result.latency_ms,
                "token_usage": {
                    "prompt_tokens": result.token_usage.prompt_tokens,
                    "completion_tokens": result.token_usage.completion_tokens,
                    "total_tokens": result.token_usage.total_tokens,
                },
                "trace_id": result.trace_id,
                "tags": list(result.tags),
            }
            for result in report.results
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
