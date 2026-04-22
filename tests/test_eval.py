"""Tests for the knowledge QA evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hello_agents.apps.knowledge_qa.eval import (
    EvalCase,
    EvalCaseResult,
    EvalMetrics,
    EvalReport,
    EvalRunner,
    TraceEvalRunner,
    _check_source_hit,
    compute_metrics,
    format_report,
    load_dataset,
    save_report_json,
)
from hello_agents.apps.knowledge_qa.models import (
    AnswerResult,
    Citation,
    RunTrace,
    TokenUsage,
)


class FakeAskService:
    """Return deterministic answers for evaluation tests."""

    def __init__(self, answers: list[AnswerResult]) -> None:
        """Store the answer list."""

        self._answers = list(answers)
        self._index = 0

    def ask(self, question: str, *, kb_id: str | None = None) -> AnswerResult:
        """Return the next answer from the list."""

        del question, kb_id
        if self._index < len(self._answers):
            result = self._answers[self._index]
            self._index += 1
            return result
        return AnswerResult(answer="", answered=False, reason="no_more_answers")


def test_load_dataset_json(tmp_path: Path) -> None:
    """Verify loading a JSON evaluation dataset."""

    dataset = tmp_path / "eval.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question": "What is Atlas?",
                    "expected_answer": "Atlas is a QA product.",
                    "expected_sources": ["01_overview.md"],
                    "tags": ["overview"],
                },
                {
                    "question": "Which vector store does Atlas use?",
                    "expected_answer": "Qdrant.",
                    "expected_sources": ["02_architecture.md"],
                },
            ]
        ),
        encoding="utf-8",
    )

    cases = load_dataset(dataset)
    assert len(cases) == 2
    assert cases[0].question == "What is Atlas?"
    assert cases[0].expected_sources == ("01_overview.md",)
    assert cases[0].tags == ("overview",)
    assert cases[1].expected_answer == "Qdrant."


def test_load_dataset_jsonl(tmp_path: Path) -> None:
    """Verify loading a JSONL evaluation dataset."""

    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        '{"question": "Q1", "expected_answer": "A1"}\n'
        '{"question": "Q2", "expected_answer": "A2"}\n',
        encoding="utf-8",
    )

    cases = load_dataset(dataset)
    assert len(cases) == 2
    assert cases[0].question == "Q1"
    assert cases[1].question == "Q2"


def test_load_dataset_skips_invalid_entries(tmp_path: Path) -> None:
    """Verify loading skips entries without a question."""

    dataset = tmp_path / "eval.json"
    dataset.write_text(
        json.dumps([{"expected_answer": "no question"}, {"question": "Valid"}]),
        encoding="utf-8",
    )

    cases = load_dataset(dataset)
    assert len(cases) == 1
    assert cases[0].question == "Valid"


def test_compute_metrics_empty() -> None:
    """Verify metrics with zero results."""

    metrics = compute_metrics(())
    assert metrics.total == 0
    assert metrics.answer_rate == 0.0


def test_compute_metrics_aggregation() -> None:
    """Verify metrics are correctly aggregated."""

    results = (
        EvalCaseResult(
            question="Q1",
            answer="A1",
            answered=True,
            reason=None,
            expected_answer="A1",
            expected_sources=("doc1.md",),
            citation_sources=("doc1.md",),
            source_hit=True,
            latency_ms=100,
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
            trace_id="t1",
        ),
        EvalCaseResult(
            question="Q2",
            answer="",
            answered=False,
            reason="insufficient_evidence",
            expected_answer="A2",
            expected_sources=("doc2.md",),
            citation_sources=(),
            source_hit=False,
            latency_ms=200,
            token_usage=TokenUsage(prompt_tokens=30, completion_tokens=0, total_tokens=30),
            trace_id="t2",
        ),
    )

    metrics = compute_metrics(results)
    assert metrics.total == 2
    assert metrics.answered == 1
    assert metrics.unanswered == 1
    assert metrics.answer_rate == 0.5
    assert metrics.source_hit_count == 1
    assert metrics.source_hit_rate == 0.5
    assert metrics.avg_latency_ms == 150.0
    assert metrics.total_tokens == 100
    assert metrics.avg_tokens_per_case == 50.0


def test_check_source_hit() -> None:
    """Verify source hit detection logic."""

    assert _check_source_hit(("doc1.md", "doc2.md"), ("doc1.md",))
    assert _check_source_hit(("path/to/doc1.md",), ("doc1.md",))
    assert not _check_source_hit(("doc1.md",), ("doc3.md",))
    assert _check_source_hit(("doc1.md",), ())
    assert not _check_source_hit((), ("doc1.md",))


def test_eval_runner_executes_cases() -> None:
    """Verify EvalRunner runs cases against the service."""

    service = FakeAskService(
        [
            AnswerResult(
                answer="Atlas uses Qdrant.",
                answered=True,
                citations=(
                    Citation(index=1, source="02_architecture.md", snippet="...", chunk_id="c1"),
                ),
                trace_id="trace-1",
            ),
            AnswerResult(
                answer="I do not know.",
                answered=False,
                reason="insufficient_evidence",
            ),
        ]
    )

    cases = (
        EvalCase(
            question="Which vector store?",
            expected_sources=("02_architecture.md",),
        ),
        EvalCase(
            question="Unknown topic",
            expected_sources=("nonexistent.md",),
        ),
    )

    runner = EvalRunner(service)
    report = runner.run(cases, dataset_path="test-dataset")

    assert report.metrics.total == 2
    assert report.metrics.answered == 1
    assert report.metrics.source_hit_count == 1
    assert report.results[0].source_hit is True
    assert report.results[1].source_hit is False
    assert report.results[0].trace_id == "trace-1"
    assert report.dataset_path == "test-dataset"


def test_eval_runner_handles_service_errors() -> None:
    """Verify EvalRunner captures exceptions as eval errors."""

    class BrokenService:
        def ask(self, question: str, *, kb_id: str | None = None) -> AnswerResult:
            raise RuntimeError("Service unavailable")

    runner = EvalRunner(BrokenService())
    cases = (EvalCase(question="Will this fail?"),)
    report = runner.run(cases)

    assert report.metrics.total == 1
    assert report.metrics.answered == 0
    assert report.results[0].reason and "eval_error" in report.results[0].reason


def test_trace_eval_runner_matches_traces() -> None:
    """Verify TraceEvalRunner matches cases to existing traces."""

    traces = (
        RunTrace(
            trace_id="t1",
            question="What is Atlas?",
            rewritten_query="What is Atlas?",
            answer="Atlas is a QA product.",
            answered=True,
            citations=(
                Citation(index=1, source="01_overview.md", snippet="...", chunk_id="c1"),
            ),
            latency_ms=150,
            token_usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
        ),
    )

    cases = (
        EvalCase(
            question="What is Atlas?",
            expected_sources=("01_overview.md",),
        ),
        EvalCase(
            question="Unmatched question",
            expected_sources=("02_architecture.md",),
        ),
    )

    runner = TraceEvalRunner()
    report = runner.evaluate_traces(cases, traces)

    assert report.metrics.total == 2
    assert report.metrics.answered == 1
    assert report.results[0].source_hit is True
    assert report.results[0].latency_ms == 150
    assert report.results[1].reason == "no_matching_trace"


def test_format_report() -> None:
    """Verify the report formatter produces readable output."""

    report = EvalReport(
        metrics=EvalMetrics(
            total=3,
            answered=2,
            unanswered=1,
            answer_rate=0.6667,
            source_hit_count=2,
            source_hit_rate=0.6667,
            avg_latency_ms=120.0,
            total_tokens=300,
            avg_tokens_per_case=100.0,
        ),
        results=(
            EvalCaseResult(
                question="Q1",
                answer="A1",
                answered=True,
                reason=None,
                expected_answer="A1",
                expected_sources=(),
                citation_sources=("doc1.md",),
                source_hit=True,
                latency_ms=100,
                token_usage=TokenUsage(),
                trace_id="t1",
            ),
            EvalCaseResult(
                question="Q2",
                answer="",
                answered=False,
                reason="insufficient_evidence",
                expected_answer="A2",
                expected_sources=(),
                citation_sources=(),
                source_hit=False,
                latency_ms=200,
                token_usage=TokenUsage(),
                trace_id="t2",
            ),
        ),
        dataset_path="test.json",
    )

    text = format_report(report)
    assert "Evaluation Report" in text
    assert "test.json" in text
    assert "Total cases:        3" in text
    assert "Answer rate:" in text
    assert "ANSWERED" in text
    assert "REFUSED" in text


def test_save_report_json(tmp_path: Path) -> None:
    """Verify saving a report as JSON."""

    report = EvalReport(
        metrics=EvalMetrics(total=1, answered=1, answer_rate=1.0),
        results=(
            EvalCaseResult(
                question="Q1",
                answer="A1",
                answered=True,
                reason=None,
                expected_answer="A1",
                expected_sources=("doc1.md",),
                citation_sources=("doc1.md",),
                source_hit=True,
                latency_ms=50,
                token_usage=TokenUsage(total_tokens=30),
                trace_id="t1",
            ),
        ),
        dataset_path="test.json",
    )

    output = tmp_path / "report.json"
    save_report_json(report, output)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["metrics"]["total"] == 1
    assert data["metrics"]["answer_rate"] == 1.0
    assert len(data["results"]) == 1
    assert data["results"][0]["question"] == "Q1"
    assert data["results"][0]["source_hit"] is True
