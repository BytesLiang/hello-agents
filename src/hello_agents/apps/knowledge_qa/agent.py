"""Agent-style orchestration for the knowledge QA workflow."""

from __future__ import annotations

from dataclasses import dataclass

from hello_agents.apps.knowledge_qa.classifier import (
    QuestionClassification,
    QuestionClassifier,
    QuestionType,
)
from hello_agents.apps.knowledge_qa.evidence import (
    EvidenceAssessment,
    EvidenceScorer,
    EvidenceStatus,
)
from hello_agents.apps.knowledge_qa.input_check import (
    InputCheckResult,
    QuestionInputChecker,
)
from hello_agents.apps.knowledge_qa.inspector import (
    DocumentInspectionResult,
    DocumentInspector,
)
from hello_agents.apps.knowledge_qa.llm_utils import SupportsChat, merge_token_usage
from hello_agents.apps.knowledge_qa.models import RetrievedChunk, TokenUsage
from hello_agents.apps.knowledge_qa.planner import QueryPlanner, RetrievalPlan
from hello_agents.apps.knowledge_qa.retrieve import KnowledgeRetriever, RetrievalResult


@dataclass(slots=True, frozen=True)
class RetrievalRound:
    """Capture one retrieval attempt within the agent workflow."""

    round_index: int
    query: str
    rewritten_query: str
    retrieved_chunks: tuple[RetrievedChunk, ...]
    assessment: EvidenceAssessment
    suggested_query: str = ""


@dataclass(slots=True, frozen=True)
class AgentRunResult:
    """Bundle the outputs produced by the question agent."""

    input_check: InputCheckResult
    classification: QuestionClassification
    plan: RetrievalPlan
    rounds: tuple[RetrievalRound, ...]
    rewritten_query: str
    retrieved_chunks: tuple[RetrievedChunk, ...]
    selected_chunks: tuple[RetrievedChunk, ...]
    inspection_result: DocumentInspectionResult | None
    final_assessment: EvidenceAssessment
    token_usage: TokenUsage = TokenUsage()


class QuestionAgent:
    """Coordinate classification, retrieval planning, and document inspection."""

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        *,
        llm: SupportsChat | None = None,
        classifier: QuestionClassifier | None = None,
        planner: QueryPlanner | None = None,
        scorer: EvidenceScorer | None = None,
        inspector: DocumentInspector | None = None,
    ) -> None:
        """Store workflow collaborators used for one agent run."""

        self._retriever = retriever
        self._input_checker = QuestionInputChecker(llm)
        self._classifier = classifier or QuestionClassifier(llm)
        self._planner = planner or QueryPlanner(llm)
        self._scorer = scorer or EvidenceScorer(llm)
        self._inspector = inspector or DocumentInspector()

    def run(
        self,
        *,
        question: str,
        source_paths: tuple[str, ...] = (),
        kb_id: str | None = None,
        answer_context_top_k: int,
    ) -> AgentRunResult:
        """Run the agent workflow and return its final evidence bundle."""

        input_check = self._input_checker.check(question)
        if not input_check.is_valid:
            failure = EvidenceAssessment(
                status=EvidenceStatus.INSUFFICIENT,
                score=0.0,
                reason=input_check.reason or "Question failed input validation.",
                failure_mode=input_check.reason or "input_invalid",
            )
            return AgentRunResult(
                input_check=input_check,
                classification=QuestionClassification(
                    question_type=QuestionType.AMBIGUOUS,
                    reason="Input validation failed.",
                ),
                plan=RetrievalPlan(
                    primary_queries=(),
                    fallback_queries=(),
                    max_rounds=0,
                    use_document_inspection_on_failure=False,
                    summary="Question failed input validation.",
                ),
                rounds=(),
                rewritten_query="",
                retrieved_chunks=(),
                selected_chunks=(),
                inspection_result=None,
                final_assessment=failure,
                token_usage=input_check.token_usage,
            )

        classification_decision = self._classifier.classify_with_trace(
            input_check.normalized_question
        )
        classification = classification_decision.classification
        plan_decision = self._planner.plan_with_trace(
            input_check.normalized_question,
            classification,
        )
        plan = plan_decision.plan
        query_queue = list(plan.primary_queries + plan.fallback_queries)
        attempted_queries: set[str] = set()
        rounds: list[RetrievalRound] = []
        last_retrieval = RetrievalResult(query="", chunks=())
        total_token_usage = merge_token_usage(
            input_check.token_usage,
            classification_decision.token_usage,
            plan_decision.token_usage,
        )

        round_index = 0
        while query_queue and round_index < plan.max_rounds:
            query = query_queue.pop(0).strip()
            if not query or query in attempted_queries:
                continue
            attempted_queries.add(query)
            round_index += 1
            retrieval = self._retriever.retrieve(
                query,
                source_paths=source_paths,
                kb_id=kb_id,
            )
            last_retrieval = retrieval
            total_token_usage = merge_token_usage(
                total_token_usage,
                retrieval.token_usage,
            )
            evidence_decision = self._scorer.assess_with_trace(
                question=input_check.normalized_question,
                classification=classification,
                chunks=retrieval.chunks,
                round_index=round_index,
                max_rounds=plan.max_rounds,
            )
            total_token_usage = merge_token_usage(
                total_token_usage,
                evidence_decision.token_usage,
            )
            assessment = evidence_decision.assessment
            rounds.append(
                RetrievalRound(
                    round_index=round_index,
                    query=query,
                    rewritten_query=retrieval.query,
                    retrieved_chunks=retrieval.chunks,
                    assessment=assessment,
                    suggested_query=evidence_decision.rewritten_query,
                )
            )
            if assessment.status is EvidenceStatus.SUFFICIENT:
                selected_chunks = retrieval.chunks[:answer_context_top_k]
                return AgentRunResult(
                    input_check=input_check,
                    classification=classification,
                    plan=plan,
                    rounds=tuple(rounds),
                    rewritten_query=retrieval.query,
                    retrieved_chunks=retrieval.chunks,
                    selected_chunks=selected_chunks,
                    inspection_result=None,
                    final_assessment=assessment,
                    token_usage=total_token_usage,
                )
            if assessment.status is EvidenceStatus.NEEDS_DOCUMENT_INSPECTION:
                inspection_result = self._inspector.inspect(
                    classification=classification,
                    source_paths=source_paths,
                    retrieved_chunks=retrieval.chunks,
                )
                if inspection_result is not None:
                    selected_chunks = retrieval.chunks[:answer_context_top_k]
                    return AgentRunResult(
                        input_check=input_check,
                        classification=classification,
                        plan=plan,
                        rounds=tuple(rounds),
                        rewritten_query=retrieval.query,
                        retrieved_chunks=retrieval.chunks,
                        selected_chunks=selected_chunks,
                        inspection_result=inspection_result,
                        final_assessment=EvidenceAssessment(
                            status=EvidenceStatus.SUFFICIENT,
                            score=max(assessment.score, 0.8),
                            reason=(
                                "Direct document inspection produced grounded evidence."
                            ),
                        ),
                        token_usage=total_token_usage,
                    )
            if (
                assessment.status is EvidenceStatus.NEEDS_REWRITE
                and evidence_decision.rewritten_query
                and evidence_decision.rewritten_query not in attempted_queries
            ):
                query_queue.insert(0, evidence_decision.rewritten_query)

        inspection_result = None
        final_assessment = (
            rounds[-1].assessment
            if rounds
            else EvidenceAssessment(
                status=EvidenceStatus.INSUFFICIENT,
                score=0.0,
                reason="No retrieval plan could be executed.",
                failure_mode="no_plan",
            )
        )
        if plan.use_document_inspection_on_failure:
            inspection_result = self._inspector.inspect(
                classification=classification,
                source_paths=source_paths,
                retrieved_chunks=last_retrieval.chunks,
            )
            if inspection_result is not None:
                return AgentRunResult(
                    input_check=input_check,
                    classification=classification,
                    plan=plan,
                    rounds=tuple(rounds),
                    rewritten_query=last_retrieval.query,
                    retrieved_chunks=last_retrieval.chunks,
                    selected_chunks=last_retrieval.chunks[:answer_context_top_k],
                    inspection_result=inspection_result,
                    final_assessment=EvidenceAssessment(
                        status=EvidenceStatus.SUFFICIENT,
                        score=0.8,
                        reason=(
                            "Fallback document inspection produced grounded evidence."
                        ),
                    ),
                    token_usage=total_token_usage,
                )
        return AgentRunResult(
            input_check=input_check,
            classification=classification,
            plan=plan,
            rounds=tuple(rounds),
            rewritten_query=last_retrieval.query,
            retrieved_chunks=last_retrieval.chunks,
            selected_chunks=last_retrieval.chunks[:answer_context_top_k],
            inspection_result=inspection_result,
            final_assessment=final_assessment,
            token_usage=total_token_usage,
        )
