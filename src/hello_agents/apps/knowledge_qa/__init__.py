"""Public API for the knowledge QA application layer."""

from __future__ import annotations

from typing import Any

from hello_agents.apps.knowledge_qa.config import KnowledgeQAConfig
from hello_agents.apps.knowledge_qa.models import (
    AnswerResult,
    Citation,
    KnowledgeBase,
    KnowledgeBaseStatus,
    RetrievedChunk,
    RunTrace,
    TokenUsage,
)
from hello_agents.apps.knowledge_qa.service import KnowledgeQAService
from hello_agents.apps.knowledge_qa.store import JsonKnowledgeBaseStore
from hello_agents.apps.knowledge_qa.trace import JsonlRunTraceStore

__all__ = [
    "AnswerResult",
    "Citation",
    "JsonKnowledgeBaseStore",
    "JsonlRunTraceStore",
    "KnowledgeBase",
    "KnowledgeBaseStatus",
    "KnowledgeQAConfig",
    "KnowledgeQARuntime",
    "KnowledgeQAService",
    "RetrievedChunk",
    "RunTrace",
    "TokenUsage",
]


def __getattr__(name: str) -> Any:
    """Load optional runtime dependencies only when requested explicitly."""

    if name == "KnowledgeQARuntime":
        from hello_agents.apps.knowledge_qa.runtime import KnowledgeQARuntime

        return KnowledgeQARuntime
    raise AttributeError(name)
