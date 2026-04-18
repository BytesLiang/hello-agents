"""Configuration models for the knowledge QA application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class KnowledgeQAConfig:
    """Configure retrieval, answer generation, and local storage paths."""

    retrieval_top_k: int = 5
    answer_context_top_k: int = 4
    max_citations: int = 4
    min_retrieved_chunks: int = 1
    answer_max_tokens: int | None = 512
    trace_prompt_chars: int = 4_000
    knowledge_base_store_path: Path = Path(".hello_agents/knowledge_bases.json")
    trace_store_path: Path = Path(".hello_agents/knowledge_qa_traces.jsonl")

    @classmethod
    def from_env(cls) -> KnowledgeQAConfig:
        """Build configuration from environment variables."""

        return cls(
            retrieval_top_k=int(os.getenv("KNOWLEDGE_QA_RETRIEVAL_TOP_K", "5")),
            answer_context_top_k=int(
                os.getenv("KNOWLEDGE_QA_ANSWER_CONTEXT_TOP_K", "4")
            ),
            max_citations=int(os.getenv("KNOWLEDGE_QA_MAX_CITATIONS", "4")),
            min_retrieved_chunks=int(
                os.getenv("KNOWLEDGE_QA_MIN_RETRIEVED_CHUNKS", "1")
            ),
            answer_max_tokens=_parse_optional_int(
                os.getenv("KNOWLEDGE_QA_ANSWER_MAX_TOKENS", "512")
            ),
            trace_prompt_chars=int(
                os.getenv("KNOWLEDGE_QA_TRACE_PROMPT_CHARS", "4000")
            ),
            knowledge_base_store_path=Path(
                os.getenv(
                    "KNOWLEDGE_QA_STORE_PATH",
                    ".hello_agents/knowledge_bases.json",
                )
            ),
            trace_store_path=Path(
                os.getenv(
                    "KNOWLEDGE_QA_TRACE_PATH",
                    ".hello_agents/knowledge_qa_traces.jsonl",
                )
            ),
        )


def _parse_optional_int(raw: str | None) -> int | None:
    """Parse an optional integer from environment configuration."""

    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(normalized)
