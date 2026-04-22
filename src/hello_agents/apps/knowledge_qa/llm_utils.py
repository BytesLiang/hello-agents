"""Shared LLM helpers for the knowledge QA workflow."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from hello_agents.apps.knowledge_qa.models import TokenUsage
from hello_agents.llm.types import LLMMessage, LLMResponse


class SupportsChat(Protocol):
    """Describe the chat interface used across the knowledge QA workflow."""

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> LLMResponse:
        """Return one chat completion."""


def load_json_object(content: str) -> dict[str, object] | None:
    """Parse a JSON object from plain or fenced model output."""

    candidate = content.strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def token_usage_from_response(response: LLMResponse) -> TokenUsage:
    """Normalize token usage from an LLM response."""

    return TokenUsage(
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
    )


def merge_token_usage(*items: TokenUsage) -> TokenUsage:
    """Sum multiple token usage values."""

    return TokenUsage(
        prompt_tokens=sum(item.prompt_tokens for item in items),
        completion_tokens=sum(item.completion_tokens for item in items),
        total_tokens=sum(item.total_tokens for item in items),
    )
