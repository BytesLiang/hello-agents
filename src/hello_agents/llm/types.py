"""Shared message and response models for the LLM client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["system", "user", "assistant"]


@dataclass(slots=True, frozen=True)
class LLMMessage:
    """Represent a chat message sent to or returned from the model."""

    role: MessageRole
    content: str


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """Represent a normalized LLM response."""

    model: str
    content: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
