"""Configuration for the unified LLM client."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class LLMConfig:
    """Store the connection settings for a model endpoint."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0
    provider: str = "openai"

    @classmethod
    def from_env(cls, prefix: str = "LLM") -> LLMConfig:
        """Build config from environment variables."""

        model = os.getenv(f"{prefix}_MODEL", "gpt-4o-mini")
        api_key = os.getenv(f"{prefix}_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv(f"{prefix}_BASE_URL")
        timeout = float(os.getenv(f"{prefix}_TIMEOUT", "30"))
        provider = os.getenv(f"{prefix}_PROVIDER", "openai")
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            provider=provider,
        )

    def resolved_api_key(self) -> str:
        """Return the API key passed into the OpenAI SDK."""

        if self.api_key:
            return self.api_key
        return "EMPTY"
