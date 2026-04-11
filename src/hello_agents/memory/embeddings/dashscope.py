"""DashScope-compatible embedding adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from hello_agents.memory.base import Embedder
from hello_agents.memory.config import EmbedConfig

if TYPE_CHECKING:
    from openai import OpenAI


class DashScopeEmbedder(Embedder):
    """Generate embeddings through an OpenAI-compatible DashScope endpoint."""

    def __init__(self, config: EmbedConfig) -> None:
        """Create an OpenAI SDK client for embedding requests."""

        from openai import OpenAI

        self._config = config
        self._client: OpenAI = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for the provided texts."""

        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._config.model_name,
            input=list(texts),
        )
        return [list(item.embedding) for item in response.data]
