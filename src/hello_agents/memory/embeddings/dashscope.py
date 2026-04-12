"""DashScope-compatible embedding adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from hello_agents.memory.base import Embedder
from hello_agents.memory.config import EmbedConfig

if TYPE_CHECKING:
    from openai import OpenAI

MAX_BATCH_SIZE = 10


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

        embeddings: list[list[float]] = []
        for batch in _batched(texts, size=MAX_BATCH_SIZE):
            response = self._client.embeddings.create(
                model=self._config.model_name,
                input=batch,
            )
            embeddings.extend(list(item.embedding) for item in response.data)
        return embeddings


def _batched(texts: Sequence[str], *, size: int) -> list[list[str]]:
    """Split texts into fixed-size batches."""

    return [list(texts[index : index + size]) for index in range(0, len(texts), size)]
