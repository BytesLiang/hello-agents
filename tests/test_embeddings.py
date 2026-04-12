"""Tests for embedding adapters."""

from __future__ import annotations

from hello_agents.memory.config import EmbedConfig
from hello_agents.memory.embeddings.dashscope import DashScopeEmbedder


class _FakeEmbeddingItem:
    """Represent one fake embedding response item."""

    def __init__(self, value: float) -> None:
        """Store the deterministic embedding value."""

        self.embedding = [value]


class _FakeEmbeddingResponse:
    """Represent one fake embeddings API response."""

    def __init__(self, batch_size: int) -> None:
        """Return one embedding vector per input item."""

        self.data = [_FakeEmbeddingItem(float(index)) for index in range(batch_size)]


class _FakeEmbeddingsAPI:
    """Capture embedding requests for batching assertions."""

    def __init__(self) -> None:
        """Initialize captured requests."""

        self.calls: list[list[str]] = []

    def create(self, *, model: str, input: list[str]) -> _FakeEmbeddingResponse:
        """Record a request and return deterministic fake embeddings."""

        del model
        self.calls.append(input)
        return _FakeEmbeddingResponse(len(input))


class _FakeClient:
    """Expose the embeddings API expected by DashScopeEmbedder."""

    def __init__(self) -> None:
        """Initialize fake embeddings API."""

        self.embeddings = _FakeEmbeddingsAPI()


def test_dashscope_embedder_batches_requests() -> None:
    """Verify embedding requests are split into batches of at most 10."""

    embedder = DashScopeEmbedder.__new__(DashScopeEmbedder)
    embedder._config = EmbedConfig(  # type: ignore[attr-defined]
        model_type="dashscope",
        model_name="text-embedding-v3",
        api_key="test-key",
        base_url="https://example.com/v1",
    )
    embedder._client = _FakeClient()  # type: ignore[attr-defined]

    texts = [f"text-{index}" for index in range(23)]
    vectors = embedder.embed_texts(texts)

    assert len(vectors) == 23
    assert (
        embedder._client.embeddings.calls
        == [  # type: ignore[attr-defined]
            texts[:10],
            texts[10:20],
            texts[20:23],
        ]
    )
