"""Retrieve RAG chunks for query-time augmentation."""

from __future__ import annotations

from hello_agents.memory.embeddings import build_embedder
from hello_agents.rag.config import RagConfig
from hello_agents.rag.models import RagChunk
from hello_agents.rag.qdrant_store import RagQdrantStore


class RagRetriever:
    """Provide query-time retrieval for RAG."""

    def __init__(
        self,
        *,
        config: RagConfig,
        store: RagQdrantStore | None = None,
    ) -> None:
        """Initialize the retriever using embedding and vector storage."""

        if config.embed is None:
            raise ValueError("RAG retrieval requires embedding configuration.")
        self._config = config
        self._embedder = build_embedder(config.embed)
        self._store = store or RagQdrantStore(config)

    @property
    def config(self) -> RagConfig:
        """Return the active retriever config."""

        return self._config

    def query(self, text: str, *, top_k: int | None = None) -> list[RagChunk]:
        """Return the top-k chunks relevant to the query."""

        if not text.strip():
            return []
        top_k = top_k or self._config.top_k
        embedding = self._embedder.embed_texts([text])[0]
        return self._store.search_hybrid(text, embedding, top_k=top_k)
