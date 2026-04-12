"""Minimal Qdrant adapter for RAG retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models  # type: ignore[import-not-found]

from hello_agents.rag.config import RagConfig
from hello_agents.rag.models import RagChunk


class RagQdrantStore:
    """Persist and search RAG chunks using the official Qdrant client."""

    def __init__(self, config: RagConfig) -> None:
        """Store Qdrant connection settings."""

        if not config.qdrant_url:
            raise ValueError("Qdrant requires QDRANT_URL for RAG.")
        self._config = config
        self._client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            timeout=config.qdrant_timeout,
        )
        self._vector_size: int | None = None

    def upsert(
        self,
        chunks: Sequence[RagChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Upsert chunk embeddings into the Qdrant collection."""

        if not chunks:
            return
        if self._vector_size is None and embeddings:
            self._ensure_collection(vector_size=len(embeddings[0]))

        points = [
            models.PointStruct(
                id=chunk.id,
                vector=list(embedding),
                payload={
                    "source": chunk.source,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._client.upsert(
            collection_name=self._config.collection,
            points=points,
            wait=True,
        )

    def search(self, embedding: Sequence[float], *, top_k: int) -> list[RagChunk]:
        """Search for the top-k closest chunks."""

        response = self._client.query_points(
            collection_name=self._config.collection,
            query=list(embedding),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            _scored_point_to_chunk(point)
            for point in response.points
            if isinstance(point.payload, dict)
        ]

    def _ensure_collection(self, *, vector_size: int) -> None:
        """Create the collection if needed."""

        self._vector_size = vector_size
        if self._client.collection_exists(self._config.collection):
            return
        self._client.create_collection(
            collection_name=self._config.collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
            timeout=int(self._config.qdrant_timeout),
        )


def _scored_point_to_chunk(point: Any) -> RagChunk:
    """Convert a Qdrant scored point into a RAG chunk."""

    payload = point.payload if isinstance(point.payload, dict) else {}
    metadata = payload.get("metadata", {})
    return RagChunk(
        id=str(point.id),
        source=str(payload.get("source", "")),
        content=str(payload.get("content", "")),
        score=float(point.score or 0.0),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )
