"""Minimal Qdrant adapter for RAG retrieval."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models  # type: ignore[import-not-found]

from hello_agents.rag.config import RagConfig
from hello_agents.rag.models import RagChunk


class RagQdrantStore:
    """Persist and search RAG chunks using Qdrant hybrid retrieval."""

    _DENSE_VECTOR_NAME = "dense"
    _SPARSE_VECTOR_NAME = "sparse"

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
                vector={
                    self._DENSE_VECTOR_NAME: list(embedding),
                    self._SPARSE_VECTOR_NAME: _text_to_sparse_vector(chunk.content),
                },
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

        # The retriever does not currently pass the raw query text, so hybrid
        # search is exposed through `search_hybrid()` below and used by the
        # retriever. Keep `search()` as dense-only compatibility fallback.
        response = self._client.query_points(
            collection_name=self._config.collection,
            query=list(embedding),
            using=self._DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            _scored_point_to_chunk(point)
            for point in response.points
            if isinstance(point.payload, dict)
        ]

    def search_hybrid(
        self,
        text: str,
        embedding: Sequence[float],
        *,
        top_k: int,
    ) -> list[RagChunk]:
        """Search with dense+sparse hybrid retrieval and RRF fusion."""

        response = self._client.query_points(
            collection_name=self._config.collection,
            prefetch=[
                models.Prefetch(
                    query=list(embedding),
                    using=self._DENSE_VECTOR_NAME,
                    limit=max(top_k * 2, top_k),
                ),
                models.Prefetch(
                    query=_text_to_sparse_vector(text),
                    using=self._SPARSE_VECTOR_NAME,
                    limit=max(top_k * 2, top_k),
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
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
            vectors_config={
                self._DENSE_VECTOR_NAME: models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self._SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
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


def _text_to_sparse_vector(text: str) -> models.SparseVector:
    """Convert text to a stable sparse vector using hashed token frequencies."""

    counts = Counter(_tokenize(text))
    if not counts:
        return models.SparseVector(indices=[], values=[])
    items = sorted(
        (_token_to_index(token), float(count)) for token, count in counts.items()
    )
    return models.SparseVector(
        indices=[index for index, _ in items],
        values=[value for _, value in items],
    )


def _tokenize(text: str) -> list[str]:
    """Extract lowercase lexical tokens for sparse retrieval."""

    return re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", text.lower())


def _token_to_index(token: str) -> int:
    """Map a token to a stable sparse-dimension index."""

    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 2_147_483_647
