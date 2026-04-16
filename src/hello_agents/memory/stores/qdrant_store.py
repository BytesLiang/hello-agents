"""Optional Qdrant-backed vector store."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from qdrant_client import QdrantClient, models  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised in import-only paths.
    QdrantClient = None  # type: ignore[assignment]

    class _MissingQdrantModels:
        """Raise a clear error when Qdrant-only symbols are used without deps."""

        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError(
                "qdrant_client is required for Qdrant-backed memory storage."
            )

    models = _MissingQdrantModels()

from hello_agents.memory.base import VectorStore
from hello_agents.memory.config import QdrantStoreConfig
from hello_agents.memory.models import (
    MemoryKind,
    MemoryScope,
    VectorDocument,
    VectorSearchHit,
)


class QdrantVectorStore(VectorStore):
    """Persist long-term memory embeddings using the official Qdrant client."""

    def __init__(self, config: QdrantStoreConfig) -> None:
        """Store the Qdrant connection configuration."""

        if not config.url:
            raise ValueError("Qdrant vector store requires QDRANT_URL.")
        if QdrantClient is None:
            raise ModuleNotFoundError(
                "qdrant_client is required for Qdrant-backed memory storage."
            )
        self._config = config
        self._client = QdrantClient(
            url=config.url,
            api_key=config.api_key,
            timeout=config.timeout,
        )
        self._vector_size: int | None = None

    def upsert(self, document: VectorDocument, embedding: Sequence[float]) -> None:
        """Upsert a vector point into Qdrant."""

        if self._vector_size is None:
            self._ensure_collection(vector_size=len(embedding))
        payload = {
            "memory_id": document.memory_id,
            "memory_kind": document.memory_kind.value,
            "user_id": document.user_id,
            "agent_id": document.agent_id,
            "summary": document.summary,
            "confidence": document.confidence,
            "created_at": document.created_at.isoformat(),
        }
        self._client.upsert(
            collection_name=self._config.collection_name,
            points=[
                models.PointStruct(
                    id=document.memory_id,
                    vector=list(embedding),
                    payload=payload,
                )
            ],
            wait=True,
        )

    def search(
        self,
        context: MemoryScope,
        *,
        embedding: Sequence[float],
        memory_kinds: Sequence[MemoryKind],
        limit: int,
    ) -> list[VectorSearchHit]:
        """Search similar memory points scoped to a user and agent namespace."""

        must_conditions: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=context.user_id),
            ),
            models.FieldCondition(
                key="agent_id",
                match=models.MatchValue(value=context.agent_id),
            ),
        ]
        if memory_kinds:
            must_conditions.append(
                models.FieldCondition(
                    key="memory_kind",
                    match=models.MatchAny(
                        any=[memory_kind.value for memory_kind in memory_kinds]
                    ),
                )
            )

        response = self._client.query_points(
            collection_name=self._config.collection_name,
            query=list(embedding),
            query_filter=models.Filter(must=must_conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            _scored_point_to_hit(point)
            for point in response.points
            if isinstance(point.payload, dict)
            and isinstance(point.payload.get("memory_kind"), str)
        ]

    def _ensure_collection(self, *, vector_size: int) -> None:
        """Create the target collection if needed."""

        self._vector_size = vector_size
        if self._client.collection_exists(self._config.collection_name):
            return
        self._client.create_collection(
            collection_name=self._config.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
            timeout=int(self._config.timeout),
        )


def _scored_point_to_hit(point: Any) -> VectorSearchHit:
    """Convert a Qdrant scored point into the framework search hit."""

    payload = point.payload if isinstance(point.payload, dict) else {}
    memory_kind = payload.get("memory_kind", "")
    return VectorSearchHit(
        memory_id=str(point.id),
        memory_kind=MemoryKind(str(memory_kind)),
        score=float(point.score or 0.0),
        payload=payload,
    )
