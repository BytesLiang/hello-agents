"""Optional Qdrant-backed vector store."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast
from urllib import error, request

from hello_agents.memory.base import VectorStore
from hello_agents.memory.config import QdrantStoreConfig
from hello_agents.memory.models import (
    MemoryKind,
    MemoryScope,
    VectorDocument,
    VectorSearchHit,
)


class QdrantVectorStore(VectorStore):
    """Persist long-term memory embeddings inside Qdrant via its HTTP API."""

    def __init__(self, config: QdrantStoreConfig) -> None:
        """Store the Qdrant connection configuration."""

        if not config.url:
            raise ValueError("Qdrant vector store requires QDRANT_URL.")
        self._config = config
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
        self._request(
            "PUT",
            f"/collections/{self._config.collection_name}/points",
            {
                "points": [
                    {
                        "id": document.memory_id,
                        "vector": list(embedding),
                        "payload": payload,
                    }
                ]
            },
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

        must_conditions: list[dict[str, object]] = [
            {"key": "user_id", "match": {"value": context.user_id}},
            {"key": "agent_id", "match": {"value": context.agent_id}},
        ]
        if memory_kinds:
            must_conditions.append(
                {
                    "key": "memory_kind",
                    "match": {
                        "any": [memory_kind.value for memory_kind in memory_kinds],
                    },
                }
            )
        response = self._request(
            "POST",
            f"/collections/{self._config.collection_name}/points/search",
            {
                "vector": list(embedding),
                "limit": limit,
                "filter": {"must": must_conditions},
                "with_payload": True,
            },
        )
        result = response.get("result", [])
        if not isinstance(result, list):
            return []
        hits: list[VectorSearchHit] = []
        for hit in result:
            if not isinstance(hit, dict):
                continue
            payload = hit.get("payload", {})
            if not isinstance(payload, dict):
                continue
            memory_kind = payload.get("memory_kind")
            if not isinstance(memory_kind, str):
                continue
            hits.append(
                VectorSearchHit(
                    memory_id=str(hit.get("id")),
                    memory_kind=MemoryKind(memory_kind),
                    score=float(hit.get("score", 0.0)),
                    payload=payload,
                )
            )
        return hits

    def _ensure_collection(self, *, vector_size: int) -> None:
        """Create the target collection if needed."""

        self._vector_size = vector_size
        try:
            self._request(
                "PUT",
                f"/collections/{self._config.collection_name}",
                {
                    "vectors": {
                        "size": vector_size,
                        "distance": "Cosine",
                    }
                },
            )
        except RuntimeError:
            self._vector_size = vector_size

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        """Issue a JSON request to the Qdrant HTTP API."""

        data = json.dumps(payload).encode("utf-8")
        base_url = self._config.url
        if base_url is None:
            raise RuntimeError("Qdrant URL is not configured.")
        url = f"{base_url.rstrip('/')}{path}"
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["api-key"] = self._config.api_key
        request_obj = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(
                request_obj,
                timeout=self._config.timeout,
            ) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"Qdrant request failed: {exc}") from exc
        if not raw:
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("Qdrant returned a non-object JSON payload.")
        return cast(dict[str, object], parsed)
