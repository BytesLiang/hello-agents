"""Minimal Qdrant adapter for RAG retrieval."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast
from urllib import error, request

from hello_agents.rag.config import RagConfig
from hello_agents.rag.models import RagChunk


class RagQdrantStore:
    """Persist and search RAG chunks via Qdrant HTTP API."""

    def __init__(self, config: RagConfig) -> None:
        """Store Qdrant connection settings."""

        if not config.qdrant_url:
            raise ValueError("Qdrant requires QDRANT_URL for RAG.")
        self._url = config.qdrant_url
        self._api_key = config.qdrant_api_key
        self._collection = config.collection
        self._timeout = config.qdrant_timeout
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

        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            points.append(
                {
                    "id": chunk.id,
                    "vector": list(embedding),
                    "payload": {
                        "source": chunk.source,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                    },
                }
            )
        self._request(
            "PUT",
            f"/collections/{self._collection}/points",
            {"points": points},
        )

    def search(self, embedding: Sequence[float], *, top_k: int) -> list[RagChunk]:
        """Search for the top-k closest chunks."""

        response = self._request(
            "POST",
            f"/collections/{self._collection}/points/search",
            {
                "vector": list(embedding),
                "limit": top_k,
                "with_payload": True,
            },
        )
        result = response.get("result", [])
        if not isinstance(result, list):
            return []
        chunks: list[RagChunk] = []
        for hit in result:
            if not isinstance(hit, dict):
                continue
            payload = hit.get("payload", {})
            if not isinstance(payload, dict):
                continue
            chunks.append(
                RagChunk(
                    id=str(hit.get("id")),
                    source=str(payload.get("source", "")),
                    content=str(payload.get("content", "")),
                    score=float(hit.get("score", 0.0)),
                    metadata=dict(payload.get("metadata", {}))
                    if isinstance(payload.get("metadata", {}), dict)
                    else {},
                )
            )
        return chunks

    def _ensure_collection(self, *, vector_size: int) -> None:
        """Create the collection if needed."""

        self._vector_size = vector_size
        try:
            self._request(
                "PUT",
                f"/collections/{self._collection}",
                {"vectors": {"size": vector_size, "distance": "Cosine"}},
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
        url = f"{self._url.rstrip('/')}{path}"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["api-key"] = self._api_key
        request_obj = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(request_obj, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"Qdrant request failed: {exc}") from exc
        if not raw:
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("Qdrant returned a non-object JSON payload.")
        return cast(dict[str, object], parsed)
