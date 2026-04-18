"""Minimal Qdrant adapter for RAG retrieval."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hello_agents.rag.config import RagConfig
from hello_agents.rag.models import RagChunk

try:
    from qdrant_client import (  # type: ignore[import-not-found]
        QdrantClient as _qdrant_client_cls,
    )
    from qdrant_client import models as _qdrant_models  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised in import-only paths.
    _qdrant_client_cls = None  # type: ignore[assignment,misc]

    @dataclass(slots=True, frozen=True)
    class _SparseVector:
        """Provide the sparse-vector shape used by tests without Qdrant."""

        indices: list[int]
        values: list[float]

    class _MissingQdrantModels:
        """Raise a clear error for Qdrant-only symbols when deps are absent."""

        SparseVector = _SparseVector

        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError(
                "qdrant_client is required for Qdrant-backed RAG storage."
            )

    _qdrant_models = _MissingQdrantModels()  # type: ignore[assignment]

QdrantClient: Any = _qdrant_client_cls
models: Any = _qdrant_models


class RagQdrantStore:
    """Persist and search RAG chunks using Qdrant hybrid retrieval."""

    _DENSE_VECTOR_NAME = "dense"
    _SPARSE_VECTOR_NAME = "sparse"

    def __init__(self, config: RagConfig) -> None:
        """Store Qdrant connection settings."""

        if not config.qdrant_url:
            raise ValueError("Qdrant requires QDRANT_URL for RAG.")
        if QdrantClient is None:
            raise ModuleNotFoundError(
                "qdrant_client is required for Qdrant-backed RAG storage."
            )
        self._config = config
        self._client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            timeout=int(config.qdrant_timeout),
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

        for chunk_batch, embedding_batch in _iter_batches(
            chunks,
            embeddings,
            batch_size=self._config.qdrant_upsert_batch_size,
        ):
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
                for chunk, embedding in zip(
                    chunk_batch,
                    embedding_batch,
                    strict=True,
                )
            ]
            try:
                self._client.upsert(
                    collection_name=self._config.collection,
                    points=points,
                    wait=self._config.qdrant_wait_for_upsert,
                    timeout=int(self._config.qdrant_timeout),
                )
            except Exception as exc:
                raise RuntimeError(
                    "Qdrant upsert failed. Consider lowering "
                    "QDRANT_UPSERT_BATCH_SIZE or increasing QDRANT_TIMEOUT. "
                    f"collection='{self._config.collection}', "
                    f"batch_size={len(points)}, "
                    f"wait={self._config.qdrant_wait_for_upsert}, "
                    f"timeout={self._config.qdrant_timeout}"
                ) from exc

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
            collection_info = self._client.get_collection(self._config.collection)
            mismatch_reason = _collection_schema_mismatch_reason(
                collection_info,
                dense_vector_name=self._DENSE_VECTOR_NAME,
                sparse_vector_name=self._SPARSE_VECTOR_NAME,
                vector_size=vector_size,
            )
            if mismatch_reason is None:
                return
            points_count = _collection_points_count(collection_info)
            if points_count == 0 or self._config.recreate_collection_on_schema_mismatch:
                self._client.delete_collection(self._config.collection)
                self._create_collection(vector_size=vector_size)
                return
            raise RuntimeError(
                "Existing Qdrant collection schema is incompatible with the "
                "current RAG setup. "
                f"Collection='{self._config.collection}', "
                f"reason='{mismatch_reason}', "
                f"points_count={points_count}. "
                "Delete the collection manually or set "
                "RAG_RECREATE_COLLECTION_ON_SCHEMA_MISMATCH=true to recreate it."
            )
        self._create_collection(vector_size=vector_size)

    def _create_collection(self, *, vector_size: int) -> None:
        """Create the Qdrant collection with the expected hybrid schema."""

        self._vector_size = vector_size
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


def _collection_schema_mismatch_reason(
    collection_info: Any,
    *,
    dense_vector_name: str,
    sparse_vector_name: str,
    vector_size: int,
) -> str | None:
    """Return the schema mismatch reason for an existing collection."""

    params = _collection_params(collection_info)
    vectors = _get_config_value(params, "vectors")
    sparse_vectors = _get_config_value(params, "sparse_vectors")

    if not isinstance(vectors, dict):
        return "collection uses unnamed dense vectors"
    dense_config = vectors.get(dense_vector_name)
    if dense_config is None:
        return f"missing dense vector '{dense_vector_name}'"

    existing_vector_size = _as_int(_get_config_value(dense_config, "size"))
    if existing_vector_size != vector_size:
        return (
            f"dense vector '{dense_vector_name}' has size {existing_vector_size}, "
            f"expected {vector_size}"
        )

    if not isinstance(sparse_vectors, dict):
        return "collection is missing sparse vector configuration"
    if sparse_vector_name not in sparse_vectors:
        return f"missing sparse vector '{sparse_vector_name}'"
    return None


def _collection_points_count(collection_info: Any) -> int:
    """Return the current point count for an existing collection."""

    return _as_int(_get_config_value(collection_info, "points_count"))


def _collection_params(collection_info: Any) -> Any:
    """Return collection params from a Qdrant collection info payload."""

    result = _get_config_value(collection_info, "result")
    if result is not None:
        collection_info = result
    config = _get_config_value(collection_info, "config")
    params = _get_config_value(config, "params")
    return params


def _get_config_value(payload: Any, key: str) -> Any:
    """Read a key from either a dict-like payload or an attribute-bearing object."""

    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


def _as_int(value: Any) -> int:
    """Convert a collection-info value to an integer when possible."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return 0


def _iter_batches(
    chunks: Sequence[RagChunk],
    embeddings: Sequence[Sequence[float]],
    *,
    batch_size: int,
) -> list[tuple[Sequence[RagChunk], Sequence[Sequence[float]]]]:
    """Split chunks and embeddings into aligned batches."""

    if batch_size <= 0:
        batch_size = len(chunks) if chunks else 1
    return [
        (
            chunks[start : start + batch_size],
            embeddings[start : start + batch_size],
        )
        for start in range(0, len(chunks), batch_size)
    ]


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
