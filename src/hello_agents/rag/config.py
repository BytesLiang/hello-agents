"""Configuration for the lightweight RAG subsystem."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from hello_agents.memory.config import EmbedConfig


@dataclass(slots=True, frozen=True)
class RagConfig:
    """Configure the RAG indexer and retriever."""

    enabled: bool = False
    paths: tuple[Path, ...] = ()
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120
    collection: str = "hello_agents_rag"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_timeout: float = 10.0
    qdrant_upsert_batch_size: int = 64
    qdrant_wait_for_upsert: bool = True
    recreate_collection_on_schema_mismatch: bool = False
    embed: EmbedConfig | None = field(default_factory=EmbedConfig.from_env)

    @classmethod
    def from_env(cls) -> RagConfig:
        """Build a config from environment variables."""

        enabled = os.getenv("RAG_ENABLED", "").lower() in {"1", "true", "yes"}
        paths = _parse_paths(os.getenv("RAG_PATHS", ""))
        top_k = int(os.getenv("RAG_TOP_K", "5"))
        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
        chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
        collection = os.getenv("RAG_COLLECTION", "hello_agents_rag")
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        qdrant_timeout = float(os.getenv("QDRANT_TIMEOUT", "10"))
        qdrant_upsert_batch_size = int(
            os.getenv("QDRANT_UPSERT_BATCH_SIZE", "64")
        )
        qdrant_wait_for_upsert = (
            os.getenv("QDRANT_WAIT_FOR_UPSERT", "true").lower()
            in {"1", "true", "yes"}
        )
        recreate_collection_on_schema_mismatch = os.getenv(
            "RAG_RECREATE_COLLECTION_ON_SCHEMA_MISMATCH", ""
        ).lower() in {"1", "true", "yes"}
        embed = EmbedConfig.from_env()

        return cls(
            enabled=enabled,
            paths=paths,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection=collection,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_timeout=qdrant_timeout,
            qdrant_upsert_batch_size=qdrant_upsert_batch_size,
            qdrant_wait_for_upsert=qdrant_wait_for_upsert,
            recreate_collection_on_schema_mismatch=(
                recreate_collection_on_schema_mismatch
            ),
            embed=embed,
        )


def _parse_paths(raw: str) -> tuple[Path, ...]:
    """Parse a path list from comma/OS-path separators."""

    if not raw.strip():
        return ()
    parts: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts.extend(token.split(os.pathsep))
    return tuple(Path(part.strip()) for part in parts if part.strip())
