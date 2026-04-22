"""Configuration models for the memory subsystem."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True, frozen=True)
class WorkingMemoryConfig:
    """Configure short-lived working memory behavior."""

    ttl_seconds: int = 60 * 60 * 24
    max_entries: int = 24


@dataclass(slots=True, frozen=True)
class SQLiteStoreConfig:
    """Configure the SQLite-backed long-term memory store."""

    path: Path = Path(".hello_agents/memory.sqlite3")
    episodic_retention_days: int = 180


@dataclass(slots=True, frozen=True)
class RedisStoreConfig:
    """Configure the optional Redis working-memory adapter."""

    enabled: bool = False
    url: str | None = None
    prefix: str = "hello_agents:memory"


@dataclass(slots=True, frozen=True)
class QdrantStoreConfig:
    """Configure the optional Qdrant vector index."""

    enabled: bool = False
    url: str | None = None
    api_key: str | None = None
    collection_name: str = "hello_agents_memory"
    timeout: float = 10.0


@dataclass(slots=True, frozen=True)
class Neo4jStoreConfig:
    """Configure the optional Neo4j graph adapter placeholder."""

    enabled: bool = False
    url: str | None = None
    username: str | None = None
    password: str | None = None
    database: str | None = None


@dataclass(slots=True, frozen=True)
class EmbedConfig:
    """Configure the embedding backend used for vector indexing."""

    model_type: str
    model_name: str
    api_key: str
    base_url: str
    timeout: float = 30.0
    max_retries: int = 3

    @classmethod
    def from_env(cls, prefix: str = "EMBED") -> EmbedConfig | None:
        """Build embedding config from environment variables."""

        model_name = os.getenv(f"{prefix}_MODEL_NAME")
        api_key = os.getenv(f"{prefix}_API_KEY")
        base_url = os.getenv(f"{prefix}_BASE_URL")
        if not model_name or not api_key or not base_url:
            return None
        model_type = os.getenv(f"{prefix}_MODEL_TYPE", "dashscope")
        timeout = float(os.getenv(f"{prefix}_TIMEOUT", "30"))
        max_retries = int(os.getenv(f"{prefix}_MAX_RETRIES", "3"))
        return cls(
            model_type=model_type,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )


@dataclass(slots=True, frozen=True)
class MemoryConfig:
    """Bundle the full memory subsystem configuration."""

    working: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    sqlite: SQLiteStoreConfig = field(default_factory=SQLiteStoreConfig)
    redis: RedisStoreConfig = field(default_factory=RedisStoreConfig)
    qdrant: QdrantStoreConfig = field(default_factory=QdrantStoreConfig)
    neo4j: Neo4jStoreConfig = field(default_factory=Neo4jStoreConfig)
    embed: EmbedConfig | None = None

    @classmethod
    def local_default(cls) -> MemoryConfig:
        """Return the default local-first memory profile."""

        return cls(embed=EmbedConfig.from_env())

    @classmethod
    def from_env(cls) -> MemoryConfig:
        """Build memory config from environment variables."""

        sqlite_path = Path(
            os.getenv("MEMORY_SQLITE_PATH", ".hello_agents/memory.sqlite3")
        )
        working = WorkingMemoryConfig(
            ttl_seconds=int(os.getenv("MEMORY_WORKING_TTL_SECONDS", "86400")),
            max_entries=int(os.getenv("MEMORY_WORKING_MAX_ENTRIES", "24")),
        )
        sqlite = SQLiteStoreConfig(
            path=sqlite_path,
            episodic_retention_days=int(
                os.getenv("MEMORY_EPISODIC_RETENTION_DAYS", "180")
            ),
        )
        redis = RedisStoreConfig(
            enabled=os.getenv("MEMORY_REDIS_ENABLED", "").lower() in {"1", "true"},
            url=os.getenv("REDIS_URL"),
            prefix=os.getenv("MEMORY_REDIS_PREFIX", "hello_agents:memory"),
        )
        qdrant = QdrantStoreConfig(
            enabled=os.getenv("QDRANT_URL") is not None,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv(
                "QDRANT_COLLECTION_NAME",
                "hello_agents_memory",
            ),
            timeout=float(os.getenv("QDRANT_TIMEOUT", "10")),
        )
        neo4j = Neo4jStoreConfig(
            enabled=os.getenv("NEO4J_URL") is not None,
            url=os.getenv("NEO4J_URL"),
            username=os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_NAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
        )
        return cls(
            working=working,
            sqlite=sqlite,
            redis=redis,
            qdrant=qdrant,
            neo4j=neo4j,
            embed=EmbedConfig.from_env(),
        )
