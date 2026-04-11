"""Optional Redis-backed working-memory store."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from importlib import import_module

from hello_agents.memory.config import RedisStoreConfig, WorkingMemoryConfig
from hello_agents.memory.models import (
    MemoryScope,
    WorkingMemoryKind,
    WorkingMemoryRecord,
)


class RedisWorkingMemoryStore:
    """Persist working memory inside Redis when the dependency is available."""

    def __init__(
        self,
        config: WorkingMemoryConfig,
        redis_config: RedisStoreConfig,
    ) -> None:
        """Create a Redis client lazily from the provided configuration."""

        if not redis_config.url:
            raise ValueError("Redis working memory requires REDIS_URL.")
        try:
            redis = import_module("redis")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Redis working memory requires the 'redis' package."
            ) from exc

        self._config = config
        self._redis_config = redis_config
        self._client = redis.from_url(redis_config.url, decode_responses=True)

    def list_entries(
        self,
        context: MemoryScope,
        *,
        now: datetime | None = None,
    ) -> list[WorkingMemoryRecord]:
        """Load and deserialize entries from Redis."""

        del now
        raw_entries = self._client.lrange(
            _namespace(self._redis_config, context),
            0,
            -1,
        )
        return [self._deserialize(entry) for entry in raw_entries]

    def append_entries(
        self,
        context: MemoryScope,
        records: Sequence[WorkingMemoryRecord],
        *,
        now: datetime | None = None,
    ) -> None:
        """Append records and enforce the configured TTL and max length."""

        del now
        key = _namespace(self._redis_config, context)
        payloads = [self._serialize(record) for record in records]
        if payloads:
            self._client.rpush(key, *payloads)
        self._client.ltrim(key, -self._config.max_entries, -1)
        self._client.expire(key, self._config.ttl_seconds)

    @staticmethod
    def _serialize(record: WorkingMemoryRecord) -> str:
        """Serialize a working-memory record for Redis storage."""

        return json.dumps(
            {
                "id": record.id,
                "user_id": record.user_id,
                "session_id": record.session_id,
                "agent_id": record.agent_id,
                "kind": record.kind.value,
                "content": record.content,
                "pinned": record.pinned,
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
                "expires_at": (
                    None if record.expires_at is None else record.expires_at.isoformat()
                ),
            }
        )

    @staticmethod
    def _deserialize(payload: str) -> WorkingMemoryRecord:
        """Deserialize a working-memory record stored in Redis."""

        data = json.loads(payload)
        expires_at = data["expires_at"]
        return WorkingMemoryRecord(
            id=str(data["id"]),
            user_id=str(data["user_id"]),
            session_id=str(data["session_id"]),
            agent_id=str(data["agent_id"]),
            kind=WorkingMemoryKind(str(data["kind"])),
            content=str(data["content"]),
            pinned=bool(data["pinned"]),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            updated_at=datetime.fromisoformat(str(data["updated_at"])),
            expires_at=(
                None if expires_at is None else datetime.fromisoformat(str(expires_at))
            ),
        )


def _namespace(config: RedisStoreConfig, context: MemoryScope) -> str:
    """Return the Redis key used for a working-memory namespace."""

    return f"{config.prefix}:{context.user_id}:{context.session_id}:{context.agent_id}"
