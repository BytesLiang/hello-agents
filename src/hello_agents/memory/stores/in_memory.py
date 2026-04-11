"""In-memory store implementations for working memory."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from hello_agents.memory.config import WorkingMemoryConfig
from hello_agents.memory.models import MemoryScope, WorkingMemoryRecord


class InMemoryWorkingMemoryStore:
    """Store working memory entries in the current process."""

    def __init__(self, config: WorkingMemoryConfig) -> None:
        """Store the configuration and initialize namespace buckets."""

        self._config = config
        self._entries: dict[str, list[WorkingMemoryRecord]] = defaultdict(list)

    def list_entries(
        self,
        context: MemoryScope,
        *,
        now: datetime | None = None,
    ) -> list[WorkingMemoryRecord]:
        """Return active entries for the given memory namespace."""

        effective_now = now or datetime.now(tz=UTC)
        namespace = _namespace(context)
        active = [
            entry
            for entry in self._entries[namespace]
            if entry.expires_at is None or entry.expires_at > effective_now
        ]
        self._entries[namespace] = active
        return list(active)

    def append_entries(
        self,
        context: MemoryScope,
        records: Sequence[WorkingMemoryRecord],
        *,
        now: datetime | None = None,
    ) -> None:
        """Append working-memory records and enforce size and TTL limits."""

        effective_now = now or datetime.now(tz=UTC)
        namespace = _namespace(context)
        active = self.list_entries(context, now=effective_now)
        ttl = timedelta(seconds=self._config.ttl_seconds)
        normalized: list[WorkingMemoryRecord] = []
        for record in records:
            expires_at = record.expires_at or (effective_now + ttl)
            normalized.append(
                WorkingMemoryRecord(
                    id=record.id,
                    user_id=record.user_id,
                    session_id=record.session_id,
                    agent_id=record.agent_id,
                    kind=record.kind,
                    content=record.content,
                    pinned=record.pinned,
                    created_at=record.created_at,
                    updated_at=effective_now,
                    expires_at=expires_at,
                )
            )

        merged = active + normalized
        merged.sort(key=lambda item: item.updated_at)
        if len(merged) > self._config.max_entries:
            pinned = [item for item in merged if item.pinned]
            unpinned = [item for item in merged if not item.pinned]
            retained = unpinned[-self._config.max_entries :]
            merged = (pinned + retained)[-self._config.max_entries :]
        self._entries[namespace] = merged


def _namespace(context: MemoryScope) -> str:
    """Return the namespace key for a working-memory session."""

    return f"{context.user_id}:{context.session_id}:{context.agent_id}"
