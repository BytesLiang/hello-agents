"""SQLite-backed long-term memory store."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hello_agents.memory.config import SQLiteStoreConfig
from hello_agents.memory.models import (
    EpisodicMemoryRecord,
    MemoryKind,
    MemoryPatch,
    MemoryRecord,
    MemoryScope,
    ProceduralMemoryRecord,
    SemanticMemoryKind,
    SemanticMemoryRecord,
)


class SQLiteMemoryStore:
    """Persist semantic, episodic, and procedural memory in SQLite."""

    def __init__(self, config: SQLiteStoreConfig) -> None:
        """Create the SQLite database and initialize its schema."""

        self._config = config
        self._db_path = Path(config.path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def save_preferences(
        self,
        records: Sequence[SemanticMemoryRecord],
    ) -> list[SemanticMemoryRecord]:
        """Persist semantic preference records with versioning."""

        self._save_semantic(records)
        return list(records)

    def save_facts(
        self,
        records: Sequence[SemanticMemoryRecord],
    ) -> list[SemanticMemoryRecord]:
        """Persist semantic fact records with versioning."""

        self._save_semantic(records)
        return list(records)

    def save_episode(self, record: EpisodicMemoryRecord) -> EpisodicMemoryRecord:
        """Persist an episodic memory record."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO episodic_memories (
                    id,
                    user_id,
                    agent_id,
                    source_session_id,
                    task,
                    summary,
                    content,
                    success,
                    tool_names,
                    confidence,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.user_id,
                    record.agent_id,
                    record.source_session_id,
                    record.task,
                    record.summary,
                    record.content,
                    int(record.success),
                    ",".join(record.tool_names),
                    record.confidence,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
        return record

    def save_procedures(
        self,
        records: Sequence[ProceduralMemoryRecord],
    ) -> list[ProceduralMemoryRecord]:
        """Persist procedural memory records."""

        with self._connect() as connection:
            for record in records:
                connection.execute(
                    """
                    INSERT INTO procedural_memories (
                        id,
                        user_id,
                        agent_id,
                        source_session_id,
                        task_type,
                        summary,
                        content,
                        tool_names,
                        success_count,
                        confidence,
                        last_applied_at,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.user_id,
                        record.agent_id,
                        record.source_session_id,
                        record.task_type,
                        record.summary,
                        record.content,
                        ",".join(record.tool_names),
                        record.success_count,
                        record.confidence,
                        record.last_applied_at.isoformat(),
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    ),
                )
        return list(records)

    def search_semantic(
        self,
        context: MemoryScope,
        *,
        kind: SemanticMemoryKind,
        query: str,
        limit: int,
        vector_scores: dict[str, float] | None = None,
    ) -> list[SemanticMemoryRecord]:
        """Return the highest-scoring semantic records for the query."""

        table = (
            "semantic_preferences"
            if kind == SemanticMemoryKind.PREFERENCE
            else "semantic_facts"
        )
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM {table}
                WHERE user_id = ? AND agent_id = ? AND superseded = 0
                ORDER BY updated_at DESC
                """,
                (context.user_id, context.agent_id),
            ).fetchall()

        ranked = self._rank_rows(
            rows,
            query=query,
            vector_scores=vector_scores,
            success_weight=0.0,
        )
        return [_semantic_from_row(row, kind=kind) for row in ranked[:limit]]

    def search_episodes(
        self,
        context: MemoryScope,
        *,
        query: str,
        limit: int,
        vector_scores: dict[str, float] | None = None,
    ) -> list[EpisodicMemoryRecord]:
        """Return the highest-scoring episodic records for the query."""

        cutoff = datetime.now(tz=UTC) - timedelta(
            days=self._config.episodic_retention_days
        )
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM episodic_memories
                WHERE user_id = ? AND agent_id = ? AND created_at >= ?
                ORDER BY updated_at DESC
                """,
                (context.user_id, context.agent_id, cutoff.isoformat()),
            ).fetchall()

        ranked = self._rank_rows(
            rows,
            query=query,
            vector_scores=vector_scores,
            success_weight=0.25,
        )
        return [_episodic_from_row(row) for row in ranked[:limit]]

    def search_procedures(
        self,
        context: MemoryScope,
        *,
        query: str,
        limit: int,
        vector_scores: dict[str, float] | None = None,
    ) -> list[ProceduralMemoryRecord]:
        """Return the highest-scoring procedural records for the query."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM procedural_memories
                WHERE user_id = ? AND agent_id = ?
                ORDER BY updated_at DESC
                """,
                (context.user_id, context.agent_id),
            ).fetchall()

        ranked = self._rank_rows(
            rows,
            query=query,
            vector_scores=vector_scores,
            success_weight=0.15,
        )
        return [_procedural_from_row(row) for row in ranked[:limit]]

    def _save_semantic(self, records: Sequence[SemanticMemoryRecord]) -> None:
        """Persist semantic memory records and supersede prior versions."""

        if not records:
            return
        with self._connect() as connection:
            for record in records:
                table = (
                    "semantic_preferences"
                    if record.kind == SemanticMemoryKind.PREFERENCE
                    else "semantic_facts"
                )
                connection.execute(
                    f"""
                    UPDATE {table}
                    SET superseded = 1, updated_at = ?
                    WHERE user_id = ? AND agent_id = ? AND key = ? AND superseded = 0
                    """,
                    (
                        record.updated_at.isoformat(),
                        record.user_id,
                        record.agent_id,
                        record.key,
                    ),
                )
                connection.execute(
                    f"""
                    INSERT INTO {table} (
                        id,
                        user_id,
                        agent_id,
                        source_session_id,
                        key,
                        value,
                        content,
                        summary,
                        confirmed,
                        confidence,
                        superseded,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.user_id,
                        record.agent_id,
                        record.source_session_id,
                        record.key,
                        record.value,
                        record.content,
                        record.summary,
                        int(record.confirmed),
                        record.confidence,
                        int(record.superseded),
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    ),
                )

    def update_record(
        self,
        record_id: str,
        patch: MemoryPatch,
        scope: MemoryScope,
    ) -> MemoryRecord:
        """Update a persisted long-term record and return the updated value."""

        for table, kind in (
            ("semantic_preferences", MemoryKind.SEMANTIC_PREFERENCE),
            ("semantic_facts", MemoryKind.SEMANTIC_FACT),
            ("episodic_memories", MemoryKind.EPISODIC),
            ("procedural_memories", MemoryKind.PROCEDURAL),
        ):
            with self._connect() as connection:
                row = connection.execute(
                    f"""
                    SELECT *
                    FROM {table}
                    WHERE id = ? AND user_id = ? AND agent_id = ?
                    """,
                    (record_id, scope.user_id, scope.agent_id),
                ).fetchone()
                if row is None:
                    continue

                assignments: list[str] = []
                values: list[object] = []
                if patch.content is not None:
                    assignments.append("content = ?")
                    values.append(patch.content)
                if patch.summary is not None:
                    assignments.append("summary = ?")
                    values.append(patch.summary)
                if patch.value is not None and table in {
                    "semantic_preferences",
                    "semantic_facts",
                }:
                    assignments.append("value = ?")
                    values.append(patch.value)
                if patch.confidence is not None:
                    assignments.append("confidence = ?")
                    values.append(patch.confidence)
                if patch.confirmed is not None and table in {
                    "semantic_preferences",
                    "semantic_facts",
                }:
                    assignments.append("confirmed = ?")
                    values.append(int(patch.confirmed))
                if patch.superseded is not None and table in {
                    "semantic_preferences",
                    "semantic_facts",
                }:
                    assignments.append("superseded = ?")
                    values.append(int(patch.superseded))

                if not assignments:
                    return _generic_record_from_row(table, row, kind)

                assignments.append("updated_at = ?")
                values.append(datetime.now(tz=UTC).isoformat())
                values.extend((record_id, scope.user_id, scope.agent_id))
                connection.execute(
                    f"""
                    UPDATE {table}
                    SET {", ".join(assignments)}
                    WHERE id = ? AND user_id = ? AND agent_id = ?
                    """,
                    tuple(values),
                )
                updated_row = connection.execute(
                    f"""
                    SELECT *
                    FROM {table}
                    WHERE id = ? AND user_id = ? AND agent_id = ?
                    """,
                    (record_id, scope.user_id, scope.agent_id),
                ).fetchone()
                if updated_row is None:
                    break
                return _generic_record_from_row(table, updated_row, kind)
        raise KeyError(f"Memory record '{record_id}' was not found.")

    def _ensure_schema(self) -> None:
        """Create the required SQLite tables if they do not exist."""

        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS semantic_preferences (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    confirmed INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    superseded INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    confirmed INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    superseded INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    content TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    tool_names TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS procedural_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_names TEXT NOT NULL,
                    success_count INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    last_applied_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_semantic_preferences_namespace
                ON semantic_preferences (user_id, agent_id, key, updated_at);

                CREATE INDEX IF NOT EXISTS idx_semantic_facts_namespace
                ON semantic_facts (user_id, agent_id, key, updated_at);

                CREATE INDEX IF NOT EXISTS idx_episodic_namespace
                ON episodic_memories (user_id, agent_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_procedural_namespace
                ON procedural_memories (user_id, agent_id, updated_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        """Return a SQLite connection configured for row access."""

        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _rank_rows(
        rows: Iterable[sqlite3.Row],
        *,
        query: str,
        vector_scores: dict[str, float] | None,
        success_weight: float,
    ) -> list[sqlite3.Row]:
        """Rank SQLite rows using hybrid lexical, vector, and recency scoring."""

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            memory_id = str(row["id"])
            content = f"{row['summary']} {row['content']}"
            lexical = _lexical_score(query, content)
            vector = 0.0 if vector_scores is None else vector_scores.get(memory_id, 0.0)
            recency = _recency_boost(str(row["updated_at"]))
            success = float(row["success"]) if "success" in row.keys() else 1.0
            success_count = (
                float(row["success_count"]) / 10.0
                if "success_count" in row.keys()
                else 0.0
            )
            score = (
                lexical + vector + recency + (success * success_weight) + success_count
            )
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored]


def _semantic_from_row(
    row: sqlite3.Row,
    *,
    kind: SemanticMemoryKind,
) -> SemanticMemoryRecord:
    """Build a semantic memory record from a SQLite row."""

    return SemanticMemoryRecord(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        agent_id=str(row["agent_id"]),
        source_session_id=str(row["source_session_id"]),
        kind=kind,
        key=str(row["key"]),
        value=str(row["value"]),
        content=str(row["content"]),
        summary=str(row["summary"]),
        confirmed=bool(row["confirmed"]),
        confidence=float(row["confidence"]),
        superseded=bool(row["superseded"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )


def _episodic_from_row(row: sqlite3.Row) -> EpisodicMemoryRecord:
    """Build an episodic memory record from a SQLite row."""

    tool_names = tuple(filter(None, str(row["tool_names"]).split(",")))
    return EpisodicMemoryRecord(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        agent_id=str(row["agent_id"]),
        source_session_id=str(row["source_session_id"]),
        task=str(row["task"]),
        summary=str(row["summary"]),
        content=str(row["content"]),
        success=bool(row["success"]),
        tool_names=tool_names,
        confidence=float(row["confidence"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )


def _procedural_from_row(row: sqlite3.Row) -> ProceduralMemoryRecord:
    """Build a procedural memory record from a SQLite row."""

    tool_names = tuple(filter(None, str(row["tool_names"]).split(",")))
    return ProceduralMemoryRecord(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        agent_id=str(row["agent_id"]),
        source_session_id=str(row["source_session_id"]),
        task_type=str(row["task_type"]),
        summary=str(row["summary"]),
        content=str(row["content"]),
        tool_names=tool_names,
        success_count=int(row["success_count"]),
        confidence=float(row["confidence"]),
        last_applied_at=datetime.fromisoformat(str(row["last_applied_at"])),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )


def _generic_record_from_row(
    table: str,
    row: sqlite3.Row,
    kind: MemoryKind,
) -> MemoryRecord:
    """Convert a raw SQLite row into a generic memory record."""

    metadata: dict[str, object] = {}
    key = None
    value = None
    confirmed = False
    if table in {"semantic_preferences", "semantic_facts"}:
        key = str(row["key"])
        value = str(row["value"])
        confirmed = bool(row["confirmed"])
        metadata["superseded"] = bool(row["superseded"])
    if table == "episodic_memories":
        metadata["task"] = str(row["task"])
        metadata["success"] = bool(row["success"])
        metadata["tool_names"] = tuple(filter(None, str(row["tool_names"]).split(",")))
    if table == "procedural_memories":
        metadata["task_type"] = str(row["task_type"])
        metadata["tool_names"] = tuple(filter(None, str(row["tool_names"]).split(",")))
        metadata["success_count"] = int(row["success_count"])

    return MemoryRecord(
        id=str(row["id"]),
        kind=kind,
        user_id=str(row["user_id"]),
        session_id=str(row["source_session_id"]),
        agent_id=str(row["agent_id"]),
        key=key,
        value=value,
        content=str(row["content"]),
        summary=str(row["summary"]),
        confidence=float(row["confidence"]),
        confirmed=confirmed,
        metadata=metadata,
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )


def _lexical_score(query: str, text: str) -> float:
    """Return a simple token-overlap score."""

    query_tokens = set(_tokenize(query))
    text_tokens = set(_tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / len(query_tokens)


def _tokenize(value: str) -> list[str]:
    """Split a string into lowercase search tokens."""

    return re.findall(r"[a-zA-Z0-9_]+", value.lower())


def _recency_boost(value: str) -> float:
    """Return a small boost for recent records."""

    updated_at = datetime.fromisoformat(value)
    age = datetime.now(tz=UTC) - updated_at
    return max(0.0, 0.2 - (age.days * 0.01))
