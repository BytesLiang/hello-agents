"""Command-style memory orchestration."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

from hello_agents.memory.base import (
    Embedder,
    GraphStore,
    Memory,
    MemoryAnalyzer,
    VectorStore,
)
from hello_agents.memory.config import MemoryConfig
from hello_agents.memory.embeddings import build_embedder
from hello_agents.memory.extractors import RuleBasedMemoryAnalyzer
from hello_agents.memory.models import (
    EpisodicMemoryRecord,
    MemoryCandidate,
    MemoryCommitResult,
    MemoryKind,
    MemoryPatch,
    MemoryProposal,
    MemoryQueryResult,
    MemoryRecord,
    MemoryScope,
    ProceduralMemoryRecord,
    RejectedMemoryCandidate,
    SemanticMemoryKind,
    SemanticMemoryRecord,
    VectorDocument,
    WorkingMemoryKind,
    WorkingMemoryRecord,
)
from hello_agents.memory.stores import (
    InMemoryWorkingMemoryStore,
    Neo4jGraphStore,
    QdrantVectorStore,
    RedisWorkingMemoryStore,
    SQLiteMemoryStore,
)
from hello_agents.tools.base import ToolResult


class LayeredMemory(Memory):
    """Implement the command-style memory API with minimal indirection."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        analyzer: MemoryAnalyzer | None = None,
        working_store: (
            InMemoryWorkingMemoryStore | RedisWorkingMemoryStore | None
        ) = None,
        long_term_store: SQLiteMemoryStore | None = None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """Initialize the concrete stores and optional vector dependencies."""

        self.config = config or MemoryConfig.local_default()
        self.logger = logging.getLogger(self.__class__.__module__)
        self.working_store = working_store or self._build_working_store(self.config)
        self.long_term_store = long_term_store or SQLiteMemoryStore(self.config.sqlite)
        self.vector_store = vector_store or self._build_vector_store(self.config)
        self.graph_store = graph_store or self._build_graph_store(self.config)
        self.embedder = embedder or self._build_embedder(self.config)
        self.analyzer = analyzer or RuleBasedMemoryAnalyzer()

    def query(
        self,
        message: str,
        *,
        scope: MemoryScope,
        kinds: Sequence[MemoryKind] | None = None,
        limit: int = 10,
    ) -> MemoryQueryResult:
        """Query working and long-term memory relevant to the message."""

        requested = set(kinds or list(MemoryKind))
        vector_scores = self._search_vector_scores(message, scope=scope)
        working = tuple(
            record for record in self._list_working(scope) if record.kind in requested
        )
        return MemoryQueryResult(
            working=working,
            preferences=self._search_long_term(
                message,
                scope=scope,
                kind=MemoryKind.SEMANTIC_PREFERENCE,
                requested=requested,
                limit=limit,
                vector_scores=vector_scores,
            ),
            facts=self._search_long_term(
                message,
                scope=scope,
                kind=MemoryKind.SEMANTIC_FACT,
                requested=requested,
                limit=limit,
                vector_scores=vector_scores,
            ),
            episodes=self._search_long_term(
                message,
                scope=scope,
                kind=MemoryKind.EPISODIC,
                requested=requested,
                limit=limit,
                vector_scores=vector_scores,
            ),
            procedures=self._search_long_term(
                message,
                scope=scope,
                kind=MemoryKind.PROCEDURAL,
                requested=requested,
                limit=limit,
                vector_scores=vector_scores,
            ),
        )

    def add(self, record: MemoryRecord, *, scope: MemoryScope) -> MemoryRecord:
        """Persist a working or long-term memory record immediately."""

        self._validate_scope(record, scope)
        if _is_working_kind(record.kind):
            return self._add_working([record], scope=scope)[0]

        written = self._save_long_term([record])
        self._index_records(written)
        return written[0]

    def update(
        self,
        record_id: str,
        patch: MemoryPatch,
        *,
        scope: MemoryScope,
    ) -> MemoryRecord:
        """Update a persisted long-term memory record."""

        return self.long_term_store.update_record(record_id, patch, scope)

    def propose(
        self,
        message: str,
        response: str,
        *,
        scope: MemoryScope,
        tool_results: Sequence[ToolResult] = (),
        success: bool = True,
    ) -> MemoryProposal:
        """Analyze a completed turn into memory commands."""

        return self.analyzer.propose(
            message,
            response,
            scope=scope,
            tool_results=tool_results,
            success=success,
        )

    def commit(
        self,
        proposal: MemoryProposal,
        *,
        scope: MemoryScope,
    ) -> MemoryCommitResult:
        """Commit a proposal, applying fixed policy checks before persistence."""

        accepted_records: list[MemoryRecord] = []
        rejected: list[RejectedMemoryCandidate] = []

        if proposal.working_records:
            accepted_records.extend(
                self._add_working(proposal.working_records, scope=scope)
            )

        accepted_candidates: list[MemoryCandidate] = []
        for candidate in proposal.candidates:
            reason = self._reject_reason(candidate)
            if reason is None:
                accepted_candidates.append(candidate)
            else:
                rejected.append(
                    RejectedMemoryCandidate(candidate=candidate, reason=reason)
                )

        if accepted_candidates:
            written = self._save_long_term(
                [
                    _candidate_to_record(candidate, scope)
                    for candidate in accepted_candidates
                ]
            )
            accepted_records.extend(written)
            self._index_records(written)

        return MemoryCommitResult(
            accepted_records=tuple(accepted_records),
            rejected=tuple(rejected),
        )

    def _list_working(self, scope: MemoryScope) -> list[MemoryRecord]:
        """Load working-memory records and normalize them to generic records."""

        return [
            _from_working_record(record)
            for record in self.working_store.list_entries(scope)
        ]

    def _add_working(
        self,
        records: Sequence[MemoryRecord],
        *,
        scope: MemoryScope,
    ) -> list[MemoryRecord]:
        """Persist working-memory records."""

        working_records = [_to_working_record(record, scope) for record in records]
        self.working_store.append_entries(scope, working_records)
        return [_from_working_record(record) for record in working_records]

    def _save_long_term(
        self,
        records: Sequence[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Persist long-term records through the SQLite truth store."""

        saved: list[MemoryRecord] = []
        preferences = [
            _to_semantic_record(record, SemanticMemoryKind.PREFERENCE)
            for record in records
            if record.kind == MemoryKind.SEMANTIC_PREFERENCE
        ]
        facts = [
            _to_semantic_record(record, SemanticMemoryKind.FACT)
            for record in records
            if record.kind == MemoryKind.SEMANTIC_FACT
        ]
        episodes = [
            _to_episodic_record(record)
            for record in records
            if record.kind == MemoryKind.EPISODIC
        ]
        procedures = [
            _to_procedural_record(record)
            for record in records
            if record.kind == MemoryKind.PROCEDURAL
        ]

        saved.extend(
            _from_semantic_record(record)
            for record in self.long_term_store.save_preferences(preferences)
        )
        saved.extend(
            _from_semantic_record(record)
            for record in self.long_term_store.save_facts(facts)
        )
        for episode in episodes:
            saved.append(
                _from_episodic_record(self.long_term_store.save_episode(episode))
            )
        saved.extend(
            _from_procedural_record(record)
            for record in self.long_term_store.save_procedures(procedures)
        )
        return saved

    def _search_long_term(
        self,
        message: str,
        *,
        scope: MemoryScope,
        kind: MemoryKind,
        requested: set[MemoryKind],
        limit: int,
        vector_scores: dict[MemoryKind, dict[str, float]],
    ) -> tuple[MemoryRecord, ...]:
        """Search one long-term memory kind when requested."""

        if kind not in requested:
            return ()
        if kind == MemoryKind.SEMANTIC_PREFERENCE:
            return tuple(
                _from_semantic_record(record)
                for record in self.long_term_store.search_semantic(
                    scope,
                    kind=SemanticMemoryKind.PREFERENCE,
                    query=message,
                    limit=limit,
                    vector_scores=vector_scores[MemoryKind.SEMANTIC_PREFERENCE],
                )
            )
        if kind == MemoryKind.SEMANTIC_FACT:
            return tuple(
                _from_semantic_record(record)
                for record in self.long_term_store.search_semantic(
                    scope,
                    kind=SemanticMemoryKind.FACT,
                    query=message,
                    limit=limit,
                    vector_scores=vector_scores[MemoryKind.SEMANTIC_FACT],
                )
            )
        if kind == MemoryKind.EPISODIC:
            return tuple(
                _from_episodic_record(record)
                for record in self.long_term_store.search_episodes(
                    scope,
                    query=message,
                    limit=limit,
                    vector_scores=vector_scores[MemoryKind.EPISODIC],
                )
            )
        if kind == MemoryKind.PROCEDURAL:
            return tuple(
                _from_procedural_record(record)
                for record in self.long_term_store.search_procedures(
                    scope,
                    query=message,
                    limit=limit,
                    vector_scores=vector_scores[MemoryKind.PROCEDURAL],
                )
            )
        return ()

    def _search_vector_scores(
        self,
        message: str,
        *,
        scope: MemoryScope,
    ) -> dict[MemoryKind, dict[str, float]]:
        """Search optional vector storage and group scores by memory kind."""

        scores: dict[MemoryKind, dict[str, float]] = defaultdict(dict)
        if not message or self.vector_store is None or self.embedder is None:
            return scores
        try:
            embeddings = self.embedder.embed_texts([message])
            if not embeddings:
                return scores
            hits = self.vector_store.search(
                scope,
                embedding=embeddings[0],
                memory_kinds=(
                    MemoryKind.EPISODIC,
                    MemoryKind.PROCEDURAL,
                    MemoryKind.SEMANTIC_PREFERENCE,
                    MemoryKind.SEMANTIC_FACT,
                ),
                limit=12,
            )
        except Exception:
            return scores

        for hit in hits:
            scores[hit.memory_kind][hit.memory_id] = hit.score
        return scores

    @staticmethod
    def _reject_reason(candidate: MemoryCandidate) -> str | None:
        """Return a rejection reason when a candidate should not be committed."""

        if not candidate.content.strip() or not candidate.summary.strip():
            return "missing_content"

        if candidate.kind == MemoryKind.SEMANTIC_PREFERENCE:
            if candidate.confidence < 0.65:
                return "low_confidence_preference"
            return None

        if candidate.kind == MemoryKind.SEMANTIC_FACT:
            if not candidate.confirmed:
                return "fact_not_confirmed"
            if candidate.confidence < 0.75:
                return "low_confidence_fact"
            return None

        if candidate.kind == MemoryKind.PROCEDURAL:
            task_type = str(candidate.metadata.get("task_type", "general_task"))
            tool_count = _metadata_int(candidate.metadata, "tool_count", default=0)
            if candidate.confidence < 0.75:
                return "low_confidence_procedure"
            if tool_count == 0 and task_type == "general_task":
                return "generic_procedure"
            return None

        if candidate.kind == MemoryKind.EPISODIC:
            return None

        return "unsupported_kind"

    def _index_records(self, records: Sequence[MemoryRecord]) -> None:
        """Index committed long-term records into the optional vector store."""

        if not records or self.vector_store is None or self.embedder is None:
            return

        vector_records = [
            record for record in records if _is_long_term_kind(record.kind)
        ]
        if not vector_records:
            return
        try:
            embeddings = self.embedder.embed_texts(
                [f"{record.summary}\n{record.content}" for record in vector_records]
            )
        except Exception:
            self.logger.exception("Embedding failed; continuing without vector index.")
            return

        for record, embedding in zip(vector_records, embeddings, strict=True):
            try:
                self.vector_store.upsert(_record_to_vector_document(record), embedding)
            except Exception:
                self.logger.exception(
                    "Vector indexing failed for memory_id=%s", record.id
                )

    @staticmethod
    def _validate_scope(record: MemoryRecord, scope: MemoryScope) -> None:
        """Validate that the record belongs to the provided scope."""

        if record.user_id != scope.user_id or record.agent_id != scope.agent_id:
            raise ValueError("Memory record namespace does not match the target scope.")

    @staticmethod
    def _build_working_store(
        config: MemoryConfig,
    ) -> InMemoryWorkingMemoryStore | RedisWorkingMemoryStore:
        """Create the configured working-memory store."""

        if config.redis.enabled:
            return RedisWorkingMemoryStore(config.working, config.redis)
        return InMemoryWorkingMemoryStore(config.working)

    @staticmethod
    def _build_vector_store(config: MemoryConfig) -> VectorStore | None:
        """Create the configured vector store if enabled."""

        if config.qdrant.enabled:
            return QdrantVectorStore(config.qdrant)
        return None

    @staticmethod
    def _build_graph_store(config: MemoryConfig) -> GraphStore | None:
        """Create the configured graph store if enabled."""

        if config.neo4j.enabled:
            return Neo4jGraphStore(config.neo4j)
        return None

    @staticmethod
    def _build_embedder(config: MemoryConfig) -> Embedder | None:
        """Create the configured embedder if vector search is enabled."""

        if not config.qdrant.enabled or config.embed is None:
            return None
        return build_embedder(config.embed)


def _is_working_kind(kind: MemoryKind) -> bool:
    """Return whether the kind is stored in working memory."""

    return kind in {
        MemoryKind.WORKING_PLAN,
        MemoryKind.WORKING_CONTEXT,
        MemoryKind.WORKING_MESSAGE,
    }


def _is_long_term_kind(kind: MemoryKind) -> bool:
    """Return whether the kind is eligible for long-term storage."""

    return kind in {
        MemoryKind.SEMANTIC_PREFERENCE,
        MemoryKind.SEMANTIC_FACT,
        MemoryKind.EPISODIC,
        MemoryKind.PROCEDURAL,
    }


def _candidate_to_record(
    candidate: MemoryCandidate,
    scope: MemoryScope,
) -> MemoryRecord:
    """Convert a candidate into a generic memory record."""

    return MemoryRecord(
        kind=candidate.kind,
        user_id=scope.user_id,
        session_id=scope.session_id,
        agent_id=scope.agent_id,
        run_id=scope.run_id,
        key=candidate.key,
        value=candidate.value,
        content=candidate.content,
        summary=candidate.summary,
        confidence=candidate.confidence,
        confirmed=candidate.confirmed,
        metadata=dict(candidate.metadata),
    )


def _record_to_vector_document(record: MemoryRecord) -> VectorDocument:
    """Convert a committed record into a vector document."""

    return VectorDocument(
        memory_id=record.id,
        memory_kind=record.kind,
        user_id=record.user_id,
        agent_id=record.agent_id,
        content=record.content,
        summary=record.summary,
        confidence=record.confidence,
        created_at=record.created_at,
    )


def _from_working_record(record: WorkingMemoryRecord) -> MemoryRecord:
    """Convert a working-memory entry into a generic memory record."""

    kind_map = {
        WorkingMemoryKind.PLAN: MemoryKind.WORKING_PLAN,
        WorkingMemoryKind.CONTEXT: MemoryKind.WORKING_CONTEXT,
        WorkingMemoryKind.TOOL_OBSERVATION: MemoryKind.WORKING_CONTEXT,
        WorkingMemoryKind.USER_MESSAGE: MemoryKind.WORKING_MESSAGE,
        WorkingMemoryKind.ASSISTANT_MESSAGE: MemoryKind.WORKING_MESSAGE,
    }
    return MemoryRecord(
        id=record.id,
        kind=kind_map[record.kind],
        user_id=record.user_id,
        session_id=record.session_id,
        agent_id=record.agent_id,
        content=record.content,
        summary=record.content,
        pinned=record.pinned,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_working_record(record: MemoryRecord, scope: MemoryScope) -> WorkingMemoryRecord:
    """Convert a generic memory record into a working-memory entry."""

    kind_map = {
        MemoryKind.WORKING_PLAN: WorkingMemoryKind.PLAN,
        MemoryKind.WORKING_CONTEXT: WorkingMemoryKind.CONTEXT,
        MemoryKind.WORKING_MESSAGE: WorkingMemoryKind.CONTEXT,
    }
    target_kind = kind_map[record.kind]
    if record.metadata.get("message_role") == "user":
        target_kind = WorkingMemoryKind.USER_MESSAGE
    if record.metadata.get("message_role") == "assistant":
        target_kind = WorkingMemoryKind.ASSISTANT_MESSAGE
    if record.metadata.get("source") == "tool":
        target_kind = WorkingMemoryKind.TOOL_OBSERVATION
    return WorkingMemoryRecord(
        id=record.id,
        user_id=scope.user_id,
        session_id=scope.session_id,
        agent_id=scope.agent_id,
        kind=target_kind,
        content=record.content,
        pinned=record.pinned,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_semantic_record(
    record: MemoryRecord,
    kind: SemanticMemoryKind,
) -> SemanticMemoryRecord:
    """Convert a generic record into a semantic SQLite record."""

    key = record.key or (record.value or record.summary or record.content)
    value = record.value or record.content
    return SemanticMemoryRecord(
        id=record.id,
        user_id=record.user_id,
        agent_id=record.agent_id,
        source_session_id=record.session_id or "",
        kind=kind,
        key=key,
        value=value,
        content=record.content,
        summary=record.summary or value,
        confirmed=record.confirmed,
        confidence=record.confidence,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _from_semantic_record(record: SemanticMemoryRecord) -> MemoryRecord:
    """Convert a semantic SQLite record into a generic memory record."""

    return MemoryRecord(
        id=record.id,
        kind=record.memory_kind,
        user_id=record.user_id,
        session_id=record.source_session_id,
        agent_id=record.agent_id,
        key=record.key,
        value=record.value,
        content=record.content,
        summary=record.summary,
        confidence=record.confidence,
        confirmed=record.confirmed,
        metadata={"superseded": record.superseded},
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_episodic_record(record: MemoryRecord) -> EpisodicMemoryRecord:
    """Convert a generic record into an episodic SQLite record."""

    tool_names = _metadata_strings(record.metadata, "tool_names")
    success = bool(record.metadata.get("success", True))
    task = str(record.metadata.get("task", record.summary or record.content))
    return EpisodicMemoryRecord(
        id=record.id,
        user_id=record.user_id,
        agent_id=record.agent_id,
        source_session_id=record.session_id or "",
        task=task,
        summary=record.summary,
        content=record.content,
        success=success,
        tool_names=tool_names,
        confidence=record.confidence,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _from_episodic_record(record: EpisodicMemoryRecord) -> MemoryRecord:
    """Convert an episodic SQLite record into a generic memory record."""

    return MemoryRecord(
        id=record.id,
        kind=MemoryKind.EPISODIC,
        user_id=record.user_id,
        session_id=record.source_session_id,
        agent_id=record.agent_id,
        content=record.content,
        summary=record.summary,
        confidence=record.confidence,
        metadata={
            "task": record.task,
            "success": record.success,
            "tool_names": record.tool_names,
        },
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_procedural_record(record: MemoryRecord) -> ProceduralMemoryRecord:
    """Convert a generic record into a procedural SQLite record."""

    tool_names = _metadata_strings(record.metadata, "tool_names")
    task_type = str(record.metadata.get("task_type", "general_task"))
    success_count = _metadata_int(record.metadata, "success_count", default=1)
    return ProceduralMemoryRecord(
        id=record.id,
        user_id=record.user_id,
        agent_id=record.agent_id,
        source_session_id=record.session_id or "",
        task_type=task_type,
        summary=record.summary,
        content=record.content,
        tool_names=tool_names,
        success_count=success_count,
        confidence=record.confidence,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _metadata_strings(
    metadata: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    """Return a metadata value normalized as a tuple of strings."""

    value = metadata.get(key, ())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _metadata_int(
    metadata: dict[str, object],
    key: str,
    *,
    default: int,
) -> int:
    """Return an integer metadata value with a safe fallback."""

    value = metadata.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _from_procedural_record(record: ProceduralMemoryRecord) -> MemoryRecord:
    """Convert a procedural SQLite record into a generic memory record."""

    return MemoryRecord(
        id=record.id,
        kind=MemoryKind.PROCEDURAL,
        user_id=record.user_id,
        session_id=record.source_session_id,
        agent_id=record.agent_id,
        content=record.content,
        summary=record.summary,
        confidence=record.confidence,
        metadata={
            "task_type": record.task_type,
            "tool_names": record.tool_names,
            "success_count": record.success_count,
        },
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
