"""Define typed models shared by the memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(tz=UTC)


def new_memory_id() -> str:
    """Return a stable identifier for a memory record."""

    return uuid4().hex


class MemoryKind(StrEnum):
    """Enumerate the supported memory categories."""

    WORKING_PLAN = "working_plan"
    WORKING_CONTEXT = "working_context"
    WORKING_MESSAGE = "working_message"
    SEMANTIC_PREFERENCE = "semantic_preference"
    SEMANTIC_FACT = "semantic_fact"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class WorkingMemoryKind(StrEnum):
    """Enumerate supported working-memory entry categories."""

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_OBSERVATION = "tool_observation"
    PLAN = "plan"
    CONTEXT = "context"


class SemanticMemoryKind(StrEnum):
    """Enumerate supported semantic-memory categories."""

    PREFERENCE = "preference"
    FACT = "fact"


@dataclass(slots=True, frozen=True)
class MemoryScope:
    """Describe the namespace used for memory retrieval and persistence."""

    user_id: str
    session_id: str
    agent_id: str
    run_id: str | None = None


@dataclass(slots=True, frozen=True)
class MemoryRecord:
    """Represent a generic persisted memory record."""

    kind: MemoryKind
    user_id: str
    agent_id: str
    content: str
    id: str = field(default_factory=new_memory_id)
    session_id: str | None = None
    run_id: str | None = None
    key: str | None = None
    value: str | None = None
    summary: str = ""
    confidence: float = 0.0
    confirmed: bool = False
    pinned: bool = False
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True, frozen=True)
class MemoryCandidate:
    """Represent a long-term memory candidate awaiting commit."""

    kind: MemoryKind
    content: str
    summary: str
    key: str | None = None
    value: str | None = None
    confidence: float = 0.0
    confirmed: bool = False
    requires_confirmation: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RejectedMemoryCandidate:
    """Represent a candidate rejected by memory policy."""

    candidate: MemoryCandidate
    reason: str


@dataclass(slots=True, frozen=True)
class MemoryProposal:
    """Represent the analyzer output for a completed turn."""

    working_records: tuple[MemoryRecord, ...] = ()
    candidates: tuple[MemoryCandidate, ...] = ()


@dataclass(slots=True, frozen=True)
class MemoryCommitDecision:
    """Represent the policy decision for a proposal."""

    accepted: tuple[MemoryCandidate, ...] = ()
    rejected: tuple[RejectedMemoryCandidate, ...] = ()


@dataclass(slots=True, frozen=True)
class MemoryCommitResult:
    """Represent the records written after a commit attempt."""

    accepted_records: tuple[MemoryRecord, ...] = ()
    rejected: tuple[RejectedMemoryCandidate, ...] = ()


@dataclass(slots=True, frozen=True)
class MemoryPatch:
    """Represent a partial update applied to a stored memory record."""

    content: str | None = None
    summary: str | None = None
    value: str | None = None
    confidence: float | None = None
    confirmed: bool | None = None
    pinned: bool | None = None
    superseded: bool | None = None
    metadata: dict[str, object] | None = None


@dataclass(slots=True, frozen=True)
class MemoryQueryResult:
    """Bundle query results grouped by memory kind."""

    working: tuple[MemoryRecord, ...] = ()
    preferences: tuple[MemoryRecord, ...] = ()
    facts: tuple[MemoryRecord, ...] = ()
    episodes: tuple[MemoryRecord, ...] = ()
    procedures: tuple[MemoryRecord, ...] = ()

    @property
    def items(self) -> tuple[MemoryRecord, ...]:
        """Return the flattened records in display order."""

        return (
            *self.working,
            *self.preferences,
            *self.facts,
            *self.episodes,
            *self.procedures,
        )


@dataclass(slots=True, frozen=True)
class WorkingMemoryRecord:
    """Store a short-lived working-memory entry."""

    user_id: str
    session_id: str
    agent_id: str
    kind: WorkingMemoryKind
    content: str
    id: str = field(default_factory=new_memory_id)
    pinned: bool = False
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    expires_at: datetime | None = None


@dataclass(slots=True, frozen=True)
class EpisodicMemoryRecord:
    """Store a long-lived task history record."""

    user_id: str
    agent_id: str
    source_session_id: str
    task: str
    summary: str
    content: str
    success: bool
    id: str = field(default_factory=new_memory_id)
    tool_names: tuple[str, ...] = ()
    confidence: float = 0.7
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True, frozen=True)
class SemanticMemoryRecord:
    """Store a long-lived user preference or confirmed fact."""

    user_id: str
    agent_id: str
    source_session_id: str
    kind: SemanticMemoryKind
    key: str
    value: str
    content: str
    summary: str
    id: str = field(default_factory=new_memory_id)
    confirmed: bool = True
    confidence: float = 0.8
    superseded: bool = False
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    @property
    def memory_kind(self) -> MemoryKind:
        """Return the persisted memory kind for this semantic record."""

        if self.kind == SemanticMemoryKind.PREFERENCE:
            return MemoryKind.SEMANTIC_PREFERENCE
        return MemoryKind.SEMANTIC_FACT


@dataclass(slots=True, frozen=True)
class ProceduralMemoryRecord:
    """Store a reusable successful approach for similar tasks."""

    user_id: str
    agent_id: str
    source_session_id: str
    task_type: str
    summary: str
    content: str
    id: str = field(default_factory=new_memory_id)
    tool_names: tuple[str, ...] = ()
    success_count: int = 1
    confidence: float = 0.75
    last_applied_at: datetime = field(default_factory=utc_now)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True, frozen=True)
class VectorDocument:
    """Represent a vectorized mirror of a long-term memory record."""

    memory_id: str
    memory_kind: MemoryKind
    user_id: str
    agent_id: str
    content: str
    summary: str
    confidence: float
    created_at: datetime


@dataclass(slots=True, frozen=True)
class VectorSearchHit:
    """Represent a vector search result before reading from SQLite."""

    memory_id: str
    memory_kind: MemoryKind
    score: float
    payload: dict[str, object] = field(default_factory=dict)


MemoryContext = MemoryScope
MemoryType = MemoryKind
