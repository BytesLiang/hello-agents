"""Abstract contracts used by the memory subsystem."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from hello_agents.memory.models import (
    MemoryCommitResult,
    MemoryKind,
    MemoryPatch,
    MemoryProposal,
    MemoryQueryResult,
    MemoryRecord,
    MemoryScope,
    VectorDocument,
    VectorSearchHit,
)
from hello_agents.tools.base import ToolResult


class Embedder(Protocol):
    """Generate embeddings for vector-indexed memories."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return a dense embedding vector for each input text."""


class VectorStore(Protocol):
    """Persist and query vectorized long-term memories."""

    def upsert(self, document: VectorDocument, embedding: Sequence[float]) -> None:
        """Upsert a vector document."""

    def search(
        self,
        scope: MemoryScope,
        *,
        embedding: Sequence[float],
        memory_kinds: Sequence[MemoryKind],
        limit: int,
    ) -> list[VectorSearchHit]:
        """Search vectorized memories inside a namespace."""


class GraphStore(Protocol):
    """Represent a graph store adapter for future use."""

    def healthcheck(self) -> bool:
        """Return whether the graph backend is reachable."""


class MemoryAnalyzer(Protocol):
    """Analyze a completed turn into working records and long-term candidates."""

    def propose(
        self,
        message: str,
        response: str,
        *,
        scope: MemoryScope,
        tool_results: Sequence[ToolResult] = (),
        success: bool = True,
    ) -> MemoryProposal:
        """Return a memory proposal for a completed turn."""


@runtime_checkable
class Memory(Protocol):
    """Define the command-style memory interface used by agents and tools."""

    def query(
        self,
        message: str,
        *,
        scope: MemoryScope,
        kinds: Sequence[MemoryKind] | None = None,
        limit: int = 10,
    ) -> MemoryQueryResult:
        """Query structured memory context."""

    def add(self, record: MemoryRecord, *, scope: MemoryScope) -> MemoryRecord:
        """Directly persist a memory record."""

    def update(
        self,
        record_id: str,
        patch: MemoryPatch,
        *,
        scope: MemoryScope,
    ) -> MemoryRecord:
        """Update a memory record."""

    def propose(
        self,
        message: str,
        response: str,
        *,
        scope: MemoryScope,
        tool_results: Sequence[ToolResult] = (),
        success: bool = True,
    ) -> MemoryProposal:
        """Produce a memory proposal for a completed turn."""

    def commit(
        self,
        proposal: MemoryProposal,
        *,
        scope: MemoryScope,
    ) -> MemoryCommitResult:
        """Commit a proposal through policy and persistence."""
