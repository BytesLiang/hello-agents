"""Define data models for the lightweight RAG subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class RagChunk:
    """Represent a retrieved RAG chunk."""

    id: str
    source: str
    content: str
    score: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
