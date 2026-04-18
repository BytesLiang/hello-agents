"""Ingestion helpers for the knowledge QA application."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class SupportsIndexFolder(Protocol):
    """Represent the indexing interface required by the app layer."""

    def index_folder(self, path: Path, *, glob: str = "**/*") -> int:
        """Index a folder or file path and return the chunk count."""


@dataclass(slots=True, frozen=True)
class IngestionResult:
    """Represent the result of indexing one or more source paths."""

    indexed_chunks: int
    indexed_documents: int


class KnowledgeBaseIngester:
    """Index one or more source paths for a knowledge base."""

    def __init__(self, indexer: SupportsIndexFolder) -> None:
        """Store the concrete indexer dependency."""

        self._indexer = indexer

    def ingest(self, paths: Sequence[Path]) -> IngestionResult:
        """Index the supplied source paths and count indexed documents."""

        indexed_chunks = 0
        indexed_documents = 0
        for path in paths:
            indexed_chunks += self._indexer.index_folder(path)
            indexed_documents += _count_documents(path)
        return IngestionResult(
            indexed_chunks=indexed_chunks,
            indexed_documents=indexed_documents,
        )


def _count_documents(path: Path) -> int:
    """Count the files represented by a path input."""

    if path.is_file():
        return 1
    if not path.exists():
        return 0
    return sum(1 for child in path.glob("**/*") if child.is_file())
