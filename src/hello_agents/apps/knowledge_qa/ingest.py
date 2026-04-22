"""Ingestion helpers for the knowledge QA application."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4


class SupportsDocumentIndex(Protocol):
    """Represent the document indexing interface required by the app layer."""

    def index_file(self, path: Path, *, kb_id: str, document_id: str) -> int:
        """Index one file and return its chunk count."""

    def delete_document(self, *, kb_id: str, document_id: str) -> None:
        """Delete one indexed document from the vector store."""


@dataclass(slots=True, frozen=True)
class IndexedDocument:
    """Describe one indexed document and its resulting chunk count."""

    document_id: str
    name: str
    source_path: str
    chunk_count: int
    size_bytes: int


@dataclass(slots=True, frozen=True)
class IngestionResult:
    """Represent the result of indexing one or more documents."""

    indexed_chunks: int
    indexed_documents: int
    documents: tuple[IndexedDocument, ...]


class KnowledgeBaseIngester:
    """Index one or more source paths for a knowledge base."""

    def __init__(self, indexer: SupportsDocumentIndex) -> None:
        """Store the concrete indexer dependency."""

        self._indexer = indexer

    def ingest(
        self,
        kb_id: str,
        paths: Sequence[Path],
        *,
        existing_source_paths: Sequence[str] = (),
    ) -> IngestionResult:
        """Index the supplied source paths and return document-level metadata."""

        documents = resolve_documents(paths)
        if not documents:
            raise ValueError("No readable documents were found to ingest.")

        seen_existing = {_normalize_path(Path(path)) for path in existing_source_paths}
        indexed_documents: list[IndexedDocument] = []
        indexed_chunks = 0
        for document_path in documents:
            normalized_path = _normalize_path(document_path)
            if normalized_path in seen_existing:
                continue
            document_id = uuid4().hex
            chunk_count = self._indexer.index_file(
                document_path,
                kb_id=kb_id,
                document_id=document_id,
            )
            indexed_documents.append(
                IndexedDocument(
                    document_id=document_id,
                    name=document_path.name,
                    source_path=str(document_path),
                    chunk_count=chunk_count,
                    size_bytes=_file_size(document_path),
                )
            )
            indexed_chunks += chunk_count
            seen_existing.add(normalized_path)

        if not indexed_documents:
            raise ValueError(
                "All selected documents are already in this knowledge base."
            )

        return IngestionResult(
            indexed_chunks=indexed_chunks,
            indexed_documents=len(indexed_documents),
            documents=tuple(indexed_documents),
        )

    def delete_document(self, *, kb_id: str, document_id: str) -> None:
        """Delete one indexed document from the backing store."""

        self._indexer.delete_document(kb_id=kb_id, document_id=document_id)


def resolve_documents(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Expand file and directory inputs into concrete document paths."""

    documents: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        for candidate in _iter_documents(raw_path.expanduser()):
            normalized = _normalize_path(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            documents.append(candidate)
    return tuple(documents)


def _iter_documents(path: Path) -> Iterable[Path]:
    """Yield files represented by one user-provided path."""

    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for child in sorted(path.glob("**/*")):
        if child.is_file():
            yield child


def _normalize_path(path: Path) -> str:
    """Return a normalized string form for deduplication."""

    try:
        return str(path.resolve(strict=False))
    except OSError:
        return str(path)


def _file_size(path: Path) -> int:
    """Return the size of one document in bytes when available."""

    try:
        return path.stat().st_size
    except OSError:
        return 0
