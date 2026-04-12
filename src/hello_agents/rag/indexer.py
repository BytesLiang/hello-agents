"""Index local files into a RAG vector store."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from hello_agents.memory.embeddings import build_embedder
from hello_agents.rag.config import RagConfig
from hello_agents.rag.models import RagChunk
from hello_agents.rag.qdrant_store import RagQdrantStore


class RagIndexer:
    """Index local files for retrieval-augmented generation."""

    def __init__(
        self,
        *,
        config: RagConfig,
        store: RagQdrantStore | None = None,
    ) -> None:
        """Prepare the indexer for local file ingestion."""

        if config.embed is None:
            raise ValueError("RAG indexing requires embedding configuration.")
        self._config = config
        self._embedder = build_embedder(config.embed)
        self._store = store or RagQdrantStore(config)

    def index_folder(self, path: Path, *, glob: str = "**/*") -> int:
        """Index all readable text files inside a folder."""

        chunks: list[RagChunk] = []
        for file_path in _iter_files(path, glob=glob):
            text = _read_text(file_path)
            if not text:
                continue
            for index, chunk_text in enumerate(
                _chunk_text(
                    text,
                    chunk_size=self._config.chunk_size,
                    overlap=self._config.chunk_overlap,
                )
            ):
                chunk_id = uuid4().hex
                chunks.append(
                    RagChunk(
                        id=chunk_id,
                        source=str(file_path),
                        content=chunk_text,
                        metadata={
                            "chunk_index": index,
                            "path": str(file_path),
                        },
                    )
                )

        if not chunks:
            return 0

        embeddings = self._embedder.embed_texts([chunk.content for chunk in chunks])
        self._store.upsert(chunks, embeddings)
        return len(chunks)


def _iter_files(path: Path, *, glob: str) -> Iterable[Path]:
    """Yield files matching the glob under a path."""

    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for file_path in path.glob(glob):
        if file_path.is_file():
            yield file_path


def _read_text(path: Path) -> str:
    """Read a file as UTF-8 text, ignoring decode errors."""

    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""

    if chunk_size <= 0:
        return []
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks
