"""Index local files into a RAG vector store."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

try:
    from markitdown import MarkItDown  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised in import-only paths.
    MarkItDown = None  # type: ignore[assignment]

from hello_agents.memory.embeddings import build_embedder
from hello_agents.rag.config import RagConfig
from hello_agents.rag.models import RagChunk
from hello_agents.rag.qdrant_store import RagQdrantStore


class _ChunkBlock(TypedDict):
    """Represent one prepared chunk before persistence."""

    content: str
    heading_path: str


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
        if MarkItDown is None:
            raise ModuleNotFoundError("markitdown is required for RAG indexing.")
        self._config = config
        self._embedder = build_embedder(config.embed)
        self._store = store or RagQdrantStore(config)
        self._converter = MarkItDown()

    def index_folder(self, path: Path, *, glob: str = "**/*") -> int:
        """Index all readable text files inside a folder."""

        chunks: list[RagChunk] = []
        for file_path in _iter_files(path, glob=glob):
            text = self._read_text(file_path)
            if not text:
                continue
            for index, chunk in enumerate(
                _chunk_markdown(
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
                        content=chunk["content"],
                        metadata={
                            "chunk_index": index,
                            "path": str(file_path),
                            "heading_path": chunk["heading_path"],
                        },
                    )
                )

        if not chunks:
            return 0

        embeddings = self._embedder.embed_texts([chunk.content for chunk in chunks])
        self._store.upsert(chunks, embeddings)
        return len(chunks)

    def _read_text(self, path: Path) -> str:
        """Convert a file into Markdown text via MarkItDown."""

        try:
            result = self._converter.convert(str(path))
        except Exception:
            return ""
        text = getattr(result, "text_content", "")
        return text if isinstance(text, str) else ""


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


def _chunk_markdown(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[_ChunkBlock]:
    """Split Markdown by headings and paragraphs, with length fallback."""

    if chunk_size <= 0:
        return []

    blocks: list[_ChunkBlock] = []
    heading_stack: list[tuple[int, str]] = []
    paragraph_buffer: list[str] = []

    def flush_paragraphs() -> None:
        if not paragraph_buffer:
            return
        for paragraph in _split_paragraph_buffer(paragraph_buffer):
            blocks.extend(
                _paragraph_to_chunks(
                    paragraph,
                    heading_stack=heading_stack,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
        paragraph_buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading = _parse_heading(line)
        if heading is not None:
            flush_paragraphs()
            level, title = heading
            heading_stack[:] = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, title))
            continue
        if not line.strip():
            flush_paragraphs()
            continue
        paragraph_buffer.append(line)

    flush_paragraphs()
    return blocks


def _parse_heading(line: str) -> tuple[int, str] | None:
    """Parse an ATX Markdown heading."""

    match = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
    if match is None:
        return None
    return len(match.group(1)), match.group(2).strip()


def _split_paragraph_buffer(lines: list[str]) -> list[str]:
    """Normalize buffered lines into one or more paragraphs."""

    paragraph = "\n".join(line.strip() for line in lines).strip()
    if not paragraph:
        return []
    return [paragraph]


def _paragraph_to_chunks(
    paragraph: str,
    *,
    heading_stack: list[tuple[int, str]],
    chunk_size: int,
    overlap: int,
) -> list[_ChunkBlock]:
    """Turn one paragraph into one or more chunks."""

    heading_titles = [title for _, title in heading_stack]
    heading_path = " > ".join(heading_titles)
    prefix = ""
    if heading_path:
        prefix = f"Section: {heading_path}\n\n"

    text = prefix + paragraph
    if len(text) <= chunk_size:
        return [{"content": text, "heading_path": heading_path}]

    chunks: list[_ChunkBlock] = []
    for piece in _chunk_text(text, chunk_size=chunk_size, overlap=overlap):
        chunks.append({"content": piece, "heading_path": heading_path})
    return chunks
