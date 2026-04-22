"""Document inspection helpers for structure and statistic questions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from hello_agents.apps.knowledge_qa.classifier import (
    QuestionClassification,
    QuestionType,
)
from hello_agents.apps.knowledge_qa.models import RetrievedChunk


class SupportsReadText(Protocol):
    """Describe the text file interface used by the inspector."""

    def read_text(self, encoding: str = "utf-8") -> str:
        """Return file text content."""


@dataclass(slots=True, frozen=True)
class DocumentInspectionResult:
    """Represent one direct document analysis result."""

    source: str
    operation: str
    output: str
    metadata: dict[str, object] = field(default_factory=dict)


class DocumentInspector:
    """Read and analyze target documents when retrieval alone is insufficient."""

    _HEADING_PATTERN = re.compile(r"^(#{2,6})\s+(.*?)\s*$", re.MULTILINE)
    _WORD_PATTERN = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")

    def inspect(
        self,
        *,
        classification: QuestionClassification,
        source_paths: tuple[str, ...],
        retrieved_chunks: tuple[RetrievedChunk, ...],
    ) -> DocumentInspectionResult | None:
        """Inspect a target document directly when the workflow requires it."""

        target_path = _resolve_target_path(
            target_files=classification.target_files,
            source_paths=source_paths,
            retrieved_chunks=retrieved_chunks,
        )
        if target_path is None or not target_path.exists() or not target_path.is_file():
            return None
        try:
            text = target_path.read_text(encoding="utf-8")
        except OSError:
            return None

        if classification.question_type is QuestionType.DOCUMENT_STRUCTURE:
            headings = [
                match.group(2).strip() for match in self._HEADING_PATTERN.finditer(text)
            ]
            section_count = len(headings)
            heading_text = ", ".join(headings) if headings else "none"
            return DocumentInspectionResult(
                source=str(target_path),
                operation="count_sections",
                output=(
                    f"The document has {section_count} sections. "
                    f"Section headings: {heading_text}."
                ),
                metadata={
                    "section_count": section_count,
                    "headings": headings,
                },
            )

        if classification.question_type is QuestionType.DOCUMENT_STATISTIC:
            word_count = len(self._WORD_PATTERN.findall(text))
            return DocumentInspectionResult(
                source=str(target_path),
                operation="count_words",
                output=f"The document contains {word_count} words.",
                metadata={"word_count": word_count},
            )
        return None


def _resolve_target_path(
    *,
    target_files: tuple[str, ...],
    source_paths: tuple[str, ...],
    retrieved_chunks: tuple[RetrievedChunk, ...],
) -> Path | None:
    """Resolve the best local file candidate for direct inspection."""

    for chunk in retrieved_chunks:
        candidate = Path(chunk.source)
        if candidate.exists() and candidate.is_file():
            if not target_files or candidate.name.lower() in target_files:
                return candidate

    for raw_path in source_paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if not target_files or path.name.lower() in target_files:
                return path
            continue
        if not path.exists():
            continue
        for target_file in target_files:
            matches = sorted(
                candidate
                for candidate in path.glob("**/*")
                if candidate.is_file() and candidate.name.lower() == target_file
            )
            if matches:
                return matches[0]
    return None
