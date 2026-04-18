"""Knowledge base metadata stores for the knowledge QA application."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from hello_agents.apps.knowledge_qa.models import (
    KnowledgeBase,
    KnowledgeBaseStatus,
    utc_now_iso,
)


class JsonKnowledgeBaseStore:
    """Persist knowledge base metadata in a local JSON file."""

    def __init__(self, path: Path) -> None:
        """Store the metadata file location."""

        self._path = path

    def save(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        """Insert or update a knowledge base record."""

        payload = self._load()
        updated = replace(knowledge_base, updated_at=utc_now_iso())
        payload[updated.kb_id] = _knowledge_base_to_dict(updated)
        self._write(payload)
        return updated

    def get(self, kb_id: str) -> KnowledgeBase | None:
        """Return one knowledge base by identifier."""

        payload = self._load()
        raw = payload.get(kb_id)
        if raw is None or not isinstance(raw, dict):
            return None
        return _knowledge_base_from_dict(raw)

    def list_all(self) -> tuple[KnowledgeBase, ...]:
        """Return all stored knowledge bases sorted by recency."""

        payload = self._load()
        knowledge_bases = [
            _knowledge_base_from_dict(raw)
            for raw in payload.values()
            if isinstance(raw, dict)
        ]
        knowledge_bases.sort(key=lambda item: item.updated_at, reverse=True)
        return tuple(knowledge_bases)

    def delete(self, kb_id: str) -> bool:
        """Delete one knowledge base record when it exists."""

        payload = self._load()
        removed = payload.pop(kb_id, None)
        if removed is None:
            return False
        self._write(payload)
        return True

    def mark_failed(self, kb_id: str) -> KnowledgeBase | None:
        """Mark one knowledge base as failed."""

        knowledge_base = self.get(kb_id)
        if knowledge_base is None:
            return None
        return self.save(replace(knowledge_base, status=KnowledgeBaseStatus.FAILED))

    def _load(self) -> dict[str, object]:
        """Load the raw JSON payload from disk."""

        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _write(self, payload: dict[str, object]) -> None:
        """Write the raw JSON payload to disk."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _knowledge_base_to_dict(knowledge_base: KnowledgeBase) -> dict[str, object]:
    """Serialize one knowledge base to a JSON-compatible dictionary."""

    return {
        "kb_id": knowledge_base.kb_id,
        "name": knowledge_base.name,
        "description": knowledge_base.description,
        "source_paths": list(knowledge_base.source_paths),
        "status": knowledge_base.status.value,
        "document_count": knowledge_base.document_count,
        "chunk_count": knowledge_base.chunk_count,
        "created_at": knowledge_base.created_at,
        "updated_at": knowledge_base.updated_at,
    }


def _knowledge_base_from_dict(payload: dict[str, object]) -> KnowledgeBase:
    """Deserialize one knowledge base from a JSON payload."""

    status = payload.get("status", KnowledgeBaseStatus.READY.value)
    source_paths = payload.get("source_paths", ())
    normalized_source_paths: tuple[str, ...] = ()
    if isinstance(source_paths, list):
        normalized_source_paths = tuple(str(item) for item in source_paths)
    return KnowledgeBase(
        kb_id=str(payload["kb_id"]),
        name=str(payload["name"]),
        description=str(payload.get("description", "")),
        source_paths=normalized_source_paths,
        status=KnowledgeBaseStatus(str(status)),
        document_count=_as_int(payload.get("document_count", 0)),
        chunk_count=_as_int(payload.get("chunk_count", 0)),
        created_at=str(payload.get("created_at", utc_now_iso())),
        updated_at=str(payload.get("updated_at", utc_now_iso())),
    )


def _as_int(value: object) -> int:
    """Convert a JSON payload value to an integer when possible."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return 0
