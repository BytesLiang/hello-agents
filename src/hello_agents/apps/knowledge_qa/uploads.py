"""Helpers for persisting uploaded documents for knowledge QA ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


@dataclass(slots=True, frozen=True)
class SavedUpload:
    """Describe one uploaded document written to local storage."""

    original_name: str
    stored_path: Path
    size_bytes: int


class UploadedDocumentStore:
    """Persist uploaded documents under a local working directory."""

    def __init__(self, root_path: Path) -> None:
        """Store the upload root path."""

        self._root_path = root_path

    async def save_files(self, files: list[UploadFile]) -> tuple[SavedUpload, ...]:
        """Write uploaded files to a unique directory and return saved metadata."""

        upload_id = uuid4().hex
        upload_dir = self._root_path / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_files: list[SavedUpload] = []
        for index, upload in enumerate(files, start=1):
            filename = upload.filename or f"upload-{index}"
            target = upload_dir / _build_safe_filename(filename, index=index)
            payload = await upload.read()
            target.write_bytes(payload)
            saved_files.append(
                SavedUpload(
                    original_name=filename,
                    stored_path=target,
                    size_bytes=len(payload),
                )
            )
            await upload.close()
        return tuple(saved_files)


def _build_safe_filename(filename: str, *, index: int) -> str:
    """Return a filesystem-safe filename while keeping the original suffix."""

    source = Path(filename)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._-")
    suffix = re.sub(r"[^A-Za-z0-9.]+", "", "".join(source.suffixes))
    normalized_stem = stem or f"upload_{index}"
    normalized_suffix = suffix[:32]
    return f"{index:03d}_{normalized_stem}{normalized_suffix}"
