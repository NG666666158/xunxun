"""Durable local store for complete parent passages.

Milvus contains only retrieval children. This store keeps the complete parent
text so a child hit can be expanded before answer generation.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import threading
from typing import Any, Iterable

from config import Settings, get_settings


class ParentStoreManager:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.directory = Path(self.settings.parent_store_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, dict[str, str]]] = {}

    def has_document(self, doc_id: str) -> bool:
        return self._path(doc_id).is_file()

    def save_document(self, doc_id: str, parents: Iterable[Any]) -> int:
        rows = [asdict(parent) for parent in parents]
        if any(row["doc_id"] != doc_id for row in rows):
            raise ValueError("parent doc_id does not match target document")
        payload = {row["parent_id"]: row for row in rows}
        target = self._path(doc_id)
        temporary = target.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(target)
            self._cache[doc_id] = payload
        return len(payload)

    def get(self, parent_id: str) -> dict[str, str] | None:
        doc_id = self._doc_id(parent_id)
        parents = self._load(doc_id)
        value = parents.get(parent_id)
        return dict(value) if value else None

    def get_many(self, parent_ids: Iterable[str]) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        seen: set[str] = set()
        for parent_id in parent_ids:
            if parent_id in seen:
                continue
            seen.add(parent_id)
            value = self.get(parent_id)
            if value:
                found.append(value)
        return found

    def list_parents(self, doc_id: str) -> list[dict[str, str]]:
        """Return a document's parents in their original order."""
        parents = self._load(doc_id)
        return [
            dict(parents[key])
            for key in sorted(parents, key=lambda parent_id: int(parent_id.rpartition("-p")[2]))
        ]

    def rename_source(self, doc_id: str, source: str) -> None:
        """Rewrite the ``source`` of a renamed file's parents."""
        parents = self._load(doc_id)
        if parents:
            self._write(doc_id, {key: {**value, "source": source} for key, value in parents.items()})

    def reassign_document(self, old_doc_id: str, new_doc_id: str) -> None:
        """Move parents to a new document id; parent ids embed it (``<doc_id>-p<n>``)."""
        parents = self._load(old_doc_id)
        if not parents:
            return
        moved = {}
        for parent_id, value in parents.items():
            new_parent_id = new_doc_id + parent_id[len(old_doc_id):]
            moved[new_parent_id] = {**value, "parent_id": new_parent_id, "doc_id": new_doc_id}
        self._write(new_doc_id, moved)
        self.delete_document(old_doc_id)

    def _write(self, doc_id: str, payload: dict[str, dict[str, str]]) -> None:
        target = self._path(doc_id)
        temporary = target.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(target)
            self._cache[doc_id] = payload

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self._cache.pop(doc_id, None)
            self._path(doc_id).unlink(missing_ok=True)

    def _load(self, doc_id: str) -> dict[str, dict[str, str]]:
        with self._lock:
            if doc_id in self._cache:
                return self._cache[doc_id]
            path = self._path(doc_id)
            if not path.is_file():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"invalid parent store file: {path}")
            self._cache[doc_id] = payload
            return payload

    def _path(self, doc_id: str) -> Path:
        if not doc_id or any(char not in "0123456789abcdef" for char in doc_id):
            raise ValueError("invalid document id")
        return self.directory / f"{doc_id}.json"

    @staticmethod
    def _doc_id(parent_id: str) -> str:
        doc_id, separator, number = parent_id.rpartition("-p")
        if not separator or not number.isdigit():
            raise ValueError("invalid parent id")
        return doc_id
