"""File-level index entries: file name + chapter titles + opening text, one per file.

Used by "find a file" questions about a topic ("有哪些关于 Agent 评估的资料"), where a
file's outline says more than any single chunk. The summary is extractive (no LLM call):
it costs one embedding per file and cannot invent content. Entries are built from the
parser snapshot (``parsed/<doc_id>.json``), so files indexed before this feature are
filled in without re-parsing their originals.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from config import Settings, get_settings

logger = logging.getLogger(__name__)

# Bump when the entry text changes; older entries are rebuilt by ``backfill``.
FILE_ENTRY_VERSION = 1

_FRONT_MATTER = re.compile(
    r"封面|扉页|版权|目录|赞誉|译者序|推荐序|作者简介|致谢|内容介绍|内容简介|出版|^O'Reilly|copyright|contents|index|索引|参考文献",
    re.I,
)
_LEADER_DOTS = re.compile(r"(?:\.\s?){6,}|…{3,}|·{6,}")
MAX_TITLES = 24
MAX_OPENING = 240


def _titles(units: list[dict[str, Any]]) -> list[str]:
    """Top-level titles first, then second-level ones, skipping front matter."""
    levels: dict[int, list[str]] = {}
    for unit in units:
        path = [part.strip() for part in str(unit.get("heading", "")).split(" > ") if part.strip()]
        for depth, title in enumerate(path[:2]):
            if _FRONT_MATTER.search(title) or len(title) > 60:
                continue
            bucket = levels.setdefault(depth, [])
            if title not in bucket:
                bucket.append(title)
    titles: list[str] = []
    for depth in sorted(levels):
        for title in levels[depth]:
            if title not in titles:
                titles.append(title)
        # Only one top-level title (the document's own name): its sections say more.
        if len(titles) >= 8:
            break
    return titles[:MAX_TITLES]


def _opening(units: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    size = 0
    for unit in units:
        heading = str(unit.get("heading", ""))
        text = re.sub(r"\s+", " ", str(unit.get("text", ""))).strip()
        if not text or _FRONT_MATTER.search(heading) or _LEADER_DOTS.search(text):
            continue
        parts.append(text[: MAX_OPENING - size])
        size += len(parts[-1])
        if size >= MAX_OPENING:
            break
    return " ".join(parts)


def build_entry_text(name: str, parsed: dict[str, Any]) -> str:
    units = list(parsed.get("units") or [])
    stem = Path(name).stem or str(parsed.get("source", ""))
    lines = [stem]
    titles = _titles(units)
    if titles:
        lines.append("章节：" + " / ".join(titles))
    opening = _opening(units)
    if opening:
        lines.append(opening)
    return "\n".join(lines)


class FileIndexer:
    def __init__(self, store, embeddings, settings: Settings | None = None):
        self.store = store
        self.embeddings = embeddings
        self.settings = settings or get_settings()

    @property
    def supported(self) -> bool:
        return hasattr(self.store, "upsert_file_entries")

    def entry_text(self, doc_id: str, name: str | None = None) -> str | None:
        path = Path(self.settings.parsed_dir) / f"{doc_id}.json"
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.debug("No parser snapshot for %s: %s", doc_id, exc)
            return None
        return build_entry_text(name or str(parsed.get("source", "")), parsed)

    def index(self, files: Iterable[tuple[str, str]], batch_size: int = 32) -> int:
        """Build and store entries for ``(doc_id, name)`` pairs. Returns how many were written."""
        if not self.supported:
            return 0
        pending = [(doc_id, name, text) for doc_id, name in files if (text := self.entry_text(doc_id, name))]
        written = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            vectors = self.embeddings.embed_documents([text for _, _, text in batch])
            written += self.store.upsert_file_entries(
                (doc_id, name, text, FILE_ENTRY_VERSION, vector)
                for (doc_id, name, text), vector in zip(batch, vectors, strict=True)
            )
        return written

    def backfill(self, files: dict[str, str]) -> int:
        """Add or rebuild entries for indexed files (``doc_id -> name``) that lack a current one."""
        if not self.supported or not files:
            return 0
        versions = self.store.file_entry_versions()
        stale = [(doc_id, name) for doc_id, name in files.items() if versions.get(doc_id) != FILE_ENTRY_VERSION]
        if not stale:
            return 0
        written = self.index(stale)
        logger.info("File entries: %d built for %d files missing one", written, len(stale))
        return written
