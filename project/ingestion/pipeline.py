"""Transactional document parsing, chunking, embedding and indexing."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config import Settings, get_settings
from db.parent_store_manager import ParentStoreManager
from retrieval.types import ChunkRecord, ChunkStore, text_key
from services.model_clients import SiliconFlowEmbeddings
from . import embed_pacing
from .chunker import HeadingAwareChunker
from .models import DocumentUnit, ParsedDocument
from .parsers import MultiFormatParser


logger = logging.getLogger(__name__)


def _worth_waiting(exc: BaseException) -> bool:
    """A busy or unreachable service (429, 5xx, timeouts, dropped connections), not a refused
    key or a bad request."""
    try:
        from services.model_clients import transient_error
    except ImportError:  # pragma: no cover - older model_clients
        return False
    return any(error is not None and transient_error(error) for error in (exc, exc.__cause__))


@dataclass(frozen=True)
class IngestionResult:
    doc_id: str
    source: str
    chunks: int
    status: str


class IngestionPipeline:
    def __init__(
        self,
        store: ChunkStore,
        embeddings: SiliconFlowEmbeddings,
        settings: Settings | None = None,
        parent_store: ParentStoreManager | None = None,
    ):
        self.settings = settings or get_settings()
        self.store = store
        self.embeddings = embeddings
        self.parent_store = parent_store or ParentStoreManager(self.settings)
        self.parser = MultiFormatParser(self.settings)
        self.chunker = HeadingAwareChunker(self.settings)
        self._lock = threading.Lock()
        # The library's embedding requests keep to the account's token limit (embed_pacing.py).
        embed_pacing.install_for(self.settings)

    # Children embedded per request; the queue may pause between batches (see ``pause``).
    # Larger requests are no faster per child once the account's token limit is reached
    # (evaluation/import_speed_v1); 32 is what earlier indexes were built with.
    embed_batch = 32
    # Seconds to wait before trying a batch again when the service stayed busy through the
    # client's own retries (see _embed_patiently).
    embed_patience = (20.0, 40.0, 60.0)
    # A scanned PDF is read and committed this many pages at a time: it is searchable as it
    # goes, and each commit embeds only the new chunks (the rest keep their vectors).
    ocr_pages_per_commit = 16

    def index_document(
        self,
        path: str | Path,
        doc_id: str,
        source: str | None = None,
        *,
        image_mode: str | None = None,
        pause: Callable[[], None] | None = None,
        report: dict[str, Any] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[int, int]:
        """(Re)index one registry file under its stable ``file_id``.

        Replaces whatever chunks the id had. Parsing and embedding run before the old
        chunks are touched, so a failure leaves the previous version searchable.
        Children whose text is unchanged keep their stored vector (looked up by text), so
        editing one paragraph of a long file embeds only the chunks around it.

        A scanned PDF is read ``ocr_pages_per_commit`` pages at a time and committed after
        each batch, so its first pages can be searched while the rest are being read.

        ``image_mode`` is the file's own image choice (see ``MultiFormatParser.parse``);
        ``pause`` is called before parsing, between recognised pages and between embedding
        requests (the library waits there while a question is answered); ``progress(read,
        total)`` follows a scan's pages; ``report`` receives what happened (embedded /
        reused children, commits, the parser's notice). Returns ``(chunks, characters)``.
        """
        if pause:
            pause()
        page_texts: dict[int, str] = {}
        totals = {"embedded": 0, "reused": 0, "embedding_requests": 0, "commits": 0}
        while True:
            document = self.parser.parse(
                path, doc_id=doc_id, image_mode=image_mode,
                page_limit=self.ocr_pages_per_commit, page_texts=page_texts, on_page=progress, pause=pause,
            )
            if source:
                document.source = source
            step: dict[str, Any] = {}
            chunks, characters = self._commit(doc_id, document, pause, step)
            for key in ("embedded", "reused", "embedding_requests"):
                totals[key] += step.get(key, 0)
            totals["commits"] += 1
            if not (document.metadata or {}).get("pages_pending"):
                break
            if pause:
                pause()
        if report is not None:
            report.update(totals, notice=(document.metadata or {}).get("notice"))
        return chunks, characters

    def _commit(
        self, doc_id: str, document: ParsedDocument, pause: Callable[[], None] | None, report: dict[str, Any]
    ) -> tuple[int, int]:
        """Chunk, embed (reusing stored vectors) and replace the document's index in one go."""
        if pause:
            pause()
        parents, drafts = self.chunker.split_parent_child(document)
        if not drafts:
            raise ValueError(f"{document.source} 未生成有效文本块")
        vectors = self._vectors(doc_id, [draft.text for draft in drafts], pause, report)
        records = [
            ChunkRecord(
                id=draft.id,
                doc_id=draft.doc_id,
                parent_id=draft.parent_id,
                source=draft.source,
                heading=draft.heading,
                locator=draft.locator,
                text=draft.text,
                dense=vector,
            )
            for draft, vector in zip(drafts, vectors, strict=True)
        ]
        if pause:
            pause()
        with self._lock:
            replace = getattr(self.store, "replace_document", None)
            if replace is not None:
                replace(doc_id, records)  # one transaction: searches never miss the document
            else:
                self.store.delete_document(doc_id)
                self.store.insert(records)
            self.parent_store.save_document(doc_id, parents)
            (self.settings.parsed_dir / f"{doc_id}.json").write_text(
                json.dumps(document.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return len(records), sum(len(unit.text) for unit in document.units)

    def _vectors(
        self,
        doc_id: str,
        texts: list[str],
        pause: Callable[[], None] | None,
        report: dict[str, Any] | None,
    ) -> list[list[float]]:
        """One vector per text: the stored one when this document already had that exact
        text (stores that can look it up), otherwise a new embedding."""
        lookup = getattr(self.store, "vectors_by_text", None)
        known: dict[str, list[float]] = {}
        if lookup is not None:
            try:
                known = lookup(doc_id)
            except Exception as exc:  # an old store file, a locked database: just embed everything
                logger.warning("Stored vectors of %s not reused: %s", doc_id, exc)
        missing = list(dict.fromkeys(text for text in texts if text_key(text) not in known))
        fresh: dict[str, list[float]] = {}
        requests = 0
        for start in range(0, len(missing), self.embed_batch):
            if pause and start:
                pause()
            batch = missing[start : start + self.embed_batch]
            fresh.update(zip((text_key(text) for text in batch), self._embed_patiently(batch, pause), strict=True))
            requests += 1
        if report is not None:
            report.update(embedded=len(missing), reused=len(texts) - len(missing), embedding_requests=requests)
        return [fresh.get(text_key(text)) or known[text_key(text)] for text in texts]

    def _embed_patiently(self, batch: list[str], pause: Callable[[], None] | None) -> list[list[float]]:
        """One batch, tried again here when the service stays busy (429, 5xx, a dropped
        connection) after the client's own retries: the file keeps the batches it already has,
        instead of failing and being read and embedded again from the start."""
        for attempt in range(len(self.embed_patience) + 1):
            try:
                return self.embeddings.embed_documents(batch)
            except Exception as exc:
                if attempt >= len(self.embed_patience) or not _worth_waiting(exc):
                    raise
                delay = self.embed_patience[attempt]
                logger.info("Embedding service busy (%s), batch tried again in %.0f s", type(exc).__name__, delay)
                deadline = time.monotonic() + delay
                while (left := deadline - time.monotonic()) > 0:
                    if pause:
                        pause()  # also stops here when the file was switched to name-only
                    time.sleep(min(left, 1.0))
        raise AssertionError("unreachable")

    def ingest(self, path: str | Path, replace: bool = False) -> IngestionResult:
        """Legacy content-hash ingestion, kept for the benchmark scripts.

        The app indexes through the file registry (``index_document``) instead.
        """
        with self._lock:
            document = self.parser.parse(path)
            exists = self.store.has_document(document.doc_id)
            parents, drafts = self.chunker.split_parent_child(document)
            if exists and not replace:
                if not self.parent_store.has_document(document.doc_id):
                    self.parent_store.save_document(document.doc_id, parents)
                return IngestionResult(document.doc_id, document.source, 0, "skipped")

            if not drafts:
                raise ValueError(f"{document.source} 未生成有效文本块")
            vectors = self.embeddings.embed_documents([draft.text for draft in drafts])
            records = [
                ChunkRecord(
                    id=draft.id,
                    doc_id=draft.doc_id,
                    parent_id=draft.parent_id,
                    source=draft.source,
                    heading=draft.heading,
                    locator=draft.locator,
                    text=draft.text,
                    dense=vector,
                )
                for draft, vector in zip(drafts, vectors, strict=True)
            ]

            # doc_id hashes name + content, so same-named files with different
            # content are distinct documents. Only an explicit replace treats
            # earlier documents with the same source as older versions.
            stale_ids = {document.doc_id} if exists else set()
            if replace:
                stale_ids.update(self.store.document_ids_for_source(document.source))
            for stale_id in stale_ids:
                self.store.delete_document(stale_id)
                if stale_id != document.doc_id:
                    self.parent_store.delete_document(stale_id)
                    (self.settings.parsed_dir / f"{stale_id}.json").unlink(missing_ok=True)
                    shutil.rmtree(self.settings.upload_dir / stale_id, ignore_errors=True)
            self.store.insert(records)
            self.parent_store.save_document(document.doc_id, parents)
            parsed_path = self.settings.parsed_dir / f"{document.doc_id}.json"
            parsed_path.write_text(
                json.dumps(document.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return IngestionResult(document.doc_id, document.source, len(records), "indexed")

    def backfill_parent_store(self, document_ids: set[str] | None = None) -> int:
        """Rebuild missing parent files from parser snapshots created by older versions."""
        rebuilt = 0
        for path in sorted(self.settings.parsed_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                doc_id = str(payload.get("doc_id", ""))
                if document_ids is not None and doc_id not in document_ids:
                    continue
                if self.parent_store.has_document(doc_id):
                    continue
                document = ParsedDocument(
                    doc_id=doc_id,
                    source=str(payload["source"]),
                    file_type=str(payload["file_type"]),
                    units=[DocumentUnit(**unit) for unit in payload.get("units", [])],
                )
                parents, _ = self.chunker.split_parent_child(document)
                if parents:
                    self.parent_store.save_document(doc_id, parents)
                    rebuilt += 1
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping invalid parser snapshot %s: %s", path, exc)
        return rebuilt
