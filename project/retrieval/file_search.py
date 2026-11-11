"""Find files, not passages: metadata filters, then content scores aggregated per file.

1. The file registry narrows the candidates by type, time and folder.
2. Chunks and file-level entries (name + outline) are searched inside the candidates
   and reranked together in one call, so their scores are comparable.
3. Each file scores ``max(best chunk + small bonus per extra hit, its entry) + name bonus``;
   files then come with their one or two best passages and page locators.
4. With ``doc_type_search``, a question naming a document type ("找一下我的简历") puts the
   files whose card has that type first (retrieval/doc_cards.py); files that only mention
   the word come after them. A question for nothing but the type ("有哪些合同") lists every
   file of it, newest first when asked ("最近的检测报告", "最新的论文").
5. Files the user is looking at get their card first (``cards``, the DocCardBuilder): those
   listed, and after a big import the likely ones of a type asked for that still have none
   (``untyped`` says how many candidates wait, so the answer can say the list may be short).

No generation and no fact check: this path is meant to answer in well under two seconds.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from itertools import zip_longest
import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from config import Settings, get_settings
from core import timings
from library.registry import SEARCHABLE_STATUSES
from .doc_cards import matching_types, name_doc_type, subject_matches, type_probe, type_words
from .file_filters import RECENT_ALONE, FileFilters, name_match, parse_query
from .types import SearchHit

logger = logging.getLogger(__name__)

_CONTEXT_PREFIX = re.compile(r"^【[^】\n]{0,200}】\n")

CHUNK_POOL = 24
ENTRY_POOL = 12
HIT_BONUS = 0.05
# A passage containing an identifier from the question verbatim ("LOG-04-ZX11", "GPT-4o"):
# the reranker scores same-template passages almost alike and is blind to such codes.
CODE_BONUS = 0.2
_CODE = re.compile(r"(?<![0-9A-Za-z])(?=[0-9A-Za-z_-]*[A-Za-z])(?=[0-9A-Za-z_-]*\d)[0-9A-Za-z][0-9A-Za-z_-]{3,}")
NAME_WEIGHT = 0.3
# Files "find a file" can list: those with content, and those kept by name only ("named"),
# which are matched on this machine by name, folder and date and never sent anywhere.
FINDABLE_STATUSES = (*SEARCHABLE_STATUSES, "named")
# Files scoring below this share of the best file are dropped from topic searches.
RELATIVE_CUTOFF = 0.3
# Files of the document type asked for are also searched on their own (their best passages
# and entries join the rerank), so "张三的简历" can tell the resumes apart.
TYPE_CHUNK_POOL = 12
TYPE_ENTRY_POOL = 8
# A type question while files still wait for their card: this many of them for the type word
# and as many for the type's description (their file entries) get theirs first.
TYPE_PROBE_FILES = 32


def _local_iso(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        if isinstance(value, (int, float)):
            moment = datetime.fromtimestamp(float(value))
        else:
            moment = datetime.fromisoformat(str(value)).astimezone().replace(tzinfo=None)
        return moment.isoformat(timespec="seconds")
    except (ValueError, OSError, OverflowError):
        return ""


def _snippet(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", _CONTEXT_PREFIX.sub("", text or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


class FileSearchService:
    def __init__(
        self,
        store,
        embeddings,
        reranker,
        registry=None,
        settings: Settings | None = None,
        clock: Callable[[], datetime] = datetime.now,
    ):
        self.store = store
        self.embeddings = embeddings
        self.reranker = reranker
        self.registry = registry
        self.settings = settings or get_settings()
        self.clock = clock
        self._vectors: OrderedDict[str, list[float]] = OrderedDict()
        self._vectors_lock = threading.Lock()
        self._names: list[str] = []
        self._names_at = -1e9
        # The document card builder (RAGSystem sets it): files looked at get their card first.
        self.cards = None
        # Type probes run in a thread (no wait in the search); False runs them inline (tests).
        self.probe_in_background = True

    # ---------------- public ----------------

    def search(
        self,
        query: str,
        filters: FileFilters | None = None,
        limit: int = 8,
        scope: set[str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        filters = self.as_searched(filters or parse_query(query, self.clock()))
        files = self._candidates(filters, scope)
        restricted = filters.has_metadata or scope is not None
        topic = filters.topic.strip()
        results: list[dict[str, Any]] = []
        unreadable = self._unreadable(filters, scope) if not topic else 0
        pool: list[str] = []
        if files:
            names = {doc_id: name_match(filters.name_keywords, row["name"]) for doc_id, row in files.items()}
            if topic:
                results = self._rank_by_content(topic, files, names, set(files) if restricted else None, filters, pool)
            else:
                results = self._rank_by_metadata(files, names, filters)
        total = len(results)
        typed = sum("type" in item["matched_by"] for item in results)
        results = results[: max(1, int(limit))]
        untyped = self._ask_for_cards(results, filters, files, pool)
        return {
            "query": query,
            "filters": filters.to_dict(),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            # How many files match, of which ``files`` holds at most ``limit``. Without a topic
            # this is the exact count of files meeting the conditions; with one, of those found.
            "total": total,
            # Of those, the files of the document type asked for (doc_type_search).
            "typed": typed,
            "total_candidates": len(files),
            # Files meeting the conditions that could not be read, so they are not in the total.
            "unreadable": unreadable,
            # A type question: candidate files still waiting for their document card.
            "untyped": untyped,
            "files": results,
        }

    def best_name_match(self, text: str) -> float:
        """How closely the words of ``text`` match some file name (0..1); 0 without a registry."""
        keywords = parse_query(text, self.clock()).name_keywords
        if not keywords or self.registry is None:
            return 0.0
        now = time.monotonic()
        if now - self._names_at > 30:
            self._names = [row["name"] for row in self.registry.query_files(statuses=FINDABLE_STATUSES)]
            self._names_at = now
        return max((name_match(keywords, name) for name in self._names), default=0.0)

    def lists_type(self, filters: FileFilters | None) -> bool:
        """A question for nothing but a document type ("有哪些合同", "列出全部简历", "我的简历呢"):
        with doc_type_search every file of the type is listed, so the list may run long."""
        return bool(filters is not None and filters.doc_type and not filters.type_rest and self._type_search_on())

    def as_searched(self, filters: FileFilters) -> FileFilters:
        """The filters as this search applies them. "最近的检测报告" lists the reports newest
        first (``_by_type``) instead of keeping the last seven days: the latest ones may be a
        month old. A window said in full ("最近三天", "上周") still filters."""
        if filters.newest_first and filters.time_range and RECENT_ALONE.fullmatch(filters.time_phrase or "") \
                and self.lists_type(filters):
            return replace(filters, time_range=None, time_phrase="")
        return filters

    # ---------------- candidates ----------------

    def _candidates(self, filters: FileFilters, scope: set[str] | None) -> dict[str, dict[str, Any]]:
        if self.registry is None:
            # Benchmarks / legacy stores without a registry: names only, no times or folders.
            documents = {item["doc_id"]: {"file_id": item["doc_id"], "name": item["source"], "path": ""}
                         for item in self.store.list_documents()}
            extensions = tuple(filters.extensions)
            return {
                doc_id: row for doc_id, row in documents.items()
                if (scope is None or doc_id in scope) and (not extensions or row["name"].lower().endswith(extensions))
            }
        rows = self._query_rows(filters, scope, (*FINDABLE_STATUSES, "indexing", "pending", "failed"))
        # A file being re-indexed (or whose re-index failed) keeps its previous chunks searchable;
        # one just added is found by its name, time and folder while the queue reads it, and so
        # is one kept by name only (it has no content to search).
        return {
            row["file_id"]: row for row in rows
            if row["status"] in FINDABLE_STATUSES or row["status"] in ("pending", "indexing")
            or (row.get("chunks") or 0) > 0
        }

    def _query_rows(self, filters: FileFilters, scope: set[str] | None, statuses: tuple[str, ...]) -> list[dict[str, Any]]:
        added = modified = None
        if filters.time_range:
            start, end = filters.time_range
            if filters.time_field in ("added", "any"):
                added = (self._utc_iso(start), self._utc_iso(end))
            if filters.time_field in ("modified", "any"):
                modified = (start.timestamp(), end.timestamp())
        return self.registry.query_files(
            statuses=statuses,
            extensions=filters.extensions,
            added=added,
            modified=modified,
            either_time=filters.time_field == "any",
            folder=filters.folder,
            doc_ids=scope,
        )

    def _unreadable(self, filters: FileFilters, scope: set[str] | None) -> int:
        """Files meeting the conditions that have nothing to search: too large, broken, unknown."""
        if self.registry is None:
            return 0
        rows = self._query_rows(filters, scope, ("failed", "unsupported"))
        return sum(1 for row in rows if not row.get("chunks"))

    @staticmethod
    def _utc_iso(moment: datetime) -> str:
        return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")

    # ---------------- ranking ----------------

    def _rank_by_content(
        self,
        topic: str,
        files: dict[str, dict[str, Any]],
        names: dict[str, float],
        doc_ids: set[str] | None,
        filters: FileFilters | None = None,
        pool: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """``pool``: filled with every file the search reached, best first."""
        typed = self._typed_files(filters, files)
        pure = bool(typed) and not filters.type_rest
        chunks: list[SearchHit] = []
        entries: list[dict[str, Any]] = []
        try:
            vector = self._embed(topic)
            chunks = self._search_chunks(topic, vector, doc_ids)
            if hasattr(self.store, "search_file_entries"):
                with timings.stage("search"):
                    entries = self.store.search_file_entries(topic, vector, limit=ENTRY_POOL, doc_ids=doc_ids)
            if typed and not pure:
                seen = {hit.id for hit in chunks}
                chunks += [hit for hit in self._search_chunks(topic, vector, set(typed), TYPE_CHUNK_POOL)
                           if hit.id not in seen]
                if hasattr(self.store, "search_file_entries"):
                    known = {entry["doc_id"] for entry in entries}
                    with timings.stage("search"):
                        scoped = self.store.search_file_entries(topic, vector, limit=TYPE_ENTRY_POOL, doc_ids=set(typed))
                    entries += [entry for entry in scoped if entry["doc_id"] not in known]
        except Exception as exc:
            logger.warning("File search: content retrieval failed, ranking by name only: %s", exc)
        chunks = [hit for hit in chunks if hit.doc_id in files]
        entries = [entry for entry in entries if entry["doc_id"] in files]
        chunk_scores, entry_scores = self._rerank(topic, chunks, entries)
        codes = [code.lower() for code in _CODE.findall(topic)]
        if codes:
            chunk_scores = [
                score + CODE_BONUS * any(code in hit.text.lower() for code in codes)
                for hit, score in zip(chunks, chunk_scores)
            ]

        best: dict[str, list[tuple[float, SearchHit]]] = {}
        for hit, score in zip(chunks, chunk_scores):
            best.setdefault(hit.doc_id, []).append((score, hit))
        summary = {entry["doc_id"]: (score, entry) for entry, score in zip(entries, entry_scores)}
        # Files of the type asked for that the search did not reach are shown with their entry.
        entry_texts = self._entry_texts([doc_id for doc_id in typed if doc_id not in best and doc_id not in summary])

        ranked = []
        for doc_id in set(best) | set(summary) | {d for d, s in names.items() if s > 0} | set(typed):
            hits = sorted(best.get(doc_id, []), key=lambda item: item[0], reverse=True)
            content = hits[0][0] + HIT_BONUS * math.log1p(len(hits)) if hits else 0.0
            entry_score = summary.get(doc_id, (0.0, None))[0]
            name = names.get(doc_id, 0.0)
            score = max(content, entry_score) + NAME_WEIGHT * name
            matched = [label for label, value in (("content", content), ("summary", entry_score), ("name", name)) if value > 0]
            snippets = [
                {"text": _snippet(hit.text), "locator": hit.locator, "heading": hit.heading, "score": round(s, 4)}
                for s, hit in hits[:2]
            ]
            if not snippets and doc_id in summary:
                snippets = [{"text": _snippet(summary[doc_id][1]["text"]), "locator": "", "heading": "", "score": None}]
            elif not snippets and entry_texts.get(doc_id):
                snippets = [{"text": _snippet(entry_texts[doc_id]), "locator": "", "heading": "", "score": None}]
            # [tier, score, doc_id, matched_by, snippets]; tier > 0 only for the type asked for.
            ranked.append([0, score, doc_id, matched, snippets])
        if pool is not None:
            pool.extend(item[2] for item in sorted(ranked, key=lambda item: item[1], reverse=True))
        if typed:
            ranked = self._by_type(ranked, typed, pure, filters, files)
        else:
            ranked.sort(key=lambda item: item[1], reverse=True)
            if ranked:
                floor = RELATIVE_CUTOFF * ranked[0][1]
                ranked = [item for item in ranked if item[1] >= floor]
        results = [self._describe(files[doc_id], score, matched, snippets) for _, score, doc_id, matched, snippets in ranked]
        if filters is not None and filters.doc_type and self._type_search_on():
            texts: dict[str, list[str]] = {}
            for doc_id, hits in best.items():
                texts.setdefault(doc_id, []).extend(hit.text for _, hit in hits)
            for doc_id, (_, entry) in summary.items():
                texts.setdefault(doc_id, []).append(str(entry.get("text", "")))
            self._label_types(results, typed, filters.doc_type, texts)
        return results

    # ---------------- document types ----------------

    def _ask_for_cards(
        self,
        listed: list[dict[str, Any]],
        filters: FileFilters | None,
        files: dict[str, dict[str, Any]],
        pool: list[str],
    ) -> int:
        """The files listed get their document card first; with a type question, so do the
        files it reached and those most like the type (``_probe``) that still have none.
        Returns how many candidates still wait for their card (0 unless a type question)."""
        cards = self.cards
        if cards is None or not getattr(self.settings, "doc_cards", False):
            return 0
        try:
            cards.prioritize([item["file_id"] for item in listed])
            if filters is None or not filters.doc_type or not self._type_search_on():
                return 0
            have = self.store.doc_card_hashes()
            waiting = {
                doc_id for doc_id, row in files.items()
                if doc_id not in have and row.get("status") in SEARCHABLE_STATUSES
                and row.get("content_mode") != "name" and not name_doc_type(str(row.get("name", "")))
            }
            if waiting:
                cards.prioritize([doc_id for doc_id in pool if doc_id in waiting])
                if self.probe_in_background:
                    threading.Thread(target=self._probe, args=(filters.doc_type, waiting),
                                     name="type-probe", daemon=True).start()
                else:
                    self._probe(filters.doc_type, waiting)
            return len(waiting)
        except Exception as exc:
            logger.warning("File search: document cards not queued: %s", exc)
            return 0

    def _probe(self, doc_type: str, waiting: set[str]) -> list[str]:
        """The files without a card most like ``doc_type``, queued for their card: their file
        entries searched for the type word and for the type's description, taken in turn (the
        word finds resumes and contracts, the description paper abstracts; measured in
        evaluation/doc_cards_after_import). Returns them, most alike first."""
        if not hasattr(self.store, "search_file_entries") or self.cards is None:
            return []
        rankings = []
        for text in dict.fromkeys((doc_type, type_probe(doc_type))):
            try:
                entries = self.store.search_file_entries(text, self._embed(text), limit=TYPE_PROBE_FILES,
                                                         doc_ids=waiting)
            except Exception as exc:
                logger.warning("File search: type probe failed: %s", exc)
                continue
            rankings.append([entry["doc_id"] for entry in entries])
        found = list(dict.fromkeys(doc_id for row in zip_longest(*rankings) for doc_id in row if doc_id))
        if found:
            self.cards.prioritize(found)
        return found

    def _type_search_on(self) -> bool:
        return bool(getattr(self.settings, "doc_type_search", False)) and hasattr(self.store, "doc_cards")

    def _typed_files(self, filters: FileFilters | None, files: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Cards of the candidate files whose type is the one asked for: by their card, or by
        their own name ("01 申报书.pdf" is one whatever its card says)."""
        if filters is None or not filters.doc_type or not files or not self._type_search_on():
            return {}
        types = matching_types(filters.doc_type)
        try:
            typed = self.store.doc_cards(doc_ids=files.keys(), types=types)
        except Exception as exc:
            logger.warning("File search: document cards unavailable: %s", exc)
            typed = {}
        for doc_id, row in files.items():
            by_name = name_doc_type(str(row.get("name", "")))
            if not by_name:
                continue
            if by_name not in types:
                # The name says another type ("…公司简介.pdf" whose card says 简历): the name wins.
                typed.pop(doc_id, None)
            elif doc_id not in typed:
                typed[doc_id] = {"doc_type": by_name, "label": by_name, "subject": "", "summary": "", "by_name": True}
        return typed

    def _entry_texts(self, doc_ids: list[str]) -> dict[str, str]:
        if not doc_ids or not hasattr(self.store, "file_entry_texts"):
            return {}
        try:
            return self.store.file_entry_texts(doc_ids)
        except Exception as exc:
            logger.warning("File search: file entries unavailable: %s", exc)
            return {}

    def _by_type(
        self,
        ranked: list[list[Any]],
        typed: dict[str, dict[str, Any]],
        pure: bool,
        filters: FileFilters,
        files: dict[str, dict[str, Any]],
    ) -> list[list[Any]]:
        """Files of the type asked for first (with 我的, those about the user first), then the
        others. Nothing else changes: the others keep the usual cutoff against the best file,
        and with more than the type in the question ("张三的简历") a file of the type must pass
        it too; with nothing but the type, every file of that type is listed, and with 最近 /
        最新 ("最新的论文") newest first: modified, or added for "最近加的…"."""
        top = max((item[1] for item in ranked), default=0.0)
        floor = RELATIVE_CUTOFF * top
        owner = str(getattr(self.settings, "owner_name", "") or "")
        mine = filters.mine and bool(getattr(self.settings, "doc_card_owner", False)) and bool(owner)
        for item in ranked:
            doc_id, score = item[2], item[1]
            if doc_id in typed and (pure or score >= floor):
                item[0] = 2 if mine and subject_matches(typed[doc_id].get("subject", ""), owner) else 1
                item[3] = ["type"] + item[3] + (["owner"] if item[0] == 2 else [])
                item[1] = score + item[0]

        def newest(doc_id: str) -> float:
            return float(files[doc_id].get("mtime") or 0.0)

        def dated(doc_id: str) -> float:
            if filters.time_field != "added":
                return newest(doc_id)
            try:
                return datetime.fromisoformat(str(files[doc_id].get("created_at"))).timestamp()
            except ValueError:
                return 0.0

        by_date = pure and filters.newest_first

        def order(item: list[Any]) -> tuple:
            if by_date and item[0] > 0:
                return item[0], dated(item[2]), item[1]
            return item[0], item[1], newest(item[2])

        ranked.sort(key=order, reverse=True)
        return [item for item in ranked if item[0] > 0 or item[1] >= floor]

    def _label_types(
        self,
        results: list[dict[str, Any]],
        typed: dict[str, dict[str, Any]],
        doc_type: str,
        texts: dict[str, list[str]] | None = None,
    ) -> None:
        """Each listed file's own card type, and "mention" on a file not of the type asked for
        whose name or matched passages (``texts``: whole passages and entries, not only the
        snippet shown) use the type's words."""
        try:
            cards = {**self.store.doc_cards(doc_ids=[item["file_id"] for item in results]), **typed}
        except Exception:
            cards = dict(typed)
        words = [word.lower() for word in type_words(doc_type)]
        for item in results:
            card = cards.get(item["file_id"])
            if card:
                item["doc_type"] = card.get("doc_type", "")
                item["type_label"] = card.get("label", "") or card.get("doc_type", "")
            if "type" in item["matched_by"]:
                continue
            parts = [item["name"], *(snippet.get("text", "") for snippet in item.get("snippets") or []),
                     *(texts or {}).get(item["file_id"], [])]
            text = " ".join(parts).lower()
            if any(word in text for word in words):
                item["matched_by"] = item["matched_by"] + ["mention"]

    def _rank_by_metadata(
        self, files: dict[str, dict[str, Any]], names: dict[str, float], filters: FileFilters
    ) -> list[dict[str, Any]]:
        """No topic ("上周加的 PDF"): name matches first, then the newest."""
        def newest(row: dict[str, Any]) -> float:
            if filters.time_field == "added":
                try:
                    return datetime.fromisoformat(str(row.get("created_at"))).timestamp()
                except ValueError:
                    return 0.0
            return float(row.get("mtime") or 0.0)

        order = sorted(files.values(), key=lambda row: (names.get(row["file_id"], 0.0), newest(row)), reverse=True)
        if filters.name_keywords and any(names.values()):
            order = [row for row in order if names.get(row["file_id"], 0.0) > 0]
        return [
            self._describe(row, NAME_WEIGHT * names.get(row["file_id"], 0.0),
                           ["metadata"] + (["name"] if names.get(row["file_id"]) else []), [])
            for row in order
        ]

    def _search_chunks(
        self, topic: str, vector: list[float], doc_ids: set[str] | None, limit: int = CHUNK_POOL
    ) -> list[SearchHit]:
        with timings.stage("search"):
            if doc_ids is None:
                return self.store.search_hybrid(topic, vector, limit=limit)
            return self.store.search_hybrid(topic, vector, limit=limit, doc_ids=doc_ids)

    def _rerank(self, topic: str, chunks: list[SearchHit], entries: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
        texts = [hit.text for hit in chunks] + [entry["text"] for entry in entries]
        if not texts:
            return [], []
        scores = [0.0] * len(texts)
        try:
            for item in self.reranker.rerank(topic, texts, top_n=len(texts)):
                scores[item.index] = float(item.score)
        except Exception as exc:
            # RRF scores, scaled so the best is 1: still a usable order, not comparable to rerank.
            logger.warning("File search: reranker failed, using RRF order: %s", exc)
            raw = [hit.score for hit in chunks] + [entry["score"] for entry in entries]
            top = max(raw) or 1.0
            scores = [value / top for value in raw]
        return scores[: len(chunks)], scores[len(chunks) :]

    def _embed(self, text: str) -> list[float]:
        with self._vectors_lock:
            if text in self._vectors:
                self._vectors.move_to_end(text)
                return self._vectors[text]
        vector = self.embeddings.embed_query(text)
        with self._vectors_lock:
            self._vectors[text] = vector
            while len(self._vectors) > 256:
                self._vectors.popitem(last=False)
        return vector

    @staticmethod
    def _describe(row: dict[str, Any], score: float, matched: list[str], snippets: list[dict[str, Any]]) -> dict[str, Any]:
        name = str(row.get("name", ""))
        return {
            "file_id": row.get("file_id", ""),
            "name": name,
            "path": row.get("path", ""),
            "ext": Path(name).suffix.lower(),
            "size": row.get("size"),
            "mtime": _local_iso(row.get("mtime")),
            "added_at": _local_iso(row.get("created_at")),
            "status": row.get("status", "indexed"),
            "score": round(float(score), 4),
            "matched_by": matched,
            "snippets": snippets,
        }
