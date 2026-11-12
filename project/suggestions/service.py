"""Home-page suggestions: build the question pool, keep it in step with the library, serve cards.

The pool is built per file in a background thread:

1. local templates write candidates from the file name and outline (no model call);
2. optionally, when the user turned it on, one model call per file writes 3–5 more from
   the file name, outline and the first ~240 characters;
3. every candidate is asked against the index: a question is kept only when its first
   hit is the file it was written from (for "find a file" questions, the first file).

The pool follows the library: a re-indexed or renamed file is rebuilt, a removed file
loses its questions, and files matching a sensitive word or an excluded folder get none.
"""

from __future__ import annotations

from collections import deque
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Iterable

from config import Settings
from retrieval.file_index import build_entry_text
from services.model_clients import chat_busy

from . import privacy
from .sampler import Sampler
from .store import HISTORY_DAYS, HISTORY_LIMIT, SuggestionStore
from .templates import ASK, FIND, Candidate, candidates as template_candidates, title_key

logger = logging.getLogger(__name__)

# Bump when templates change: every file's template questions are rebuilt.
POOL_VERSION = 1
MAX_PER_FILE = 5
MAX_FIND_PER_FILE = 1
MAX_CANDIDATES = 9
SYNC_INTERVAL_SECONDS = 60
RECENT_FOR_HISTORY = 20

AI_PROMPT = """下面是用户知识库里一个文件的名称、目录和开头。请写 3 到 5 个用户可能会针对这个文件提出的具体问题。
要求：
- 每个问题都能从这个文件里找到答案，问题里带上能定位到这个文件的关键词；
- 用中文，口语自然，每个不超过 30 个字；
- 每行一个问题，不要编号，不要解释。

{entry}"""


class SuggestionService:
    def __init__(
        self,
        settings: Settings,
        registry,
        store: SuggestionStore,
        retriever=None,
        file_search=None,
        chat_model: Callable[[], Any] | None = None,
        sampler: Sampler | None = None,
    ):
        self.settings = settings
        self.registry = registry
        self.store = store
        self.retriever = retriever
        self.file_search = file_search
        self.chat_model = chat_model
        self.sampler = sampler or Sampler()
        self._queue: deque[str] = deque()
        self._queued: set[str] = set()
        self._wake = threading.Condition()
        self._sync_lock = threading.Lock()
        self._last_sync = 0.0
        self._worker: threading.Thread | None = None
        self._stopped = False
        self._busy = False
        # Waits while a question is being answered (LibraryService.pause_for_questions): the
        # builder's embedding, rerank and model calls share the account's rate limits with
        # the answer, and right after a start or an upgrade it may have many files to go.
        self.pause: Callable[[], None] | None = None

    def _yield_to_questions(self) -> None:
        # The chat model was just "too busy" for a question: wait before adding to its load.
        model = str(getattr(getattr(self, "settings", None), "llm_model", "") or "")
        if model:
            chat_busy.wait(model)
        if self.pause is not None:
            try:
                self.pause()
            except Exception as exc:  # never let the courtesy wait stop the builder
                logger.debug("Suggestion builder pause failed: %s", exc)

    # ---------------- preferences ----------------

    def prefs(self) -> privacy.SuggestionPrefs:
        return privacy.load_prefs(self.settings)

    # ---------------- background work ----------------

    def start(self) -> None:
        """Sync with the library and start the builder thread (once)."""
        self.store.prune_history()
        if self._worker is None:
            self._worker = threading.Thread(target=self._worker_loop, name="suggestions", daemon=True)
            self._worker.start()
        self.sync()

    def stop(self) -> None:
        with self._wake:
            self._stopped = True
            self._wake.notify_all()

    def enqueue(self, file_id: str) -> None:
        with self._wake:
            if file_id not in self._queued:
                self._queued.add(file_id)
                self._queue.append(file_id)
                self._wake.notify_all()

    @property
    def building(self) -> bool:
        with self._wake:
            return bool(self._queue) or self._busy

    def _worker_loop(self) -> None:
        while True:
            with self._wake:
                while not self._queue and not self._stopped:
                    self._wake.wait(timeout=30)
                if self._stopped:
                    return
                file_id = self._queue.popleft()
                self._queued.discard(file_id)
                self._busy = True
            try:
                self._yield_to_questions()
                self.build_file(file_id)
            except Exception as exc:
                # Model or network trouble: the next sync tries this file again.
                logger.warning("Suggestions for %s not built: %s", file_id, exc)
            finally:
                with self._wake:
                    self._busy = False

    def run_pending(self) -> int:
        """Build every queued file now, in this thread (tests and evaluation)."""
        done = 0
        while True:
            with self._wake:
                if not self._queue:
                    return done
                file_id = self._queue.popleft()
                self._queued.discard(file_id)
            self.build_file(file_id)
            done += 1

    # ---------------- library events ----------------

    def file_indexed(self, file_id: str) -> None:
        self.enqueue(file_id)

    def file_removed(self, file_id: str) -> None:
        self.store.delete_file(file_id)

    def question_asked(self, question: str) -> None:
        """Remember a question the user sent, only while "同时参考我的提问" is on."""
        prefs = self.prefs()
        if prefs.uses_history:
            self.store.log_question(question)

    def sync(self) -> dict[str, int]:
        """Bring the pool in line with the registry and the current preferences."""
        with self._sync_lock:
            self._last_sync = time.monotonic()
            prefs = self.prefs()
            live = {row["file_id"]: row for row in self.registry.list_files(statuses=("indexed",))}
            states = self.store.file_states()
            pooled = {row["file_id"] for row in self.store.questions()}
            counts = {"queued": 0, "removed": 0, "excluded": 0}
            for file_id in (set(states) | pooled) - set(live):
                record = self.registry.get_file(file_id)
                # Missing/offline originals stay searchable; only deleted ones lose their questions.
                if record is None or record["status"] not in ("missing", "offline", "indexing", "pending"):
                    self.store.delete_file(file_id)
                    counts["removed"] += 1
            for file_id, record in live.items():
                state = states.get(file_id)
                reason = privacy.exclusion_reason(record, prefs)
                if reason:
                    if state is None or state["excluded"] != reason or file_id in pooled:
                        self._exclude(file_id, record, reason)
                        counts["excluded"] += 1
                    continue
                stale = (
                    state is None
                    or state["excluded"]
                    or state["sha256"] != record.get("indexed_sha256")
                    or int(state["version"]) != POOL_VERSION
                    or state["name"] != record["name"]
                    or (prefs.ai_questions and not state["ai_done"])
                )
                if stale:
                    self.enqueue(file_id)
                    counts["queued"] += 1
            return counts

    def sync_soon(self) -> None:
        if time.monotonic() - self._last_sync < SYNC_INTERVAL_SECONDS:
            return
        self._last_sync = time.monotonic()
        threading.Thread(target=self._safe_sync, name="suggestions-sync", daemon=True).start()

    def _safe_sync(self) -> None:
        try:
            self.sync()
        except Exception:
            logger.exception("Suggestion sync failed")

    def _exclude(self, file_id: str, record: dict[str, Any], reason: str) -> None:
        self.store.replace_questions(file_id, "template", [])
        self.store.replace_questions(file_id, "ai", [])
        self.store.set_file_state(
            file_id, sha256=record.get("indexed_sha256"), name=record["name"], version=POOL_VERSION,
            excluded=reason, ai_done=0,
        )

    # ---------------- building ----------------

    def _parsed(self, file_id: str) -> dict[str, Any] | None:
        path = Path(self.settings.parsed_dir) / f"{file_id}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _sources(self, record: dict[str, Any]) -> set[str]:
        """The file and its copies / other formats: a hit on either answers the question."""
        key = title_key(record["name"])
        same = {
            row["file_id"]
            for row in self.registry.list_files(statuses=("indexed",))
            if title_key(row["name"]) == key
        }
        return same | {record["file_id"]}

    def verify(self, question: str, qtype: str, sources: set[str], vector: list[float] | None = None) -> bool:
        if qtype == FIND:
            if self.file_search is None:
                return False
            files = (self.file_search.search(question, limit=3) or {}).get("files") or []
            return bool(files) and files[0].get("file_id") in sources
        if self.retriever is None:
            return False
        hits = self.retriever.retrieve(question, top_k=3, query_vector=vector)
        return bool(hits) and hits[0].doc_id in sources

    def _verified_rows(self, record: dict[str, Any], found: Iterable[Candidate], limit: int) -> list[dict[str, Any]]:
        found = list(found)[:MAX_CANDIDATES]
        sources = self._sources(record)
        asks = [c.question for c in found if c.qtype == ASK]
        vectors: dict[str, list[float]] = {}
        embeddings = getattr(self.retriever, "embeddings", None)
        if asks and embeddings is not None:
            try:
                vectors = dict(zip(asks, embeddings.embed_documents(asks), strict=True))
            except Exception as exc:
                logger.debug("Batch embedding for suggestions failed, one by one: %s", exc)
        rows: list[dict[str, Any]] = []
        kept = finds = 0
        for candidate in found:
            if kept >= limit:
                break
            if candidate.qtype == FIND and finds >= MAX_FIND_PER_FILE:
                continue
            self._yield_to_questions()
            ok = self.verify(candidate.question, candidate.qtype, sources, vectors.get(candidate.question))
            rows.append({
                "question": candidate.question, "qtype": candidate.qtype, "template": candidate.template,
                "verified": ok, "title_key": title_key(record["name"]), "file_name": record["name"],
            })
            if ok:
                kept += 1
                finds += candidate.qtype == FIND
        return rows

    def build_file(self, file_id: str) -> int:
        """(Re)build one file's questions. Returns how many passed verification."""
        record = self.registry.get_file(file_id)
        if record is None:
            self.store.delete_file(file_id)
            return 0
        if record["status"] != "indexed":
            return 0
        prefs = self.prefs()
        reason = privacy.exclusion_reason(record, prefs)
        if reason:
            self._exclude(file_id, record, reason)
            return 0
        parsed = self._parsed(file_id)
        if parsed is None:
            return 0
        state = self.store.file_states().get(file_id) or {}
        changed = (
            state.get("sha256") != record.get("indexed_sha256")
            or int(state.get("version") or 0) != POOL_VERSION
            or state.get("name") != record["name"]
            or bool(state.get("excluded"))
        )
        passed = 0
        if changed:
            rows = self._verified_rows(record, template_candidates(record["name"], parsed, file_id), MAX_PER_FILE)
            self.store.replace_questions(file_id, "template", rows)
            passed = sum(row["verified"] for row in rows)
            # The file changed: questions a model wrote for the old version are dropped.
            self.store.replace_questions(file_id, "ai", [])
            state["ai_done"] = 0
        ai_done = bool(state.get("ai_done")) and not changed
        if prefs.ai_questions and not ai_done and self.chat_model is not None:
            passed += self._build_ai(record, parsed)
            ai_done = True
        self.store.set_file_state(
            file_id, sha256=record.get("indexed_sha256"), name=record["name"], version=POOL_VERSION,
            excluded="", ai_done=int(ai_done),
        )
        return passed

    def _build_ai(self, record: dict[str, Any], parsed: dict[str, Any]) -> int:
        """One model call: file name, outline and opening go to the model service."""
        entry = build_entry_text(record["name"], parsed)
        self._yield_to_questions()
        reply = self.chat_model().invoke(AI_PROMPT.format(entry=entry))
        text = str(getattr(reply, "content", reply) or "")
        questions = []
        for line in text.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.、)）])\s*", "", line).strip()
            if 6 <= len(line) <= 60 and line not in questions:
                questions.append(line if line.endswith(("？", "?")) else line + "？")
        candidates = [Candidate(question, ASK, "ai") for question in questions[:5]]
        rows = self._verified_rows(record, candidates, 5)
        self.store.replace_questions(record["file_id"], "ai", rows)
        return sum(row["verified"] for row in rows)

    # ---------------- serving ----------------

    def suggest(self, n: int = 4, exclude: Iterable[str] = ()) -> dict[str, Any]:
        prefs = self.prefs()
        live = {row["file_id"]: row for row in self.registry.list_files(statuses=("indexed",))}
        payload: dict[str, Any] = {
            "enabled": prefs.enabled,
            "personalize": prefs.personalize,
            "ai_questions": prefs.ai_questions,
            "file_count": len(live),
            "library_empty": not live,
            "building": self.building,
            "cards": [],
        }
        if not prefs.enabled or not live:
            return payload
        self.sync_soon()
        candidates = []
        for row in self.store.candidates():
            record = live.get(row["file_id"])
            if record is None or (row["method"] == "ai" and not prefs.ai_questions):
                continue
            # Preferences apply at once, before the next sync removes the questions.
            if privacy.exclusion_reason(record, prefs):
                continue
            candidates.append({**row, "added_at": record["created_at"], "path": record["path"],
                               "file_name": record["name"]})
        recent = self.store.recent_questions(RECENT_FOR_HISTORY) if prefs.uses_history else []
        cold = not recent and not self.store.has_feedback()
        cards = self.sampler.pick(candidates, recent, n=n, exclude=exclude, cold_start=cold)
        payload.update(pool_size=len(candidates), cold_start=cold, cards=[card.to_dict() for card in cards])
        return payload

    def record(self, action: str, ids: Iterable[str]) -> int:
        return self.store.record(action, ids)

    def block_file(self, file_id: str) -> dict[str, Any]:
        record = self.registry.get_file(file_id)
        if record is None:
            raise KeyError(file_id)
        self.store.block_file(file_id, record["name"])
        return {"doc_id": file_id, "name": record["name"]}

    def overview(self) -> dict[str, Any]:
        return overview(self.settings, self.store, building=self.building)


def overview(settings: Settings, store: SuggestionStore | None, building: bool = False) -> dict[str, Any]:
    """For the settings page: preferences, blocked files, how much history is kept."""
    if store is None:
        return {**privacy.load_prefs(settings).to_dict(), "blocked_files": [], "history_count": 0,
                "history_limit": {"count": HISTORY_LIMIT, "days": HISTORY_DAYS}, "pool": {}, "building": False}
    return {
        **privacy.load_prefs(settings).to_dict(),
        "blocked_files": [{"doc_id": row["file_id"], "name": row["name"]} for row in store.blocked_files()],
        "history_count": store.history_count(),
        "history_limit": {"count": HISTORY_LIMIT, "days": HISTORY_DAYS},
        "pool": store.stats(),
        "building": building,
    }

