"""Application service container used by FastAPI and evaluation scripts."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import shutil
import threading
import time
import uuid
from typing import Any, Iterator

from langchain_core.messages import HumanMessage, RemoveMessage

from config import Settings, get_settings
from core import timings, user_settings
from core.conversations import ConversationCheckpointer, ConversationStore, state_from_text
from core.data_safety import DataSafety
from core.scrubber import Scrubber
from db.parent_store_manager import ParentStoreManager
from ingestion.parsers import stable_doc_id
from ingestion.pipeline import IngestionPipeline
from library import FileRegistry, LibraryService
from rag_agent.document_nodes import DocumentAccess
from rag_agent.graph import create_agent_graph
from rag_agent.whole_doc import NoteStore
from library.preview import readable_original
from library.registry import SEARCHABLE_STATUSES
from retrieval.doc_cards import DocCardBuilder
from retrieval.file_filters import FileFilters, parse_query
from retrieval.file_index import FileIndexer
from retrieval.file_search import FileSearchService
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.factory import create_store
from retrieval.types import ChunkStore
from services.model_clients import (
    MANUAL_RETRY_KINDS,
    SiliconFlowEmbeddings,
    SiliconFlowReranker,
    build_chat_model,
    open_connections,
    within_budget,
)
from suggestions import SuggestionService, SuggestionStore
from suggestions.followups import SCENE_NONE, FollowupService, Result as FollowupResult


logger = logging.getLogger(__name__)

# A turn is stored once, when it ends: one that never finished is not part of what a
# later question refers to, and the steps in between are not worth a write each.
TURN_DURABILITY = "exit"


def _is_doc_id(doc_id: str) -> bool:
    return len(doc_id) in (24, 32) and all(char in "0123456789abcdef" for char in doc_id)


def _name_and_mtime(record: dict | None) -> tuple[str, float, str] | None:
    """A registry row as the retriever's version check reads it (name, modification time, path)."""
    return (str(record["name"]), float(record.get("mtime") or 0.0), str(record.get("path") or "")) if record else None


class RAGSystem:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.safety = DataSafety(self.settings)
        self._lock = threading.Lock()
        self.initialized = False
        self.store: ChunkStore | None = None
        self.retriever: HybridRetriever | None = None
        self.ingestion: IngestionPipeline | None = None
        self.parent_store: ParentStoreManager | None = None
        self.library: LibraryService | None = None
        self.scrubber: Scrubber | None = None
        self.file_indexer: FileIndexer | None = None
        self.file_search: FileSearchService | None = None
        self.doc_cards: DocCardBuilder | None = None
        self.suggestions: SuggestionService | None = None
        self.followup_service: FollowupService | None = None
        self._suggestion_store: SuggestionStore | None = None
        self._conversations: ConversationStore | None = None
        self._checkpointer: ConversationCheckpointer | None = None
        self._store_lock = threading.Lock()
        self.agent_graph = None
        self._library_started = False
        # Follow-up questions written while an answer is being checked (see _stream):
        # thread_id -> ((question, answer), future).
        self._followup_jobs: OrderedDict[str, tuple[tuple[str, str], Future]] = OrderedDict()
        self._followup_lock = threading.Lock()
        self._followup_pool: ThreadPoolExecutor | None = None
        self._warm_lock = threading.Lock()
        self._warmed_at = float("-inf")

    def initialize(self) -> "RAGSystem":
        if self.initialized:
            return self
        with self._lock:
            if self.initialized:
                return self
            # Without a key nothing could be indexed or answered; the library would only
            # mark every file failed. Raises MissingApiKeyError with the user-facing reason.
            self.settings.require_api_key()
            # A damaged index is not opened: raises DatabaseDamaged, the UI offers the way back.
            self.safety.open_for_use()
            embeddings = SiliconFlowEmbeddings(self.settings)
            reranker = SiliconFlowReranker(self.settings)
            self.store = create_store(self.settings)
            self.store.ensure_collection()
            self.parent_store = ParentStoreManager(self.settings)
            self.retriever = HybridRetriever(self.store, embeddings, reranker, self.settings)
            self.ingestion = IngestionPipeline(
                self.store,
                embeddings,
                self.settings,
                parent_store=self.parent_store,
            )
            document_ids = {item["doc_id"] for item in self.store.list_documents()}
            self.ingestion.backfill_parent_store(document_ids)
            self.file_indexer = FileIndexer(self.store, embeddings, self.settings)
            self.library = LibraryService(
                self.settings,
                FileRegistry(self.settings.registry_path),
                self.store,
                self.parent_store,
                indexer=self._index_file,
            )
            registry = self.library.registry
            self.retriever.describe_document = lambda doc_id: _name_and_mtime(registry.get_file(doc_id))
            # Files in the Recycle Bin keep their index for a quick restore, but are not cited.
            self.retriever.hidden_documents = self.library.hidden_ids
            self.file_search = FileSearchService(
                self.store, embeddings, reranker, self.library.registry, self.settings
            )
            self.suggestions = SuggestionService(
                self.settings,
                self.library.registry,
                self.suggestion_store,
                retriever=self.retriever,
                file_search=self.file_search,
                chat_model=lambda: build_chat_model(self.settings, max_tokens=400),
            )
            # What kind of document each file is (settings.doc_cards): one model call per file,
            # in its own paced thread, waiting while a question is answered.
            self.doc_cards = DocCardBuilder(
                self.store,
                self.settings,
                chat_model=self._card_model,
                files=self._card_files,
                eligible=self._card_eligible,
                pause=self.library.pause_for_questions,
            )
            self.library.on_indexed = self._file_indexed
            # Home-page questions are built in the background; they wait while a question is
            # answered, as indexing does (they use the same account and rate limits).
            self.suggestions.pause = self.library.pause_for_questions
            self.library.on_dropped = self._file_removed
            # Text dropped for good (removed, excluded, kept by name only) also leaves the
            # index file's free pages and the backups.
            self.scrubber = Scrubber(self.settings, self.safety, self.store, self.library.registry)
            self.library.on_purged = self.scrubber.add
            self.followup_service = FollowupService(
                self.settings,
                chat_model=lambda: build_chat_model(
                    self.settings,
                    max_tokens=self.settings.followup_max_tokens,
                    model=self.settings.followup_model or None,
                    timeout=self.settings.followup_timeout_seconds,
                    max_retries=0,
                ),
                pool=lambda n: self.suggestions.suggest(n=min(8, n))["cards"],
                describe_file=registry.get_file,
            )
            self.summary_notes = NoteStore(self.settings.data_dir / "summaries")
            self.agent_graph = self._build_graph()
            self.initialized = True
        return self

    def _build_graph(self):
        return create_agent_graph(
            llm=build_chat_model(self.settings),
            retriever=self.retriever,
            settings=self.settings,
            parent_store=self.parent_store,
            checkpointer=self.checkpointer,
            file_search=self.file_search,
            documents=self._document_access(),
            summary_llm=build_chat_model(self.settings, model=self.settings.summary_model) if self.settings.summary_model else None,
        )

    def _document_access(self) -> DocumentAccess:
        """What questions about a whole file read: the registry, parsed passages, the original."""
        registry = self.library.registry
        return DocumentAccess(
            rows=lambda: registry.list_files(statuses=SEARCHABLE_STATUSES),
            parents=self.parent_store.list_parents,
            metadata=self._parsed_metadata,
            original=lambda doc_id: self.original_file(doc_id, ""),
            notes=self.summary_notes,
        )

    def _parsed_metadata(self, doc_id: str) -> dict[str, Any]:
        try:
            parsed = json.loads((self.settings.parsed_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return parsed.get("metadata") or {}

    def reload_models(self) -> None:
        """Rebuild the chat graph after the chat model changed at runtime.

        Clients read the API key per request, so a saved or removed key needs no rebuild.
        Before the first successful initialize there is nothing to rebuild: the next
        request initializes with the new settings. The next turn uses the new model; a
        turn already running finishes on the graph it started with.
        """
        with self._lock:
            if not self.initialized:
                return
            self.agent_graph = self._build_graph()

    def start_library(self) -> LibraryService:
        """Migrate/reconcile the file registry and start indexing + folder watching (once)."""
        self.initialize()
        with self._lock:
            if not self._library_started:
                self._library_started = True
                # Purges left from the last run first: they must be done before anything new.
                self.scrubber.resume()
                # Before the folders are scanned: the first question may come at any moment.
                threading.Thread(target=self.warm_up, name="warm-up", daemon=True).start()
                self.library.start(watch=self.settings.watch_enabled)
                threading.Thread(target=self._start_background, name="file-entries", daemon=True).start()
        return self.library

    # Connections are reopened at most this often (typing in the question box asks for it).
    WARM_INTERVAL_SECONDS = 20.0

    def warm_up(self, connections_only: bool = False) -> dict[str, int]:
        """Pay a first question's one-off costs before it is asked: the word segmenter's
        dictionary, the index pages a search reads, and open connections to the model
        service. Returns milliseconds per step; failures are only logged."""
        with self._warm_lock:
            if connections_only and time.monotonic() - self._warmed_at < self.WARM_INTERVAL_SECONDS:
                return {}
            self._warmed_at = time.monotonic()
        steps: dict[str, int] = {}

        def step(name: str, action) -> None:
            started = time.perf_counter()
            try:
                action()
            except Exception as exc:
                logger.info("Warm-up step %s failed: %s", name, exc)
            steps[name] = round((time.perf_counter() - started) * 1000)

        if not connections_only:
            step("index", self._touch_index)
        step("connections", lambda: open_connections(self.settings))
        logger.info("Warm-up (ms): %s", steps)
        return steps

    def warm_up_soon(self) -> None:
        """Reopen connections in the background, e.g. while a question is being typed."""
        if self.initialized:
            threading.Thread(target=self.warm_up, kwargs={"connections_only": True}, name="warm-up", daemon=True).start()

    def _touch_index(self) -> None:
        """One hybrid search: loads jieba's dictionary and the vector pages into memory."""
        dimension = int(self.settings.embedding_dimension)
        self.store.search_hybrid("知识库", [1.0] + [0.0] * (dimension - 1), limit=1)

    def _file_indexed(self, doc_id: str) -> None:
        self.suggestions.file_indexed(doc_id)
        if self.settings.doc_cards and self.doc_cards is not None:
            self.doc_cards.file_indexed(doc_id)

    def _card_model(self):
        model = build_chat_model(self.settings, max_tokens=200, max_retries=0)
        # Background work: a slow first token is waited for, not sent twice.
        model.hedge_after = 0.0
        return model

    def _card_files(self) -> dict[str, str]:
        """Files that may have a document card: indexed, and not kept by name only."""
        return {
            row["file_id"]: row["name"]
            for row in self.library.registry.list_files(statuses=SEARCHABLE_STATUSES)
            if row.get("content_mode") != "name"
        }

    def _card_eligible(self, doc_id: str) -> str | None:
        record = self.library.registry.get_file(doc_id)
        if record is None or record.get("content_mode") == "name" or record.get("status") == "named":
            return None
        return str(record["name"])

    def start_doc_cards(self) -> int:
        """Queue the files whose card is missing or out of date and start the builder."""
        if not (self.settings.doc_cards and self.doc_cards is not None and self.doc_cards.supported):
            return 0
        try:
            queued = self.doc_cards.sync()
        except Exception as exc:
            logger.warning("Doc cards not synced, retried next start: %s", exc)
            return 0
        self.doc_cards.start()
        return queued

    def _file_removed(self, doc_id: str) -> None:
        self.suggestions.file_removed(doc_id)
        self.summary_notes.remove(doc_id)

    def _start_background(self) -> None:
        self.backfill_file_entries()
        self.start_doc_cards()
        try:
            # After the file entries: verifying "find a file" questions searches them.
            self.suggestions.start()
        except Exception as exc:
            logger.warning("Home-page suggestions did not start: %s", exc)

    @property
    def suggestion_store(self) -> SuggestionStore:
        """The suggestion tables; usable before initialize (settings page without an API key)."""
        with self._store_lock:
            if self._suggestion_store is None:
                self.safety.open_for_use()
                self._suggestion_store = SuggestionStore(self.settings.registry_path)
        return self._suggestion_store

    @property
    def conversations(self) -> ConversationStore:
        """The conversation history; usable before initialize (no API key yet) and on a damaged
        index. Only ``prepare``: a recovery waiting for this start runs before the file opens."""
        with self._store_lock:
            if self._conversations is None:
                self.safety.prepare()
                self._conversations = ConversationStore(self.settings.conversations_db_path)
        return self._conversations

    @property
    def checkpointer(self) -> ConversationCheckpointer:
        """The agent state of every conversation; survives restarts and model reloads."""
        with self._store_lock:
            if self._checkpointer is None:
                self.safety.prepare()
                self._checkpointer = ConversationCheckpointer(self.settings.conversations_db_path)
        return self._checkpointer

    def _index_file(
        self, path: str, doc_id: str, source: str | None = None, **options: Any
    ) -> tuple[int, int, dict[str, Any]]:
        """Index a file's chunks, then its file-level entry (name + outline).

        ``options`` come from the library: the file's ``image_mode`` and ``pause`` (waits
        while a question is answered). Returns (chunks, characters, report)."""
        report: dict[str, Any] = {}
        chunks, characters = self.ingestion.index_document(path, doc_id, source, report=report, **options)
        if pause := options.get("pause"):
            pause()  # stops here if the file was switched to name-only meanwhile
        try:
            self.file_indexer.index([(doc_id, source or Path(path).name)])
        except Exception as exc:
            # The chunks are in; a missing entry only weakens topic-level file search and
            # is rebuilt by the next backfill.
            logger.warning("File entry for %s not built: %s", doc_id, exc)
        if report.get("reused"):
            logger.info("Re-indexed %s: %d chunks embedded, %d reused", source or path, report["embedded"], report["reused"])
        return chunks, characters, report

    @contextmanager
    def _answering(self) -> Iterator[None]:
        """While a question is answered the indexing queue waits (see LibraryService.answering)."""
        if self.library is None:
            yield
            return
        with self.library.answering():
            yield

    def backfill_file_entries(self) -> int:
        """Build file-level entries for indexed files that lack a current one (older installs)."""
        if not self.file_indexer or not self.library:
            return 0
        names = {row["file_id"]: row["name"] for row in self.library.registry.list_files(statuses=SEARCHABLE_STATUSES)}
        try:
            return self.file_indexer.backfill(names)
        except Exception as exc:
            logger.warning("File entry backfill stopped, retried next start: %s", exc)
            return 0

    def search_files(
        self, query: str, limit: int = 8, filters: FileFilters | None = None, scope: list[str] | None = None
    ) -> dict[str, Any]:
        self.initialize()
        return self.file_search.search(query, filters or parse_query(query), limit=limit,
                                       scope=set(scope) if scope is not None else None)

    def resolve_scope(self, doc_ids: list[str] | None = None, folder: str = "") -> list[str]:
        """Document ids for "ask within these files / this folder"; empty means the whole library."""
        self.initialize()
        ids = [doc_id for doc_id in (doc_ids or []) if _is_doc_id(doc_id)]
        if folder:
            ids += [
                row["file_id"] for row in self.library.registry.files_under(folder)
                if row["status"] in SEARCHABLE_STATUSES
            ]
            if not ids:
                raise ValueError("该文件夹下没有已入库的文件")
        return sorted(set(ids))

    def annotate_citations(self, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach each cited file's registry status, so the UI can flag lost originals."""
        if not self.library or not citations:
            return citations
        info = self.library.describe_files(c.get("doc_id", "") for c in citations if c.get("doc_id"))
        annotated = []
        for citation in citations:
            extra = info.get(citation.get("doc_id", ""))
            annotated.append({**citation, "status": extra["status"], "path": extra["path"]} if extra else citation)
        return annotated

    @staticmethod
    def new_thread_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _graph_config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}

    def _begin_turn(self, thread_id: str) -> None:
        self.checkpointer.begin_turn(thread_id)
        self._restore_thread(thread_id)

    def _restore_thread(self, thread_id: str) -> None:
        """A conversation saved before the state was kept has only its text; that is enough
        to know what a follow-up refers to."""
        config = self._graph_config(thread_id)
        if self.checkpointer.get_tuple(config) is not None:
            return
        saved = self.conversations.get(thread_id)
        values = state_from_text(saved) if saved else None
        if values:
            self.agent_graph.update_state(config, values, as_node="finish")

    def chat(
        self, question: str, thread_id: str, scope: list[str] | None = None, received_at: float | None = None
    ) -> dict[str, Any]:
        turn = timings.TurnTimings(started=received_at)
        with timings.activate(turn):
            with timings.stage("prepare"):
                self.initialize()
                self._begin_turn(thread_id)
            with self._answering():
                result = self.agent_graph.invoke(
                    self._turn_input(question, scope),
                    config=self._graph_config(thread_id),
                    durability=TURN_DURABILITY,
                )
        record = self._save_timings(turn, result)
        return {
            "thread_id": thread_id,
            "answer": result.get("answer", ""),
            "intent": result.get("intent", ""),
            "rewritten_query": result.get("rewritten_query", ""),
            "citations": self.annotate_citations(result.get("citations", [])),
            "grounded": bool(result.get("grounded", False)),
            "refused": bool(result.get("refused", False)),
            "retrieval_attempts": int(result.get("retrieval_attempts", 0)),
            "fact_check_skipped": bool(result.get("check_skipped", False)),
            "check_mode": str(result.get("check_mode") or ""),
            "files": list(result.get("files") or []),
            "file_search": dict(result.get("file_search") or {}),
            "coverage": dict(result.get("coverage") or {}),
            "timings": record,
        }

    def _save_timings(self, turn: timings.TurnTimings, values: dict[str, Any]) -> dict[str, Any]:
        """Close the turn's timings, keep them for diagnostics and return them."""
        record = turn.finish(
            intent=values.get("intent") or "",
            profile=values.get("answer_profile") or "",
            check_mode=values.get("check_mode") or "",
            grounded=bool(values.get("grounded")),
            refused=bool(values.get("refused")),
            retrieval_attempts=int(values.get("retrieval_attempts") or 0),
            model=getattr(self.settings, "llm_model", ""),
            answer_mode=getattr(self.settings, "answer_mode", ""),
            answer_chars=len(str(values.get("answer") or "")),
        )
        if getattr(self.settings, "log_dir", None):
            timings.save(record, self.settings.log_dir)
        return record

    @staticmethod
    def _turn_input(question: str, scope: list[str] | None) -> dict[str, Any]:
        # The scope is always written, so a scoped turn never leaks into the next one.
        return {"messages": [HumanMessage(content=question)], "question": question, "scope_doc_ids": list(scope or [])}

    def stream(
        self, question: str, thread_id: str, scope: list[str] | None = None, received_at: float | None = None,
        retry: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """``received_at`` (``time.perf_counter()``) is when the request arrived: the timings
        then include any wait for start-up. ``retry``: the user asked the failed question again
        ("重新提问"); the failed turn leaves the history first."""
        turn = timings.TurnTimings(started=received_at)
        budget = float(getattr(self.settings, "turn_retry_budget_seconds", 0.0) or 0.0)
        yield from timings.iterate(within_budget(self._timed_stream(question, thread_id, scope, turn, retry), budget), turn)

    def _timed_stream(
        self, question: str, thread_id: str, scope: list[str] | None, turn: timings.TurnTimings, retry: bool = False
    ) -> Iterator[dict[str, Any]]:
        with timings.stage("prepare"):
            self.initialize()
        with self._answering():
            yield from self._stream(question, thread_id, scope, turn, retry)

    def _drop_failed_turn(self, config: dict[str, Any], question: str) -> int:
        """"重新提问": the last turn, if it asked this question and failed (the model or the
        search gave up), leaves the history, so the question is not in it twice and a
        follow-up does not refer to the failure. Returns how many messages went."""
        values = self.agent_graph.get_state(config).values or {}
        if not (values.get("error_kind") or values.get("model_error") or values.get("last_error")):
            return 0  # the last saved turn did not fail (a crash saves nothing)
        messages = list(values.get("messages") or [])
        last = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), None)
        if last is None or str(messages[last].content).strip() != question.strip():
            return 0
        ids = [m.id for m in messages[last:] if getattr(m, "id", None)]
        if ids:
            self.agent_graph.update_state(config, {"messages": [RemoveMessage(id=i) for i in ids],
                                                   "error_kind": "", "model_error": "", "last_error": ""},
                                          as_node="finish")
        return len(ids)

    def _stream(
        self, question: str, thread_id: str, scope: list[str] | None, turn: timings.TurnTimings | None = None,
        retry: bool = False,
    ) -> Iterator[dict[str, Any]]:
        turn = turn or timings.TurnTimings()
        config = self._graph_config(thread_id)
        with timings.stage("prepare"):
            self._begin_turn(thread_id)
            if retry:
                dropped = self._drop_failed_turn(config, question)
                if dropped:
                    logger.info("Asked again: the failed turn left the history (%d messages)", dropped)
        yield {"event": "start", "thread_id": thread_id}
        # Draft tokens from the answer node are forwarded as they are generated;
        # the fact check still runs afterwards and `final` stays authoritative.
        stream_draft = bool(getattr(self.settings, "stream_draft", False))
        drafted = False
        # The turn's state, kept from each node's update: once the answer is written, its
        # follow-up questions are written while the fact check runs, instead of after it.
        running = self._turn_values(config, question, scope) if self._followups_wanted() else None
        for mode, payload in self._graph_stream(question, config, stream_draft, scope):
            if mode == "custom":
                yield {"event": "progress", **payload}
                continue
            if mode == "messages":
                chunk, metadata = payload
                content = getattr(chunk, "content", "")
                if metadata.get("langgraph_node") == "generate_answer" and isinstance(content, str) and content:
                    drafted = True
                    yield {"event": "draft", "content": content}
                continue
            for node_name, values in payload.items():
                values = values or {}
                if running is not None:
                    running.update({key: value for key, value in values.items() if key != "messages"})
                    if node_name == "generate_answer" and values.get("answer") and not values.get("model_error"):
                        self._start_followups(thread_id, dict(running))
                event: dict[str, Any] = {"event": "node", "node": node_name}
                if node_name == "analyze_query":
                    event.update(
                        intent=values.get("intent"),
                        rewritten_query=values.get("rewritten_query"),
                    )
                elif node_name == "retrieve":
                    event.update(
                        retrieved=len(values.get("documents", [])),
                        retrieval_attempts=values.get("retrieval_attempts", 0),
                    )
                elif node_name == "fact_check":
                    event.update(
                        grounded=values.get("grounded", False),
                        check_skipped=bool(values.get("check_skipped")),
                        check_mode=values.get("check_mode", ""),
                    )
                elif node_name == "search_files":
                    event.update(found=(values.get("file_search") or {}).get("total", len(values.get("files") or [])))
                elif node_name == "generate_answer" and values.get("model_error"):
                    event.update(model_error=values["model_error"])
                yield event

        snapshot = self.agent_graph.get_state(config).values
        answer = str(snapshot.get("answer", ""))
        # Without a live draft (chitchat, refusals before generation, or drafting disabled) the verified answer is sliced instead.
        if not drafted:
            if answer:
                turn.mark_first_token()
            for start in range(0, len(answer), self.settings.sse_chunk_size):
                yield {
                    "event": "token",
                    "content": answer[start : start + self.settings.sse_chunk_size],
                }
        yield {
            "event": "final",
            "thread_id": thread_id,
            "answer": answer,
            "intent": snapshot.get("intent", ""),
            "rewritten_query": snapshot.get("rewritten_query", ""),
            "citations": self.annotate_citations(snapshot.get("citations", [])),
            "grounded": bool(snapshot.get("grounded", False)),
            "refused": bool(snapshot.get("refused", False)),
            "retrieval_attempts": int(snapshot.get("retrieval_attempts", 0)),
            "fact_check_skipped": bool(snapshot.get("check_skipped", False)),
            "check_mode": str(snapshot.get("check_mode") or ""),
            "files": list(snapshot.get("files") or []),
            "file_search": dict(snapshot.get("file_search") or {}),
            "coverage": dict(snapshot.get("coverage") or {}),
            # A failure the user may simply ask again (the UI offers "重新提问"); auth and
            # balance failures send them to the settings instead.
            "error_kind": str(snapshot.get("error_kind") or ""),
            "retryable": bool(snapshot.get("refused")) and str(snapshot.get("error_kind") or "") in MANUAL_RETRY_KINDS,
            "timings": self._save_timings(turn, snapshot),
        }

    def _graph_stream(self, question: str, config: dict[str, Any], stream_draft: bool, scope: list[str] | None = None):
        """Yield ``(mode, payload)`` pairs; ``messages`` is only requested when drafting."""
        state = self._turn_input(question, scope)
        if stream_draft:
            yield from self.agent_graph.stream(
                state, config=config, stream_mode=["updates", "messages", "custom"], durability=TURN_DURABILITY
            )
        else:
            for update in self.agent_graph.stream(state, config=config, stream_mode="updates", durability=TURN_DURABILITY):
                yield "updates", update

    def followups(self, thread_id: str, question: str = "", n: int = 4) -> dict[str, Any]:
        """Follow-up questions for a conversation's last finished turn.

        ``question`` is the turn the caller saw finish; when the conversation has moved on
        nothing is suggested.
        """
        self.initialize()
        self._restore_thread(thread_id)
        values = self.agent_graph.get_state(self._graph_config(thread_id)).values or {}
        if not values or (question and str(values.get("question", "")).strip() != question.strip()):
            return FollowupResult(SCENE_NONE, []).to_dict()
        with self._followup_lock:
            job = self._followup_jobs.pop(thread_id, None)
        if job is not None and n == self.FOLLOWUPS_AHEAD and job[0] == self._followup_key(values):
            # Written while the answer was checked, from the same question and answer.
            waited = time.perf_counter()
            try:
                result = job[1].result(timeout=float(getattr(self.settings, "followup_timeout_seconds", 4.0)) + 10)
            except Exception as exc:
                logger.warning("Follow-ups written ahead failed, writing them now: %s", exc)
            else:
                timings.save({"kind": "followups", "ahead": True,
                              "total_ms": round((time.perf_counter() - waited) * 1000, 1)}, self.settings.log_dir)
                return result.to_dict()
        return self._suggest_followups(values, n).to_dict()

    # How many follow-ups the app asks for (app.js requestFollowups); those are written ahead.
    FOLLOWUPS_AHEAD = 4

    def _suggest_followups(self, values: dict[str, Any], n: int, ahead: bool = False):
        turn = timings.TurnTimings(kind="followups_written" if ahead else "followups")
        with timings.activate(turn), timings.stage("followups"):
            result = self.followup_service.suggest(values, n=n)
        timings.save(turn.finish(source=getattr(result, "source", ""), fallback=getattr(result, "fallback", ""),
                                 model=self.settings.followup_model or self.settings.llm_model), self.settings.log_dir)
        return result

    @staticmethod
    def _followup_key(values: dict[str, Any]) -> tuple[str, str]:
        return str(values.get("question") or "").strip(), str(values.get("answer") or "")

    def _followups_wanted(self) -> bool:
        if self.followup_service is None or not getattr(self.settings, "log_dir", None):
            return False
        try:
            return user_settings.load_preferences(self.settings).get("followups_enabled", True) is not False
        except Exception:
            return False

    def _turn_values(self, config: dict[str, Any], question: str, scope: list[str] | None) -> dict[str, Any]:
        """The conversation's state as this turn starts (follow-ups read the earlier questions)."""
        try:
            previous = dict(self.agent_graph.get_state(config).values or {})
        except Exception:
            previous = {}
        messages = [*(previous.get("messages") or []), HumanMessage(content=question)]
        return {**previous, "messages": messages, "question": question, "scope_doc_ids": list(scope or [])}

    def _start_followups(self, thread_id: str, values: dict[str, Any]) -> None:
        with self._followup_lock:
            if self._followup_pool is None:
                self._followup_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="followups-ahead")
            future = self._followup_pool.submit(self._suggest_followups, values, self.FOLLOWUPS_AHEAD, True)
            self._followup_jobs.pop(thread_id, None)
            self._followup_jobs[thread_id] = (self._followup_key(values), future)
            while len(self._followup_jobs) > 32:
                self._followup_jobs.popitem(last=False)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Return parsed units and retrieval parents for the document viewer."""
        if not _is_doc_id(doc_id):
            return None
        parsed_path = self.settings.parsed_dir / f"{doc_id}.json"
        if not parsed_path.is_file():
            return None
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        source = str(parsed.get("source", ""))
        units = []
        for unit in parsed.get("units", []):
            item = {
                "heading": str(unit.get("heading", "")),
                "locator": str(unit.get("locator", "")),
                "text": str(unit.get("text", "")),
            }
            # PDF sections spanning pages: [offset, page] pairs let the viewer find a page.
            page_map = (unit.get("metadata") or {}).get("page_map")
            if page_map:
                item["page_map"] = page_map
            units.append(item)
        parents = [
            {key: parent.get(key, "") for key in ("parent_id", "heading", "locator", "text")}
            for parent in (self.parent_store or ParentStoreManager(self.settings)).list_parents(doc_id)
        ]
        document = {
            "doc_id": doc_id,
            "source": source,
            "file_type": str(parsed.get("file_type", "")),
            "units": units,
            "parents": parents,
            "characters": sum(len(unit["text"]) for unit in units),
            "image_only_pages": list((parsed.get("metadata") or {}).get("image_only_pages", [])),
            "has_file": self.original_file(doc_id, source) is not None,
        }
        record = self.library.registry.get_file(doc_id) if self.library else None
        if record:
            document.update(
                source=record["name"], path=record["path"], status=record["status"], size=record["size"],
                image_mode=record.get("image_mode"), notice=record.get("notice"),
            )
        return document

    def original_file(self, doc_id: str, source: str) -> Path | None:
        """Locate an uploaded original; documents ingested by script have none.

        Uploads live at ``upload_dir/<doc_id>/<name>``. Older uploads were stored
        flat as ``upload_dir/<name>``; those are only returned when their content
        still hashes to ``doc_id``, since a same-named file may have replaced them.
        """
        if not _is_doc_id(doc_id):
            return None
        record = self.library.registry.get_file(doc_id) if self.library else None
        if record:
            # Only files in a watched folder (or the upload directory) may be read.
            return readable_original(record, self.library.registry, self.settings.upload_dir)
        name = Path(source).name
        if not name:
            return None
        path = self.settings.upload_dir / doc_id / name
        if path.is_file():
            return path
        legacy = self.settings.upload_dir / name
        if legacy.is_file() and stable_doc_id(legacy) == doc_id:
            return legacy
        return None

    def delete_document(self, doc_id: str) -> None:
        self.initialize()
        if self.library.registry.get_file(doc_id):
            self.library.remove_file(doc_id)
            return
        self.store.delete_document(doc_id)
        self.parent_store.delete_document(doc_id)
        (self.settings.parsed_dir / f"{doc_id}.json").unlink(missing_ok=True)
        shutil.rmtree(self.settings.upload_dir / doc_id, ignore_errors=True)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Remove a conversation: its history and the agent state stored for it."""
        removed = self.conversations.delete(conversation_id)
        self.checkpointer.delete_thread(conversation_id)
        return removed
