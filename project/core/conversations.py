"""Conversations in the data folder: what the user sees and what the agent remembers.

Both live in ``conversations.sqlite3``, away from the knowledge database, so indexing never
holds up a chat and a rebuilt index never touches a chat.

* ``ConversationStore``        the history in the sidebar. One row per conversation; its
                               messages are the UI's own JSON, written whole on every save.
* ``ConversationCheckpointer`` the LangGraph state behind a conversation, so a follow-up
                               keeps its context across restarts. Newest checkpoint only.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# The enterprise-demo tools answered these; their answers are not knowledge to continue from.
_DEMO_INTENTS = ("structured_data", "business_api")
_INTENTS = ("knowledge", "find_file", "chitchat")


class ConversationStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=15000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                updated REAL NOT NULL DEFAULT 0,
                scope TEXT,
                messages TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS conversations_updated ON conversations(updated DESC)")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row(conversation: dict[str, Any]) -> tuple[str, str, float, str | None, str]:
        scope = conversation.get("scope")
        return (
            str(conversation["id"]),
            str(conversation.get("title") or ""),
            float(conversation.get("updated") or 0),
            json.dumps(scope, ensure_ascii=False) if scope else None,
            json.dumps(conversation.get("messages") or [], ensure_ascii=False),
        )

    def list(self) -> list[dict[str, Any]]:
        """Newest first, without the messages: those are read when a conversation is opened."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, updated, scope FROM conversations ORDER BY updated DESC, rowid ASC"
            ).fetchall()
        return [
            {"id": row["id"], "title": row["title"], "updated": row["updated"],
             "scope": json.loads(row["scope"]) if row["scope"] else None}
            for row in rows
        ]

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"], "title": row["title"], "updated": row["updated"],
            "scope": json.loads(row["scope"]) if row["scope"] else None,
            "messages": json.loads(row["messages"]),
        }

    def save(self, conversation: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO conversations (id, title, updated, scope, messages) VALUES (?, ?, ?, ?, ?)",
                self._row(conversation),
            )

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            return self._conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0

    def import_many(self, conversations: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Conversations from an older place (the WebView's localStorage), in one transaction.

        A conversation already here is left alone: it can only have been saved later, so the
        import can safely run again if its first attempt never got its answer back.
        """
        imported = existing = 0
        ids: list[str] = []
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for conversation in conversations:
                    row = self._row({**conversation, "id": conversation.get("id") or uuid.uuid4().hex})
                    added = self._conn.execute(
                        "INSERT OR IGNORE INTO conversations (id, title, updated, scope, messages) VALUES (?, ?, ?, ?, ?)",
                        row,
                    ).rowcount
                    imported += added
                    existing += 1 - added
                    ids.append(row[0])
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")
        return {"imported": imported, "existing": existing, "ids": ids}


def state_from_text(conversation: dict[str, Any]) -> dict[str, Any] | None:
    """Graph state for a conversation saved before its state was kept, rebuilt from its text.

    Enough for a follow-up to know what "它" refers to and for the follow-up suggestions of
    its last answer; the passages are the quotes its citations kept. None if nothing was answered.
    """
    messages: list[Any] = []
    last: dict[str, Any] | None = None
    for message in conversation.get("messages") or []:
        if message.get("role") != "bot" or not message.get("done") or message.get("error"):
            continue
        if not message.get("answer") or message.get("intent") in _DEMO_INTENTS:
            continue
        messages += [HumanMessage(content=str(message.get("question") or "")), AIMessage(content=str(message["answer"]))]
        last = message
    if last is None:
        return None
    citations = [c for c in last.get("citations") or [] if isinstance(c, dict)]
    return {
        "messages": messages,
        "question": str(last.get("question") or ""),
        "answer": str(last["answer"]),
        "intent": last["intent"] if last.get("intent") in _INTENTS else "knowledge",
        "rewritten_query": str(last.get("rewritten") or ""),
        "refused": bool(last.get("refused")),
        "grounded": bool(last.get("grounded")),
        "citations": citations,
        "files": list(last.get("files") or []),
        "file_search": dict(last.get("fileSearch") or {}),
        "context_docs": [
            {"doc_id": c.get("doc_id", ""), "source": c.get("source", ""), "locator": c.get("locator", ""),
             "heading": "", "text": c.get("quote", "")}
            for c in citations
        ],
    }


class ConversationCheckpointer(SqliteSaver):
    """LangGraph checkpoints in SQLite, the newest per conversation only.

    Nothing here goes back in time, and each checkpoint holds the whole state. A conversation
    being deleted may still have an answer running; what that run writes afterwards is dropped.
    """

    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(sqlite3.connect(db_path, check_same_thread=False, timeout=15))
        self._deleted: set[str] = set()
        self.setup()

    def close(self) -> None:
        self.conn.close()

    def put(self, config, checkpoint, metadata, new_versions):
        configurable = config["configurable"]
        thread_id, namespace = str(configurable["thread_id"]), configurable["checkpoint_ns"]
        if thread_id in self._deleted:
            return {"configurable": {"thread_id": thread_id, "checkpoint_ns": namespace, "checkpoint_id": checkpoint["id"]}}
        saved = super().put(config, checkpoint, metadata, new_versions)
        with self.cursor() as cur:
            for table in ("checkpoints", "writes"):
                cur.execute(
                    f"DELETE FROM {table} WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id <> ?",  # noqa: S608
                    (thread_id, namespace, checkpoint["id"]),
                )
        return saved

    def put_writes(self, config, writes, task_id, task_path=""):
        if str(config["configurable"]["thread_id"]) not in self._deleted:
            super().put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        self._deleted.add(str(thread_id))
        super().delete_thread(thread_id)

    def begin_turn(self, thread_id: str) -> None:
        """A new question in a deleted thread starts it afresh (only API clients reuse an id)."""
        self._deleted.discard(str(thread_id))
