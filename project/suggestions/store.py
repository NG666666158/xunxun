"""Local tables behind the home-page suggestions.

They live in the registry's SQLite file (``knowledge.sqlite3``) on a connection of their
own, like the file registry. Nothing here is synced or uploaded.

* ``suggestion_pool``    one row per candidate question: source file, kind, how it was
                         written, whether retrieval verified it, shown/clicked counts and
                         the "不感兴趣" flag.
* ``suggestion_files``   which version of each file the pool was built from.
* ``suggestion_blocked`` files the user asked never to be suggested again.
* ``question_log``       questions the user sent, kept for "接着上次" only: the newest
                         200, at most 90 days old.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable
import uuid

HISTORY_LIMIT = 200
HISTORY_DAYS = 90

_POOL_FIELDS = (
    "id", "file_id", "title_key", "file_name", "question", "qtype", "method", "template",
    "verified", "verified_at", "shown_count", "click_count", "dismissed", "last_shown_at", "created_at",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")


class SuggestionStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=15000")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS suggestion_pool (
                    id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    title_key TEXT NOT NULL DEFAULT '',
                    file_name TEXT NOT NULL DEFAULT '',
                    question TEXT NOT NULL,
                    qtype TEXT NOT NULL,
                    method TEXT NOT NULL,
                    template TEXT NOT NULL DEFAULT '',
                    verified INTEGER NOT NULL DEFAULT 0,
                    verified_at TEXT,
                    shown_count INTEGER NOT NULL DEFAULT 0,
                    click_count INTEGER NOT NULL DEFAULT 0,
                    dismissed INTEGER NOT NULL DEFAULT 0,
                    last_shown_at TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(file_id, question)
                );
                CREATE INDEX IF NOT EXISTS suggestion_pool_file ON suggestion_pool(file_id);

                CREATE TABLE IF NOT EXISTS suggestion_files (
                    file_id TEXT PRIMARY KEY,
                    sha256 TEXT,
                    name TEXT NOT NULL DEFAULT '',
                    version INTEGER NOT NULL DEFAULT 0,
                    excluded TEXT NOT NULL DEFAULT '',
                    ai_done INTEGER NOT NULL DEFAULT 0,
                    built_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS suggestion_blocked (
                    file_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT '',
                    blocked_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS question_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    asked_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS question_log_time ON question_log(asked_at);
                """
            )

    # ---------------- pool ----------------

    def replace_questions(self, file_id: str, method: str, rows: Iterable[dict[str, Any]]) -> int:
        """Set a file's questions written by ``method``; counters of kept questions survive."""
        rows = list(rows)
        now = iso(utc_now())
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                keep = [row["question"] for row in rows]
                existing = {
                    r["question"]: r
                    for r in self._conn.execute(
                        "SELECT * FROM suggestion_pool WHERE file_id = ? AND method = ?", (file_id, method)
                    )
                }
                for question, old in existing.items():
                    if question not in keep:
                        self._conn.execute("DELETE FROM suggestion_pool WHERE id = ?", (old["id"],))
                for row in rows:
                    old = existing.get(row["question"])
                    if old is not None:
                        self._conn.execute(
                            "UPDATE suggestion_pool SET title_key = ?, file_name = ?, qtype = ?, template = ?, "
                            "verified = ?, verified_at = ? WHERE id = ?",
                            (row.get("title_key", ""), row.get("file_name", ""), row["qtype"],
                             row.get("template", ""), int(bool(row.get("verified"))), now, old["id"]),
                        )
                        continue
                    self._conn.execute(
                        f"INSERT OR IGNORE INTO suggestion_pool({', '.join(_POOL_FIELDS)}) "
                        f"VALUES ({', '.join('?' * len(_POOL_FIELDS))})",
                        (uuid.uuid4().hex, file_id, row.get("title_key", ""), row.get("file_name", ""),
                         row["question"], row["qtype"], method, row.get("template", ""),
                         int(bool(row.get("verified"))), now, 0, 0, 0, None, now),
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return len(rows)

    def delete_file(self, file_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM suggestion_pool WHERE file_id = ?", (file_id,))
            self._conn.execute("DELETE FROM suggestion_files WHERE file_id = ?", (file_id,))

    def rename_file(self, file_id: str, name: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE suggestion_pool SET file_name = ? WHERE file_id = ?", (name, file_id))

    def questions(self, file_id: str | None = None, verified_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM suggestion_pool WHERE 1 = 1"
        params: list[Any] = []
        if file_id is not None:
            sql += " AND file_id = ?"
            params.append(file_id)
        if verified_only:
            sql += " AND verified = 1"
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql + " ORDER BY created_at, question", params)]

    def candidates(self) -> list[dict[str, Any]]:
        """Verified, not dismissed, from files that are not blocked."""
        with self._lock:
            return [
                dict(row)
                for row in self._conn.execute(
                    "SELECT * FROM suggestion_pool WHERE verified = 1 AND dismissed = 0 "
                    "AND file_id NOT IN (SELECT file_id FROM suggestion_blocked)"
                )
            ]

    def get(self, question_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM suggestion_pool WHERE id = ?", (question_id,)).fetchone()
        return dict(row) if row else None

    def record(self, action: str, ids: Iterable[str]) -> int:
        ids = [str(item) for item in ids][:50]
        if not ids:
            return 0
        now = iso(utc_now())
        sql = {
            "shown": "UPDATE suggestion_pool SET shown_count = shown_count + 1, last_shown_at = ? WHERE id = ?",
            "click": "UPDATE suggestion_pool SET click_count = click_count + 1, last_shown_at = ? WHERE id = ?",
            "dismiss": "UPDATE suggestion_pool SET dismissed = 1, last_shown_at = ? WHERE id = ?",
        }[action]
        with self._lock:
            return sum(self._conn.execute(sql, (now, item)).rowcount for item in ids)

    def has_feedback(self) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM suggestion_pool WHERE click_count > 0 OR dismissed = 1 LIMIT 1"
            ).fetchone()
        return row is not None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT method, qtype, verified, COUNT(*) AS n, COUNT(DISTINCT file_id) AS files "
                "FROM suggestion_pool GROUP BY method, qtype, verified"
            ).fetchall()
            files = self._conn.execute(
                "SELECT COUNT(DISTINCT file_id) FROM suggestion_pool WHERE verified = 1"
            ).fetchone()[0]
        verified = sum(row["n"] for row in rows if row["verified"])
        return {
            "verified": verified,
            "rejected": sum(row["n"] for row in rows if not row["verified"]),
            "files": int(files),
            "by_kind": {
                f"{row['method']}/{row['qtype']}": int(row["n"]) for row in rows if row["verified"]
            },
        }

    # ---------------- per-file build state ----------------

    def file_states(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {row["file_id"]: dict(row) for row in self._conn.execute("SELECT * FROM suggestion_files")}

    def set_file_state(self, file_id: str, **fields: Any) -> None:
        current = self.file_states().get(file_id) or {
            "file_id": file_id, "sha256": None, "name": "", "version": 0, "excluded": "", "ai_done": 0,
        }
        current.update(fields, built_at=iso(utc_now()))
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO suggestion_files(file_id, sha256, name, version, excluded, ai_done, built_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (file_id, current["sha256"], current["name"], int(current["version"]), current["excluded"],
                 int(bool(current["ai_done"])), current["built_at"]),
            )

    # ---------------- blocked files ----------------

    def block_file(self, file_id: str, name: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO suggestion_blocked(file_id, name, blocked_at) VALUES (?, ?, ?)",
                (file_id, name, iso(utc_now())),
            )

    def unblock_file(self, file_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM suggestion_blocked WHERE file_id = ?", (file_id,))

    def blocked_files(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._conn.execute("SELECT * FROM suggestion_blocked ORDER BY blocked_at DESC")]

    # ---------------- question log ----------------

    def log_question(self, question: str, now: datetime | None = None) -> None:
        question = " ".join(question.split())[:500]
        if not question:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO question_log(question, asked_at) VALUES (?, ?)", (question, iso(now or utc_now()))
            )
        self.prune_history(now)

    def prune_history(self, now: datetime | None = None) -> int:
        """Keep only the newest ``HISTORY_LIMIT`` questions of the last ``HISTORY_DAYS`` days."""
        cutoff = iso((now or utc_now()) - timedelta(days=HISTORY_DAYS))
        with self._lock:
            removed = self._conn.execute("DELETE FROM question_log WHERE asked_at < ?", (cutoff,)).rowcount
            removed += self._conn.execute(
                "DELETE FROM question_log WHERE id NOT IN "
                "(SELECT id FROM question_log ORDER BY asked_at DESC, id DESC LIMIT ?)",
                (HISTORY_LIMIT,),
            ).rowcount
        return removed

    def recent_questions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(row)
                for row in self._conn.execute(
                    "SELECT question, asked_at FROM question_log ORDER BY asked_at DESC, id DESC LIMIT ?", (limit,)
                )
            ]

    def history_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM question_log").fetchone()[0])

    def clear_history(self) -> int:
        with self._lock:
            return self._conn.execute("DELETE FROM question_log").rowcount
