"""SQLite registry of watched folders and the local files the knowledge base indexes.

The registry only records where a file lives; originals are never copied. Every row
carries a UUID, ``updated_at``, ``device_id`` and a ``deleted_at`` tombstone so a later
phone/desktop sync can merge rows by id and last-writer-wins without losing deletes.

The tables live in the same SQLite file as the embedded chunk store by default
(``data/knowledge.sqlite3``): one file to back up or sync, and ``file_id`` is the
``doc_id`` of the file's chunks. A separate connection is used, so the registry also
works when ``VECTOR_BACKEND=milvus``.

Writes go through one connection; lookups use read-only connections of their own (see
``db/read_pool.py``), so a question's lookups never queue behind an update that is waiting
for the chunk store to finish writing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import ntpath
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable
import uuid

from db.read_pool import ReaderPool

# "named": kept by name only (``content_mode = "name"``, library/privacy.py): findable by name,
# folder and date, never parsed, nothing about it sent to the model service.
FILE_STATUSES = ("pending", "indexing", "indexed", "named", "failed", "unsupported", "missing", "offline", "recycled")
ROOT_STATUSES = ("active", "missing", "offline", "recycled")
# Files whose parsed text is kept in the index, i.e. those that can be cited and viewed.
# A file in the Recycle Bin keeps its index too (restoring it needs no re-indexing), but
# it is left out of answers and file search until it is restored.
SEARCHABLE_STATUSES = ("indexed", "missing", "offline")
# Files the library cannot reach: the "找不到的文件" list.
LOST_STATUSES = ("missing", "offline", "recycled")

_FILE_FIELDS = (
    "file_id", "root_id", "origin", "path", "path_key", "name", "volume_id", "inode",
    "size", "mtime", "sha256", "indexed_sha256", "status", "error", "chunks", "characters",
    "legacy_doc_id", "created_at", "updated_at", "deleted_at", "device_id", "parser_version",
    "image_mode", "notice", "content_mode",
)
_ROOT_FIELDS = (
    "root_id", "path", "path_key", "volume_id", "status", "created_at", "updated_at",
    "deleted_at", "device_id", "inode", "image_mode",
)
# Columns added after the first release: (table, column, type). NULL means "not set".
_ADDED_COLUMNS = (
    # Registries created before parser versions: NULL reads as version 1.
    ("files", "parser_version", "INTEGER"),
    # How an image file is read ("local" | "cloud" | "name"), or a PDF's scanned pages
    # ("local" | "cloud" | "off" = declined); NULL: the global setting.
    ("files", "image_mode", "TEXT"),
    # Something the UI should ask about, e.g. "scanned_pdf", "ocr_unavailable".
    ("files", "notice", "TEXT"),
    # The folder's own NTFS file ID: finds the folder again after it is renamed or moved.
    ("watch_roots", "inode", "INTEGER"),
    # Image mode chosen when the folder was added; used for images that appear later.
    ("watch_roots", "image_mode", "TEXT"),
    # "name": only the file's name, folder and dates are kept (library/privacy.py); NULL: read it.
    ("files", "content_mode", "TEXT"),
)
# What the library may read (library/privacy.py): path rules inside a watched folder and name
# patterns, for one folder or (root_id NULL) for all of them. Also written into backups.
RULES_SCHEMA = """
CREATE TABLE IF NOT EXISTS library_rules (
    rule_id TEXT PRIMARY KEY,
    root_id TEXT,
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    action TEXT NOT NULL,
    inode INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    device_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS library_rules_root ON library_rules(root_id);
"""
RULE_FIELDS = (
    "rule_id", "root_id", "kind", "value", "action", "inode", "created_at", "updated_at", "deleted_at", "device_id",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id() -> str:
    """Random UUID in compact hex form; also used as the chunk store's ``doc_id``."""
    return uuid.uuid4().hex


def normalize_path(path: str | os.PathLike[str]) -> str:
    return os.path.normpath(os.path.abspath(os.fspath(path)))


def path_key(path: str | os.PathLike[str]) -> str:
    """Comparison key: Windows paths are case-insensitive and use either slash."""
    return os.path.normcase(normalize_path(path))


def _name(path: str) -> str:
    return ntpath.basename(path) if os.name == "nt" else os.path.basename(path)


class FileRegistry:
    # Read-only connections, opened as lookups overlap (a question while the library works).
    read_connections = 2

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Held for every use of the write connection.
        self._lock = threading.RLock()
        self._conn = self._connect()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        self.device_id = self._device_id()
        self._readers = ReaderPool(lambda: self._connect(read_only=True), self.read_connections)

    def _connect(self, read_only: bool = False) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        # The chunk store writes to the same file from another connection.
        conn.execute("PRAGMA busy_timeout=15000")
        if read_only:
            conn.execute("PRAGMA query_only=ON")
        return conn

    def _read(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._readers.connection() as conn:
            return conn.execute(sql, list(params)).fetchall()

    def close(self) -> None:
        self._readers.close()
        with self._lock:
            self._conn.close()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS watch_roots (
                    root_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    path_key TEXT NOT NULL,
                    volume_id INTEGER,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    device_id TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS watch_roots_live_path
                    ON watch_roots(path_key) WHERE deleted_at IS NULL;

                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    root_id TEXT REFERENCES watch_roots(root_id),
                    origin TEXT NOT NULL DEFAULT 'watch',
                    path TEXT NOT NULL,
                    path_key TEXT NOT NULL,
                    name TEXT NOT NULL,
                    volume_id INTEGER,
                    inode INTEGER,
                    size INTEGER,
                    mtime REAL,
                    sha256 TEXT,
                    indexed_sha256 TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error TEXT,
                    chunks INTEGER NOT NULL DEFAULT 0,
                    characters INTEGER,
                    legacy_doc_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    device_id TEXT NOT NULL,
                    parser_version INTEGER
                );
                CREATE UNIQUE INDEX IF NOT EXISTS files_live_path
                    ON files(path_key) WHERE deleted_at IS NULL;
                CREATE INDEX IF NOT EXISTS files_root ON files(root_id);
                CREATE INDEX IF NOT EXISTS files_identity ON files(volume_id, inode);
                CREATE INDEX IF NOT EXISTS files_content ON files(size, sha256);
                CREATE INDEX IF NOT EXISTS files_status ON files(status);
                CREATE INDEX IF NOT EXISTS files_updated ON files(updated_at);

                CREATE TABLE IF NOT EXISTS registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """
            )
            self._conn.executescript(RULES_SCHEMA)
            for table, column, kind in _ADDED_COLUMNS:
                columns = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
                if column not in columns:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

    def adopt_rows(
        self, roots: Iterable[dict[str, Any]], files: Iterable[dict[str, Any]], rules: Iterable[dict[str, Any]] = ()
    ) -> None:
        """Rows saved from a damaged database, into this empty one. The index they described
        is gone, so every live file is read again (except those kept by name only: their
        ``content_mode`` and the rules come along); deleted rows stay as tombstones."""
        reset = {"error": None, "notice": None, "chunks": 0, "characters": None, "indexed_sha256": None, "parser_version": None}
        with self._lock:
            for row in rules:
                self._conn.execute(
                    f"INSERT OR REPLACE INTO library_rules({', '.join(RULE_FIELDS)}) VALUES ({', '.join('?' * len(RULE_FIELDS))})",
                    [row.get(field) for field in RULE_FIELDS],
                )
            for row in roots:
                self._conn.execute(
                    f"INSERT OR REPLACE INTO watch_roots({', '.join(_ROOT_FIELDS)}) VALUES ({', '.join('?' * len(_ROOT_FIELDS))})",
                    [row.get(field) for field in _ROOT_FIELDS],
                )
            for row in files:
                fields = {**row, **reset}
                if row.get("status") != "unsupported":
                    fields["status"] = "pending"
                self._conn.execute(
                    f"INSERT OR REPLACE INTO files({', '.join(_FILE_FIELDS)}) VALUES ({', '.join('?' * len(_FILE_FIELDS))})",
                    [fields.get(field) for field in _FILE_FIELDS],
                )

    def _device_id(self) -> str:
        with self._lock:
            row = self._conn.execute("SELECT value FROM registry_meta WHERE key = 'device_id'").fetchone()
            if row:
                return str(row[0])
            device_id = new_id()
            self._conn.execute("INSERT INTO registry_meta(key, value) VALUES ('device_id', ?)", (device_id,))
            return device_id

    # ---------------- watch roots ----------------

    def add_root(
        self, path: str, volume_id: int | None, inode: int | None = None, image_mode: str | None = None
    ) -> dict[str, Any]:
        now = utc_now()
        row = {
            "root_id": new_id(),
            "path": normalize_path(path),
            "path_key": path_key(path),
            "volume_id": volume_id,
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "deleted_at": None,
            "device_id": self.device_id,
            "inode": inode,
            "image_mode": image_mode,
        }
        with self._lock:
            self._conn.execute(
                f"INSERT INTO watch_roots({', '.join(_ROOT_FIELDS)}) VALUES ({', '.join('?' * len(_ROOT_FIELDS))})",
                [row[field] for field in _ROOT_FIELDS],
            )
        return row

    def get_root(self, root_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        sql = "SELECT * FROM watch_roots WHERE root_id = ?"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        rows = self._read(sql, (root_id,))
        return dict(rows[0]) if rows else None

    def list_roots(self) -> list[dict[str, Any]]:
        rows = self._read("SELECT * FROM watch_roots WHERE deleted_at IS NULL ORDER BY created_at")
        return [dict(row) for row in rows]

    def update_root(self, root_id: str, **fields: Any) -> None:
        if "path" in fields:
            fields["path"] = normalize_path(fields["path"])
            fields["path_key"] = path_key(fields["path"])
        self._update("watch_roots", "root_id", root_id, _ROOT_FIELDS, fields)

    def move_root(self, root_id: str, new_path: str) -> int:
        """Point a watched folder and every file under it at the folder's new location, in
        one transaction (the folder was renamed or moved). Returns the files re-pointed."""
        root = self.get_root(root_id)
        if root is None:
            return 0
        old = normalize_path(root["path"])
        new = normalize_path(new_path)
        now = utc_now()
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_id, path FROM files WHERE root_id = ? AND deleted_at IS NULL", (root_id,)
            ).fetchall()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "UPDATE watch_roots SET path = ?, path_key = ?, updated_at = ? WHERE root_id = ?",
                    (new, path_key(new), now, root_id),
                )
                moved = 0
                for file_id, path in rows:
                    if path_key(path).startswith(path_key(old).rstrip(os.sep) + os.sep):
                        target = new + path[len(old):]
                        self._conn.execute(
                            "UPDATE files SET path = ?, path_key = ?, updated_at = ? WHERE file_id = ?",
                            (target, path_key(target), now, file_id),
                        )
                        moved += 1
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return moved

    def soft_delete_root(self, root_id: str) -> None:
        now = utc_now()
        self._update("watch_roots", "root_id", root_id, _ROOT_FIELDS, {"deleted_at": now})

    def root_for_path(self, path: str) -> dict[str, Any] | None:
        """The live watched folder containing ``path`` (or equal to it)."""
        key = path_key(path)
        for root in self.list_roots():
            if key == root["path_key"] or key.startswith(root["path_key"].rstrip(os.sep) + os.sep):
                return root
        return None

    # ---------------- files ----------------

    def insert_file(self, path: str, *, root_id: str | None, origin: str = "watch", **fields: Any) -> dict[str, Any]:
        now = utc_now()
        normalized = normalize_path(path)
        row: dict[str, Any] = {field: None for field in _FILE_FIELDS}
        row.update(
            file_id=new_id(),
            root_id=root_id,
            origin=origin,
            path=normalized,
            path_key=path_key(normalized),
            name=_name(normalized),
            status="pending",
            chunks=0,
            created_at=now,
            updated_at=now,
            device_id=self.device_id,
        )
        row.update(fields)
        self._check_status(row["status"])
        with self._lock:
            self._conn.execute(
                f"INSERT INTO files({', '.join(_FILE_FIELDS)}) VALUES ({', '.join('?' * len(_FILE_FIELDS))})",
                [row[field] for field in _FILE_FIELDS],
            )
        return row

    def get_file(self, file_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        sql = "SELECT * FROM files WHERE file_id = ?"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        rows = self._read(sql, (file_id,))
        return dict(rows[0]) if rows else None

    def get_by_path(self, path: str) -> dict[str, Any] | None:
        rows = self._read("SELECT * FROM files WHERE path_key = ? AND deleted_at IS NULL", (path_key(path),))
        return dict(rows[0]) if rows else None

    def get_by_legacy_id(self, legacy_doc_id: str) -> dict[str, Any] | None:
        rows = self._read("SELECT * FROM files WHERE legacy_doc_id = ? AND deleted_at IS NULL", (legacy_doc_id,))
        return dict(rows[0]) if rows else None

    def find_by_identity(self, volume_id: int, inode: int) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT * FROM files WHERE volume_id = ? AND inode = ? AND deleted_at IS NULL", (volume_id, inode)
        )
        return [dict(row) for row in rows]

    def find_by_content(self, size: int, sha256: str) -> list[dict[str, Any]]:
        rows = self._read("SELECT * FROM files WHERE size = ? AND sha256 = ? AND deleted_at IS NULL", (size, sha256))
        return [dict(row) for row in rows]

    def list_files(
        self,
        root_id: str | None = None,
        statuses: Iterable[str] | None = None,
        include_deleted: bool = False,
        limit: int | None = None,
        offset: int = 0,
        search: str = "",
    ) -> list[dict[str, Any]]:
        sql, params = self._file_filter(root_id, statuses, include_deleted, search)
        sql = f"SELECT * FROM files {sql} ORDER BY path_key"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [int(limit), int(offset)]
        return [dict(row) for row in self._read(sql, params)]

    def count_files(
        self, root_id: str | None = None, statuses: Iterable[str] | None = None, search: str = ""
    ) -> int:
        sql, params = self._file_filter(root_id, statuses, False, search)
        return int(self._read(f"SELECT COUNT(*) FROM files {sql}", params)[0][0])

    def count_name_only(self) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM files WHERE deleted_at IS NULL AND content_mode = 'name'"
            ).fetchone()[0])

    def files_with_notice(self, notice: str) -> list[dict[str, Any]]:
        rows = self._read("SELECT * FROM files WHERE deleted_at IS NULL AND notice = ?", (notice,))
        return [dict(row) for row in rows]

    def list_single_files(self) -> list[dict[str, Any]]:
        """Live files added one by one (not in a watched folder, not an uploaded copy)."""
        rows = self._read(
            "SELECT * FROM files WHERE deleted_at IS NULL AND root_id IS NULL AND origin != 'upload' ORDER BY path_key"
        )
        return [dict(row) for row in rows]

    def files_under(self, directory: str) -> list[dict[str, Any]]:
        """Live files whose path lies inside ``directory`` (recursively)."""
        prefix = path_key(directory).rstrip(os.sep) + os.sep
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._read(
            "SELECT * FROM files WHERE deleted_at IS NULL AND path_key LIKE ? ESCAPE '\\'", (escaped + "%",)
        )
        # LIKE ignores ASCII case; keep the match exact for case-sensitive file systems.
        return [dict(row) for row in rows if row["path_key"].startswith(prefix)]

    def query_files(
        self,
        *,
        statuses: Iterable[str] = SEARCHABLE_STATUSES,
        extensions: Iterable[str] = (),
        added: tuple[str, str] | None = None,
        modified: tuple[float, float] | None = None,
        either_time: bool = False,
        folder: str = "",
        doc_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Live files matching metadata filters, for "find a file" questions.

        added is an ISO-8601 UTC [start, end) on created_at (when the file was
        first registered); modified an epoch [start, end) on mtime. With
        either_time a file matches when either time is in range. folder matches a
        path fragment, case-insensitively (a folder name or a full path).
        """
        clauses = ["deleted_at IS NULL"]
        params: list[Any] = []
        statuses = list(statuses)
        if statuses:
            clauses.append(f"status IN ({', '.join('?' * len(statuses))})")
            params += statuses
        extensions = [ext.lower() for ext in extensions]
        if extensions:
            clauses.append("(" + " OR ".join("lower(name) LIKE ?" for _ in extensions) + ")")
            params += ["%" + ext for ext in extensions]
        times: list[str] = []
        if added:
            times.append("(created_at >= ? AND created_at < ?)")
            params += list(added)
        if modified:
            times.append("(mtime >= ? AND mtime < ?)")
            params += list(modified)
        if times:
            clauses.append("(" + (" OR " if either_time else " AND ").join(times) + ")")
        if doc_ids is not None:
            clauses.append("file_id IN (SELECT value FROM json_each(?))")
            params.append(json.dumps(sorted(set(doc_ids))))
        rows = [dict(row) for row in self._read(f"SELECT * FROM files WHERE {' AND '.join(clauses)}", params)]
        if folder:
            key = os.path.normcase(folder.strip().strip("\\/").replace("/", os.sep))
            rows = [row for row in rows if key and key in os.path.dirname(row["path_key"])]
        return rows

    def update_file(self, file_id: str, **fields: Any) -> None:
        if "path" in fields:
            fields["path"] = normalize_path(fields["path"])
            fields["path_key"] = path_key(fields["path"])
            fields["name"] = _name(fields["path"])
        if "status" in fields:
            self._check_status(fields["status"])
        self._update("files", "file_id", file_id, _FILE_FIELDS, fields)

    def soft_delete_file(self, file_id: str) -> None:
        self._update("files", "file_id", file_id, _FILE_FIELDS, {"deleted_at": utc_now()})

    def status_counts(self, root_id: str | None = None) -> dict[str, int]:
        sql = "SELECT status, COUNT(*) FROM files WHERE deleted_at IS NULL"
        params: list[Any] = []
        if root_id is not None:
            sql += " AND root_id = ?"
            params.append(root_id)
        rows = self._read(sql + " GROUP BY status", params)
        counts = {status: 0 for status in FILE_STATUSES}
        counts.update({str(status): int(count) for status, count in rows})
        return counts

    def changes_since(self, updated_at: str = "") -> list[dict[str, Any]]:
        """Rows (tombstones included) changed after ``updated_at``; the future sync feed."""
        rows = self._read("SELECT * FROM files WHERE updated_at > ? ORDER BY updated_at", (updated_at,))
        return [dict(row) for row in rows]

    # ---------------- rules (library/privacy.py) ----------------

    def add_rule(self, *, root_id: str | None, kind: str, value: str, action: str, inode: int | None = None) -> dict[str, Any]:
        now = utc_now()
        row = {
            "rule_id": new_id(), "root_id": root_id, "kind": kind, "value": value, "action": action,
            "inode": inode, "created_at": now, "updated_at": now, "deleted_at": None, "device_id": self.device_id,
        }
        with self._lock:
            self._conn.execute(
                f"INSERT INTO library_rules({', '.join(RULE_FIELDS)}) VALUES ({', '.join('?' * len(RULE_FIELDS))})",
                [row[field] for field in RULE_FIELDS],
            )
        return row

    def get_rule(self, rule_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM library_rules WHERE rule_id = ? AND deleted_at IS NULL", (rule_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_rules(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM library_rules"
        if not include_deleted:
            sql += " WHERE deleted_at IS NULL"
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    def update_rule(self, rule_id: str, **fields: Any) -> None:
        self._update("library_rules", "rule_id", rule_id, RULE_FIELDS, fields)

    def soft_delete_rule(self, rule_id: str) -> None:
        self._update("library_rules", "rule_id", rule_id, RULE_FIELDS, {"deleted_at": utc_now()})

    # ---------------- helpers ----------------

    @staticmethod
    def _file_filter(
        root_id: str | None, statuses: Iterable[str] | None, include_deleted: bool, search: str
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if root_id is not None:
            clauses.append("root_id = ?")
            params.append(root_id)
        statuses = list(statuses or [])
        if statuses:
            clauses.append(f"status IN ({', '.join('?' * len(statuses))})")
            params += statuses
        if search:
            clauses.append("path LIKE ? ESCAPE '\\'")
            params.append("%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        return ("WHERE " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def _check_status(status: str) -> None:
        if status not in FILE_STATUSES:
            raise ValueError(f"invalid file status: {status}")

    def _update(self, table: str, key: str, value: str, allowed: tuple[str, ...], fields: dict[str, Any]) -> None:
        unknown = set(fields) - set(allowed)
        if unknown:
            raise ValueError(f"unknown {table} fields: {sorted(unknown)}")
        fields = {**fields, "updated_at": utc_now()}
        assignments = ", ".join(f"{field} = ?" for field in fields)
        with self._lock:
            self._conn.execute(f"UPDATE {table} SET {assignments} WHERE {key} = ?", [*fields.values(), value])
