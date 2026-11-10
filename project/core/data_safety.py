"""Keep the local databases safe: a check at start-up, a few backups, and two ways back.

``knowledge.sqlite3`` holds the file registry, chunks, vectors and suggestions; nothing else
can rebuild the registry, so it gets three protections (design/DATA_SAFETY.md):

* ``PRAGMA quick_check`` before anything opens it. A damaged file is not opened at all.
* Backups through SQLite's backup API, each verified before it counts. The daily copy also
  notices damage that appears while the app is running.
* Two ways back: restore a backup, or rebuild an empty index from the registry rows that can
  still be read and read every file again.

Both ways swap files, which needs every connection closed. Before the databases are in use
that happens right away; afterwards it is written down and applied at the next start.

``conversations.sqlite3`` is backed up with it but is not the index: a restore leaves it alone
unless it is damaged itself, and damage to it never blocks the library. It cannot be read again
from the files, so it goes back to its newest good copy on its own (``_heal_extras``).
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

BACKUP_DIR = "backups"
# The files a restore or rebuild replaced, kept once so nothing is lost for certain.
REPLACED_DIR = "replaced"
PENDING_FILE = "recovery-pending.json"
MANIFEST = "manifest.json"
# Other databases in the data folder that are backed up when they exist.
EXTRA_DATABASES = ("conversations.sqlite3",)
# Free space needed for a backup, as a multiple of its size (the disk should not end up full).
DISK_MARGIN = 1.5
DAMAGED_MESSAGE = "索引库已损坏。原文件没有受影响，可以从备份恢复，或重新读一遍所有文件。"


class DatabaseDamaged(RuntimeError):
    """The index failed its check; the message is for the user."""


class RecoveryError(RuntimeError):
    """A restore or rebuild could not be done; the message is for the user."""


def check_file(path: str | os.PathLike[str], *, immutable: bool = False) -> str:
    """"" if the file passes ``quick_check`` or does not exist, else what is wrong.

    A file that is merely locked or unreadable is not called damaged. ``immutable`` is for
    backups: it leaves no -wal/-shm files beside them.
    """
    path = Path(path)
    if not path.exists():
        return ""
    target = f"{path.as_uri()}?immutable=1" if immutable else str(path)
    try:
        conn = sqlite3.connect(target, uri=immutable, timeout=15)
        try:
            rows = conn.execute("PRAGMA quick_check(1)").fetchall()
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        logger.warning("Could not check %s: %s", path.name, exc)
        return ""
    except sqlite3.DatabaseError as exc:
        return str(exc)
    first = str(rows[0][0]) if rows else "no result"
    return "" if first == "ok" else first


def read_registry(path: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Registry rows that can still be read from a possibly damaged database.

    A damaged page ends a table scan, so each table is read from both ends: only the rows on
    the damaged pages themselves are lost.
    """
    path = Path(path)
    if not path.exists():
        return [], []
    rows: dict[str, dict[str, dict[str, Any]]] = {"watch_roots": {}, "files": {}}
    keys = {"watch_roots": "root_id", "files": "file_id"}
    for table, by_id in rows.items():
        for order in ("ASC", "DESC"):
            try:
                conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=15)
                conn.row_factory = sqlite3.Row
                try:
                    for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid {order}"):
                        by_id[row[keys[table]]] = dict(row)
                finally:
                    conn.close()
            except sqlite3.DatabaseError as exc:
                logger.warning("Reading %s from %s stopped: %s", table, path.name, exc)
    return list(rows["watch_roots"].values()), list(rows["files"].values())


def read_rules(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """The library's rules (library/privacy.py) that can be read from a database, if any."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute("SELECT * FROM library_rules")]
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        logger.warning("Reading rules from %s failed: %s", path.name, exc)
        return []


# Files kept by name only or removed: what of them a backup may still hold.
_LOST_OR_UNREAD = ("missing", "offline", "recycled", "unsupported")


def _connect_with_vectors(path: Path) -> sqlite3.Connection:
    """A connection that can delete from the vector tables (they need sqlite-vec loaded)."""
    import sqlite_vec

    conn = sqlite3.connect(path, isolation_level=None, timeout=30)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def scrub_database(
    path: Path,
    file_ids: list[str],
    live_rows: dict[str, dict[str, Any] | None],
    rules: list[dict[str, Any]],
) -> None:
    """Take purged files' text out of a backup copy of the index, the way it left the live one.

    Their chunks, vectors, file entries and suggestions go; their registry rows say what the
    live registry says now (removed, or kept by name only, or to be read again), and the
    library's rules are copied over, so restoring this backup brings neither the text nor an
    older choice back. Then the full-text indexes are merged and the file rebuilt (VACUUM), so
    no deleted page keeps a word.
    """
    from library.registry import RULE_FIELDS, RULES_SCHEMA, utc_now

    conn = _connect_with_vectors(path)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        now = utc_now()
        registry = "files" in tables
        if registry:
            # Older backups: the schema this version reads (outside the transaction: executescript commits).
            conn.executescript(RULES_SCHEMA)
            if "content_mode" not in {row[1] for row in conn.execute("PRAGMA table_info(files)")}:
                conn.execute("ALTER TABLE files ADD COLUMN content_mode TEXT")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for doc_id in file_ids:
                if "chunks" in tables:
                    for (rowid,) in conn.execute("SELECT rowid FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall():
                        for table in ("chunks_vec", "chunks_fts", "chunks"):
                            if table in tables:
                                conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
                if "file_entries" in tables:
                    for (rowid,) in conn.execute("SELECT rowid FROM file_entries WHERE doc_id = ?", (doc_id,)).fetchall():
                        for table in ("file_entries_vec", "file_entries_fts", "file_entries"):
                            if table in tables:
                                conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
                if "doc_cards" in tables:
                    conn.execute("DELETE FROM doc_cards WHERE doc_id = ?", (doc_id,))
                for table in ("suggestion_pool", "suggestion_files"):
                    if table in tables:
                        conn.execute(f"DELETE FROM {table} WHERE file_id = ?", (doc_id,))
            if registry:
                lost = ", ".join(f"'{status}'" for status in _LOST_OR_UNREAD)
                for doc_id in file_ids:
                    live = live_rows.get(doc_id)
                    if live is None or live.get("deleted_at"):
                        conn.execute(
                            "UPDATE files SET deleted_at = COALESCE(deleted_at, ?), updated_at = ? WHERE file_id = ?",
                            ((live or {}).get("deleted_at") or now, now, doc_id),
                        )
                        continue
                    name_only = live.get("content_mode") == "name"
                    conn.execute(
                        f"""UPDATE files SET content_mode = ?, chunks = 0, characters = NULL, indexed_sha256 = NULL,
                               parser_version = NULL, notice = NULL,
                               status = CASE WHEN status IN ({lost}) THEN status ELSE ? END, updated_at = ?
                            WHERE file_id = ?""",
                        ("name" if name_only else None, "named" if name_only else "pending", now, doc_id),
                    )
                conn.execute("DELETE FROM library_rules")
                for rule in rules:
                    conn.execute(
                        f"INSERT INTO library_rules({', '.join(RULE_FIELDS)}) VALUES ({', '.join('?' * len(RULE_FIELDS))})",
                        [rule.get(field) for field in RULE_FIELDS],
                    )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        for table in ("chunks_fts", "file_entries_fts"):
            if table in tables:
                conn.execute(f"INSERT INTO {table}({table}) VALUES('optimize')")
        conn.execute("VACUUM")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _copy_database(source: Path, target: Path) -> None:
    """One consistent copy. All pages in one step: in WAL mode that holds a read snapshot and
    does not block writers, while copying in small steps restarts whenever someone writes."""
    src = sqlite3.connect(source, timeout=30)
    try:
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _side_files(path: Path) -> list[Path]:
    return [path.with_name(path.name + suffix) for suffix in ("-wal", "-shm", "-journal")]


def _day_and_time(iso: str) -> str:
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{moment.month}月{moment.day}日 {moment:%H:%M}"


class DataSafety:
    def __init__(self, settings: Any):
        self.settings = settings
        self.problem = ""
        self.checked_at = ""
        self.check_ms: int | None = None
        self.backup_note = ""
        # Something done or noticed about the conversations that the user should read.
        self.notice = ""
        self._prepared = False
        self._in_use = False
        self._lock = threading.RLock()
        self._backup_lock = threading.Lock()
        self._schedule_started = False

    # ---------------- files ----------------

    @property
    def data_dir(self) -> Path:
        return Path(self.settings.data_dir)

    @property
    def backup_root(self) -> Path:
        return self.data_dir / BACKUP_DIR

    @property
    def primary(self) -> Path:
        """The database with the registry and the chunks."""
        path = getattr(self.settings, "embedded_db_path", None)
        return Path(path) if path else self.data_dir / "knowledge.sqlite3"

    def _index_databases(self) -> list[Path]:
        paths = [self.primary]
        registry = getattr(self.settings, "registry_path", None)
        if registry and Path(registry) != self.primary:
            paths.append(Path(registry))
        return paths

    def databases(self) -> list[Path]:
        return [*self._index_databases(), *(self.data_dir / name for name in EXTRA_DATABASES)]

    @property
    def _pending_path(self) -> Path:
        return self.data_dir / PENDING_FILE

    # ---------------- start-up ----------------

    def prepare(self) -> None:
        """Once per run, before anything opens the databases: apply a recovery waiting for
        this start, check the files, and keep a copy when the app was just upgraded (schema
        changes run when the stores open)."""
        with self._lock:
            if self._prepared:
                return
            self._apply_pending()
            self._check()
            if not self.problem:
                self._backup_before_upgrade()
            self._prepared = True

    def open_for_use(self) -> None:
        """Called by whatever is about to open a database connection."""
        with self._lock:
            self.prepare()
            if self.problem:
                raise DatabaseDamaged(DAMAGED_MESSAGE)
            self._in_use = True

    def _check(self) -> None:
        started = time.perf_counter()
        self._heal_extras()
        problems = [f"{path.name}: {problem}" for path in self._index_databases() if (problem := check_file(path))]
        self.problem = "; ".join(problems)
        self.check_ms = round((time.perf_counter() - started) * 1000)
        self.checked_at = datetime.now().isoformat(timespec="seconds")
        if self.problem:
            logger.error("Database check failed: %s", self.problem)
        else:
            logger.info("Database check passed in %d ms", self.check_ms)

    def _heal_extras(self) -> None:
        """A damaged conversations file goes aside and its newest good copy takes its place.

        Neither way back would help it: re-reading the files cannot bring it back, and rolling
        the index back has nothing to do with it. So the library keeps working and the user is
        told what happened. The damaged file is kept with the other replaced files.
        """
        for name in EXTRA_DATABASES:
            path = self.data_dir / name
            problem = check_file(path)
            if not problem:
                continue
            logger.error("%s is damaged: %s", name, problem)
            copy = next(
                (backup for backup in self.list_backups()
                 if name in backup.get("files", []) and not check_file(self.backup_root / backup["name"] / name, immutable=True)),
                None,
            )
            working = path.with_name(name + ".restoring")
            try:
                if copy:
                    shutil.copy2(self.backup_root / copy["name"] / name, working)
                self._set_aside([path])
                if copy:
                    os.replace(working, path)
            except OSError:
                working.unlink(missing_ok=True)
                logger.exception("Could not replace the damaged %s", name)
                self.notice = "对话记录文件损坏了，没能恢复。其他功能不受影响。"
                continue
            if copy:
                self.notice = f"对话记录文件损坏了，已换成 {_day_and_time(copy['created_at'])} 的备份，这之后的对话没有保留。损坏的文件留在备份文件夹的 replaced 里。"
                logger.info("Conversations restored from backup %s", copy["name"])
            else:
                self.notice = "对话记录文件损坏了，又没有可用的备份，只能从空白开始。损坏的文件留在备份文件夹的 replaced 里。"
                logger.info("Conversations started afresh: no good backup")

    def _backup_before_upgrade(self) -> None:
        from core.user_settings import load_preferences, save_preferences

        version = str(getattr(self.settings, "app_version", ""))
        try:
            last = load_preferences(self.settings).get("last_run_version")
            if last == version:
                return
            if last is not None or self.primary.exists():
                self.backup(reason="upgrade", version=str(last or "1.0.0"))
            save_preferences(self.settings, {"last_run_version": version})
        except Exception:
            logger.exception("Backup before upgrade failed")

    # ---------------- backups ----------------

    def backup(self, reason: str = "auto", version: str | None = None) -> dict[str, Any] | None:
        """A verified copy of the databases, or None when it could not be made (see backup_note)."""
        with self._backup_lock:
            sources = [path for path in self.databases() if path.exists()]
            if not sources:
                return None
            size = sum(path.stat().st_size for path in sources)
            self.backup_root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(self.backup_root).free < size * DISK_MARGIN:
                self.backup_note = "磁盘空间不足，没有备份。"
                logger.warning("Backup skipped: not enough free disk space for %d bytes", size)
                return None
            self._remove_leftovers()
            now = datetime.now()
            name = now.strftime("%Y%m%d-%H%M%S")
            working = self.backup_root / f".tmp-{name}"
            working.mkdir(parents=True)
            started = time.perf_counter()
            try:
                copied = []
                for source in sources:
                    target = working / source.name
                    extra = source.name in EXTRA_DATABASES
                    try:
                        _copy_database(source, target)
                    except sqlite3.DatabaseError as exc:
                        if not extra:
                            raise
                        problem = str(exc)
                    else:
                        # A consistent copy that fails means the live file is damaged.
                        problem = check_file(target, immutable=True)
                    if problem and extra:
                        # The index still gets its backup; the next start handles this file.
                        logger.error("Backup found damage in %s: %s", source.name, problem)
                        self.notice = "对话记录文件可能损坏了，下次启动时会处理。"
                        target.unlink(missing_ok=True)
                        continue
                    if problem:
                        self.problem = f"{source.name}: {problem}"
                        logger.error("Backup found damage: %s", self.problem)
                        shutil.rmtree(working, ignore_errors=True)
                        return None
                    copied.append(source.name)
                manifest = {
                    "name": name,
                    "created_at": now.isoformat(timespec="seconds"),
                    "reason": reason,
                    "app_version": version or str(getattr(self.settings, "app_version", "")),
                    "size": sum(path.stat().st_size for path in working.iterdir()),
                    "files": copied,
                }
                (working / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                working.rename(self.backup_root / name)
            except (OSError, sqlite3.Error) as exc:
                shutil.rmtree(working, ignore_errors=True)
                self.backup_note = "备份没有完成。"
                logger.warning("Backup failed: %s", exc)
                return None
            self.backup_note = ""
            self._rotate()
            logger.info("Backup %s (%s): %d bytes in %.1f s", name, reason, manifest["size"], time.perf_counter() - started)
            return manifest

    def scrub_backups(
        self,
        file_ids: list[str],
        live_rows: dict[str, dict[str, Any] | None],
        rules: list[dict[str, Any]],
    ) -> int:
        """Take these files' text out of every backup (``scrub_database``). Returns how many
        backups were changed; a damaged backup is left alone (it cannot be restored anyway)."""
        changed = 0
        with self._backup_lock:
            for backup in self.list_backups():
                folder = self.backup_root / backup["name"]
                names = [name for name in backup.get("files", []) if name not in EXTRA_DATABASES]
                for name in names:
                    path = folder / name
                    if not path.exists():
                        continue
                    if problem := check_file(path, immutable=True):
                        logger.warning("Backup %s/%s is damaged, not scrubbed: %s", backup["name"], name, problem)
                        continue
                    scrub_database(path, file_ids, live_rows, rules)
                    changed += 1
                manifest = {**backup, "size": sum(item.stat().st_size for item in folder.iterdir() if item.is_file() and item.name != MANIFEST)}
                manifest["scrubbed_at"] = datetime.now().isoformat(timespec="seconds")
                (folder / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return changed

    def _remove_leftovers(self) -> None:
        for path in self.backup_root.glob(".tmp-*"):
            shutil.rmtree(path, ignore_errors=True)

    def _rotate(self) -> None:
        keep = max(1, int(getattr(self.settings, "backup_keep", 3)))
        for backup in self.list_backups()[keep:]:
            shutil.rmtree(self.backup_root / backup["name"], ignore_errors=True)

    def list_backups(self) -> list[dict[str, Any]]:
        """Newest first. A folder without a readable manifest never finished and is not listed."""
        backups = []
        if not self.backup_root.is_dir():
            return backups
        for path in self.backup_root.iterdir():
            if not path.is_dir() or path.name.startswith(".") or path.name == REPLACED_DIR:
                continue
            try:
                manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            backups.append({**manifest, "name": path.name})
        return sorted(backups, key=lambda item: item["name"], reverse=True)

    def backup_due(self) -> bool:
        hours = float(getattr(self.settings, "backup_interval_hours", 24.0))
        backups = self.list_backups()
        if not backups:
            return True
        created = datetime.fromisoformat(backups[0]["created_at"])
        return (datetime.now() - created).total_seconds() >= hours * 3600

    def start_schedule(self) -> None:
        """A backup a while after start-up (not while the app is at its busiest), then whenever one is due."""
        with self._lock:
            if self._schedule_started:
                return
            self._schedule_started = True

        def loop() -> None:
            time.sleep(float(getattr(self.settings, "backup_first_delay_seconds", 600.0)))
            while True:
                try:
                    if not self.problem and self.backup_due():
                        self.backup()
                except Exception:
                    logger.exception("Scheduled backup failed")
                time.sleep(3600)

        threading.Thread(target=loop, name="data-backups", daemon=True).start()

    # ---------------- the two ways back ----------------

    def restore(self, name: str) -> dict[str, Any]:
        """Put a backup back. Returns whether the app must restart for it to take effect."""
        backup = next((item for item in self.list_backups() if item["name"] == name), None)
        if backup is None:
            raise RecoveryError("找不到这份备份。")
        return self._run({"action": "restore", "name": name})

    def rebuild(self) -> dict[str, Any]:
        """Start over from the registry: every file is read again."""
        return self._run({"action": "rebuild"})

    def _run(self, action: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._in_use:
                self._pending_path.write_text(json.dumps(action), encoding="utf-8")
                return {"restart_required": True, "action": action["action"]}
            result = self._apply(action)
            self._check()
            self._prepared = True
            return {"restart_required": False, **result}

    def _apply_pending(self) -> None:
        try:
            action = json.loads(self._pending_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable %s", self._pending_path.name)
            action = None
        # Gone first: a recovery that keeps failing must not keep the app from starting.
        self._pending_path.unlink(missing_ok=True)
        if not isinstance(action, dict):
            return
        try:
            logger.info("Applying recovery from the last run: %s", self._apply(action))
        except RecoveryError as exc:
            logger.error("Recovery failed: %s", exc)

    def _apply(self, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("action") == "restore":
            return self._apply_restore(str(action.get("name") or ""))
        if action.get("action") == "rebuild":
            return self._apply_rebuild()
        raise RecoveryError("不认识的恢复操作。")

    def _set_aside(self, paths: list[Path]) -> Path:
        """Move files out of the way, keeping only the latest such set: the first try at a
        recovery may have been the wrong one, but damaged copies should not pile up."""
        target = self.backup_root / REPLACED_DIR
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        for path in paths:
            for file in (path, *_side_files(path)):
                if file.exists():
                    os.replace(file, target / file.name)
        return target

    def _apply_restore(self, name: str) -> dict[str, Any]:
        source = self.backup_root / name
        files = [source / item for item in self._backup_files(source)]
        if not files:
            raise RecoveryError("这份备份是空的。")
        for file in files:
            if problem := check_file(file, immutable=True):
                logger.error("Backup %s is damaged: %s", name, problem)
                raise RecoveryError("这份备份也损坏了，请换一份。")
        # The index goes back to the backup. Other databases only when they are damaged:
        # conversations written since the backup stay.
        replace = [file for file in files if file.name == self.primary.name or check_file(self.data_dir / file.name)]
        staged: list[tuple[Path, Path]] = []
        try:
            # Copied beside the target first, so a full disk leaves everything as it was.
            for file in replace:
                target = self.primary if file.name == self.primary.name else self.data_dir / file.name
                working = target.with_name(target.name + ".restoring")
                shutil.copy2(file, working)
                staged.append((working, target))
            self._set_aside([target for _, target in staged])
            for working, target in staged:
                os.replace(working, target)
        except OSError as exc:
            for working, _ in staged:
                working.unlink(missing_ok=True)
            logger.exception("Restore failed")
            raise RecoveryError(f"恢复没有完成：{exc.strerror or exc}") from exc
        logger.info("Restored backup %s", name)
        return {"action": "restore", "name": name}

    @staticmethod
    def _backup_files(source: Path) -> list[str]:
        try:
            return list(json.loads((source / MANIFEST).read_text(encoding="utf-8"))["files"])
        except (OSError, ValueError, KeyError):
            return []

    def _apply_rebuild(self) -> dict[str, Any]:
        from library.registry import FileRegistry

        roots, files = read_registry(self.primary)
        # Which files are kept by name only and what is excluded must survive a rebuild.
        rules = read_rules(self.primary)
        # Rows on damaged pages may survive in the newest backup that can be read.
        known_roots = {row["root_id"] for row in roots}
        known_files = {row["file_id"] for row in files}
        known_rules = {row["rule_id"] for row in rules}
        for backup in self.list_backups():
            old_roots, old_files = read_registry(self.backup_root / backup["name"] / self.primary.name)
            if old_roots or old_files:
                roots += [row for row in old_roots if row["root_id"] not in known_roots]
                files += [row for row in old_files if row["file_id"] not in known_files]
                rules += [row for row in read_rules(self.backup_root / backup["name"] / self.primary.name)
                          if row["rule_id"] not in known_rules]
                break
        try:
            self._set_aside([self.primary])
            registry = FileRegistry(self.primary)
            try:
                registry.adopt_rows(roots, files, rules)
            finally:
                registry.close()
        except (OSError, sqlite3.Error) as exc:
            logger.exception("Rebuild failed")
            raise RecoveryError(f"重建没有完成：{exc}") from exc
        logger.info("Rebuilt the index from %d folders and %d files", len(roots), len(files))
        return {"action": "rebuild", "folders": len(roots), "files": len(files)}

    # ---------------- reporting ----------------

    def note_damage(self, exc: BaseException) -> None:
        """A running query hit a damaged page: show the same recovery as a failed check.

        SQLite's error does not say which file. When the index passes and the conversations do
        not, offering to restore or re-read the index would not help: the next start handles it.
        """
        if not check_file(self.primary) and any(check_file(self.data_dir / name) for name in EXTRA_DATABASES):
            self.notice = "对话记录文件损坏了，下次启动时会处理。"
            logger.error("Conversations damage noticed: %s", exc)
            return
        self.problem = f"{self.primary.name}: {exc}"
        logger.error("Database damage noticed: %s", exc)

    def status(self) -> dict[str, Any]:
        pending = None
        try:
            pending = json.loads(self._pending_path.read_text(encoding="utf-8")).get("action")
        except (OSError, ValueError, AttributeError):
            pass
        return {
            "state": "damaged" if self.problem else "ok" if self._prepared else "checking",
            "message": DAMAGED_MESSAGE if self.problem else "",
            "checked_at": self.checked_at,
            "check_ms": self.check_ms,
            "backups": self.list_backups(),
            "backup_note": self.backup_note,
            "notice": self.notice,
            "pending": pending,
        }
