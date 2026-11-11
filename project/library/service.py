"""Local file model: watched folders, background indexing, change handling and reconciliation.

Originals stay where the user keeps them; the registry records their path and identity
(volume serial + NTFS file ID, plus size + sha256). Parsed text, parents and chunks are
keyed by the registry ``file_id``, so a renamed or moved file keeps its index, and a
missing or offline file stays searchable and viewable.

Files and watched folders are followed when they move: a path that disappears is looked
up again by its NTFS file ID (``winfs.resolve``) before it is called missing. A move to
the Recycle Bin is the ``recycled`` state (index kept, left out of answers); restoring the
file brings it back without re-indexing. Watched folders are watched recursively, and
their parent folders non-recursively, so a renamed folder is noticed at once. Single
imported files have no watcher: they are checked at start-up, when the UI asks
(``sync``, e.g. when the window regains focus; ``check_file`` when one is shown) and by a
stat sweep every ``library_sweep_seconds``.
"""

from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
import inspect
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Iterator

from ingestion import ocr_runtime
from ingestion.images import DEFAULT_IMAGE_MODE, IMAGE_MODES, estimate_seconds, forget_readings, is_image
from ingestion.parsers import parser_version, pdf_image_mode, stable_doc_id, usable_extensions
from ingestion.priority import BELOW_NORMAL, lower_current_thread
from services.model_clients import background_calls
from . import privacy, winfs
from .jobs import LIGHT, JobQueue, Slots, estimate as estimate_job
from .registry import LOST_STATUSES, FileRegistry, SEARCHABLE_STATUSES, new_id, normalize_path, path_key
from .scanner import (
    FileFacts,
    classify,
    SizeLimits,
    estimate_characters,
    file_facts,
    path_excluded,
    sha256_file,
    summarize,
    volume_available,
    walk,
)

logger = logging.getLogger(__name__)
_LEGACY_ID_LENGTH = 24
# Where a file / folder with this NTFS ID is now (tests replace it).
resolve_file_id = winfs.resolve
# Image choices a file can hold: image files use IMAGE_MODES; a PDF's scanned pages
# "local" / "cloud", or "off" once the user declined.
PDF_IMAGE_MODES = ("local", "cloud", "off")
# What the import estimate assumes (design/LIBRARY_SYNC_AND_IMAGES.md §4): parsing at about
# 30k characters a second plus a little per file, ~1.5 s per embedding request of 32 chunks.
SECONDS_PER_FILE = 0.2
CHARACTERS_PER_SECOND = 30000
SECONDS_PER_EMBEDDING_REQUEST = 1.5
CHARACTERS_PER_CHUNK = 600
# A file that failed on the network or a model service is tried again after these delays
# (seconds); after the last one it waits for the next start, as before.
RETRY_DELAYS = (15, 30, 60, 120, 240)
# Change kinds the status reports, e.g. for "寻寻发现 3 个新文件".
CHANGE_KINDS = (
    "new", "changed", "moved", "removed", "missing", "offline", "recycled", "restored",
    "folder_added", "folder_moved", "folder_missing", "folder_recycled", "folder_restored", "folder_removed",
)


class LibraryError(ValueError):
    """A user-facing validation error (bad folder, overlapping roots, ...)."""


class ReadingStopped(Exception):
    """A file being read was switched to name-only or removed: stop sending its text."""


def _transient(exc: BaseException) -> bool:
    """A failure of the network or a model service (timeout, rate limit, 5xx), not of the file."""
    for error in (exc, exc.__cause__, exc.__context__):
        if error is None:
            continue
        name = type(error).__name__
        module = type(error).__module__.split(".")[0]
        if module in ("openai", "httpx", "httpcore", "requests", "urllib3") or isinstance(error, (TimeoutError, ConnectionError)):
            return True
        if any(word in name for word in ("Timeout", "Connection", "RateLimit", "ServiceUnavailable")):
            return True
    return False


def _account_refused(exc: BaseException) -> bool:
    """The model service refused the account (bad key 401, no balance 402, no access 403).

    Retrying within minutes cannot help, and every retry reads the whole file again (a
    scanned PDF means OCR on every page): right after a start that kept the machine busy
    for the user's first questions. Such a file waits for the next start instead."""
    for error in (exc, exc.__cause__, exc.__context__):
        status = getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)
        if status in (401, 402, 403):
            return True
    return False


def _clean_path(raw: str) -> str:
    return (raw or "").strip().strip('"').strip("'")


def _absolute(raw: str) -> str:
    """A path the user picked, dropped or typed, normalized; it must be absolute."""
    text = os.path.expanduser(_clean_path(raw))
    if not text or not os.path.isabs(text):
        raise LibraryError("请提供完整的绝对路径。")
    return normalize_path(text)


def _located(found: str) -> bool:
    """A path from ``resolve_file_id`` (not one of its "not found" / "cannot tell" answers)."""
    return found not in (winfs.NOT_FOUND, winfs.UNSUPPORTED)


def _indexer_options(indexer: Callable[..., Any] | None, options: dict[str, Any]) -> dict[str, Any]:
    """The options the indexer accepts (``image_mode``, ``pause``, ``progress``): all of them
    for the app's (it takes ``**options``), only the named ones for a test double."""
    if indexer is None:
        return {}
    try:
        parameters = inspect.signature(indexer).parameters.values()
    except (TypeError, ValueError):
        return {}
    if any(p.kind is p.VAR_KEYWORD for p in parameters):
        return options
    names = {p.name for p in parameters}
    return {name: value for name, value in options.items() if name in names}


class LibraryService:
    def __init__(
        self,
        settings: Any,
        registry: FileRegistry,
        store: Any,
        parent_store: Any,
        indexer: Callable[..., tuple] | None,
    ):
        """``indexer(path, file_id, source, **options) -> (chunks, characters[, info])`` builds a
        file's index. Options (when it takes them): ``image_mode`` and ``pause``; ``info`` may
        carry the parser's ``notice``."""
        self.settings = settings
        self.registry = registry
        self.store = store
        self.parent_store = parent_store
        self.indexer = indexer
        self.extensions = usable_extensions(tuple(settings.allowed_extensions))
        # Files are read where they are, so the browser upload limit does not apply here.
        self.max_bytes = SizeLimits(
            int(settings.max_local_file_mb) * 1024 * 1024, int(settings.max_local_text_mb) * 1024 * 1024
        )
        self.max_files = int(getattr(settings, "watch_max_files", 50000))
        self._mutex = threading.RLock()
        # Files waiting to be indexed: the user's first, then the cheapest (see jobs.py).
        self._queue = JobQueue()
        self._slots = Slots(
            int(getattr(settings, "index_ocr_jobs", 2)), int(getattr(settings, "index_long_jobs", 1))
        )
        self.workers = max(1, int(getattr(settings, "index_workers", 4)))
        background_calls.resize(int(getattr(settings, "index_api_concurrency", 4)))
        # Files the user asked to index again: parsed even when their content is unchanged.
        self._forced: set[str] = set()
        self._wake = threading.Condition()
        # Jobs running now, one per busy worker: file_id -> what the status shows.
        self._running: dict[str, dict[str, Any]] = {}
        self._batch = {"done": 0, "failed": 0, "skipped": 0}
        self._scanning: set[str] = set()
        self._stopping = threading.Event()
        self._threads: list[threading.Thread] = []
        self.watcher = None
        # Questions being answered: the queue waits meanwhile (see pause_for_questions).
        self._answering = 0
        self._answer_cond = threading.Condition()
        # Recent changes for the UI's toasts; seq only grows.
        self._changes: deque[dict[str, Any]] = deque(maxlen=2000)
        self._change_seq = 0
        self._changes_lock = threading.Lock()
        self._last_sync: tuple[float, dict[str, Any]] | None = None
        self._last_sweep = time.monotonic()
        self._hidden: tuple[float, frozenset[str]] | None = None
        # Transient failures waiting to be retried: file_id -> (attempts so far, due on time.monotonic()).
        self._retries: dict[str, tuple[int, float]] = {}
        # Optional listeners (file_id) -> None, e.g. the home-page suggestions.
        self.on_indexed: Callable[[str], None] | None = None
        self.on_dropped: Callable[[str], None] | None = None
        # Called with the ids of files whose text was dropped for good (removed, excluded or
        # kept by name only), so it also leaves the database's free pages and the backups.
        self.on_purged: Callable[[list[str]], None] | None = None
        # Rules (library/privacy.py), read once and kept until one changes.
        self._rules_cache: list[dict[str, Any]] | None = None
        # Files being read that were switched to name-only or removed: the reading stops.
        self._cancelled: set[str] = set()

    # ======================================================================
    # lifecycle
    # ======================================================================

    def start(self, watch: bool = True) -> None:
        """Migrate, reconcile, then start the index worker, folder watchers and volume monitor."""
        try:
            self.migrate_legacy()
        except Exception:
            logger.exception("Legacy document migration failed")
        self.reconcile()
        for number in range(self.workers):
            self._spawn(self._worker_loop, f"library-indexer-{number}")
        if watch:
            from .watcher import FolderWatcher

            self.watcher = FolderWatcher(self, float(getattr(self.settings, "watch_debounce_seconds", 2.0)))
            self.watcher.start()
            for root in self.registry.list_roots():
                if root["status"] == "active":
                    self.watcher.watch(root)
                else:
                    # Its parent is still watched: a folder renamed back or restored is seen at once.
                    self.watcher.watch_parent(root)
        self._spawn(self._monitor_loop, "library-monitor")

    def stop(self) -> None:
        self._stopping.set()
        with self._wake:
            self._wake.notify_all()
        with self._answer_cond:
            self._answer_cond.notify_all()
        if self.watcher:
            self.watcher.stop()

    def _spawn(self, target: Callable[..., Any], name: str, *args: Any) -> threading.Thread:
        thread = threading.Thread(target=target, args=args, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)
        return thread

    # ======================================================================
    # watched folders
    # ======================================================================

    def validate_root_path(self, raw: str) -> str:
        text = _clean_path(raw)
        if not text:
            raise LibraryError("请输入文件夹路径。")
        path = os.path.expanduser(text)
        if not os.path.isabs(path):
            raise LibraryError("请输入完整的绝对路径，例如 D:\\资料\\合同。")
        path = normalize_path(path)
        if not os.path.isdir(path):
            raise LibraryError(f"文件夹不存在或无法访问：{path}")
        key = path_key(path)
        for root in self.registry.list_roots():
            other = root["path_key"]
            if key == other:
                raise LibraryError("这个文件夹已经在监控列表中。")
            if key.startswith(other.rstrip(os.sep) + os.sep):
                raise LibraryError(f"它已包含在监控目录 {root['path']} 中。")
            if other.startswith(key.rstrip(os.sep) + os.sep):
                raise LibraryError(f"它包含已监控的目录 {root['path']}，请先移除那个目录。")
        return path

    def preview_root(self, raw: str) -> dict[str, Any]:
        path = self.validate_root_path(raw)
        mode = getattr(self.settings, "image_understanding", "off")
        summary = summarize(path, self.extensions, self.max_bytes, self.max_files, count_figures=mode != "off")
        return {"path": path, **summary.to_dict(), "max_files": self.max_files, "image_understanding": mode}

    def add_root(
        self, raw: str, background: bool = True, image_mode: str | None = None, source: str = "user"
    ) -> dict[str, Any]:
        """Watch a folder. ``image_mode`` is how its images are read, now and when new ones appear."""
        path = self.validate_root_path(raw)
        decision = self._decision(path, single=True)
        if decision.excluded:
            raise LibraryError(f"这个文件夹在排除范围里（{decision.rule['value']}），没有添加。可以在设置的“数据去向”里改。")
        info = os.stat(path)
        with self._mutex:
            root = self.registry.add_root(
                path,
                volume_id=info.st_dev,
                inode=int(info.st_ino) or None,
                image_mode=image_mode if image_mode in IMAGE_MODES else None,
            )
        if self.watcher:
            self.watcher.watch(root)
        if background:
            self._scanning.add(root["root_id"])
            self._spawn(self._scan_in_background, f"scan-{root['root_id'][:8]}", root["root_id"], source)
        else:
            self.scan_root(root["root_id"], source=source)
        return self.describe_root(root)

    def _scan_in_background(self, root_id: str, source: str = "auto") -> None:
        try:
            self.scan_root(root_id, source=source)
        except Exception:
            logger.exception("Scanning watched folder %s failed", root_id)
        finally:
            self._scanning.discard(root_id)

    def remove_root(self, root_id: str, source: str = "user") -> int:
        root = self.registry.get_root(root_id)
        if root is None:
            raise KeyError(root_id)
        if self.watcher:
            self.watcher.unwatch(root_id)
            self.watcher.forget_parent(root_id)
        with self._mutex:
            files = self.registry.list_files(root_id=root_id)
            for record in files:
                self.remove_file(record["file_id"], source=None)
            for rule in self._rules():
                if rule["root_id"] == root_id:
                    self.registry.soft_delete_rule(rule["rule_id"])
            self._rules_changed()
            self.registry.soft_delete_root(root_id)
        self._note("folder_removed", None, source=source, root_id=root_id, path=root["path"], files=len(files))
        return len(files)

    def rescan_root(self, root_id: str) -> dict[str, Any]:
        root = self.registry.get_root(root_id)
        if root is None:
            raise KeyError(root_id)
        self._scanning.add(root_id)
        self._spawn(self._scan_in_background, f"scan-{root_id[:8]}", root_id)
        return self.describe_root(root)

    def describe_root(self, root: dict[str, Any]) -> dict[str, Any]:
        counts = self.registry.status_counts(root["root_id"])
        return {
            "root_id": root["root_id"],
            "path": root["path"],
            "status": root["status"],
            "scanning": root["root_id"] in self._scanning,
            "counts": counts,
            "total": sum(counts.values()),
            "image_mode": root.get("image_mode"),
            # "name" when the whole folder is kept by name only; its own rules (exclusions,
            # sub-folders kept by name only) for the folder's settings sheet.
            "content_mode": self._decision(root["path"], root).mode,
            "rules": [
                privacy.describe_rule(rule, root["path"]) for rule in self._rules()
                if rule["root_id"] == root["root_id"]
            ],
            "created_at": root["created_at"],
            "updated_at": root["updated_at"],
        }

    def list_roots(self) -> list[dict[str, Any]]:
        return [self.describe_root(root) for root in self.registry.list_roots()]

    def scan_root(self, root_id: str, source: str = "auto") -> dict[str, int]:
        """Bring the registry in line with a folder's current contents and queue indexing.

        A scan the user started (adding the folder) reports one ``folder_added`` change;
        other scans report each new, moved or missing file."""
        root = self.registry.get_root(root_id)
        if root is None:
            return {}
        stats = {"new": 0, "changed": 0, "moved": 0, "missing": 0}
        seen: set[str] = set()
        lost_by_size = self._lost_records_by_size()
        per_file = source != "user"
        # Folders renamed while the app was closed keep their rules (followed by file ID).
        self._follow_rules(root)
        skip = self._skip_for(root)
        for entry in walk(root["path"], self.extensions, self.max_bytes, max_files=self.max_files, skip=skip):
            if self._stopping.is_set():
                break
            seen.add(path_key(entry.facts.path))
            with self._mutex:
                outcome = self._upsert_entry(
                    entry.facts, entry.supported, entry.reason, root, lost_by_size, per_file, urgent=not per_file
                )
            if outcome in stats:
                stats[outcome] += 1
        with self._mutex:
            for record in self.registry.list_files(root_id=root_id):
                if self._decision(record["path"], root).excluded:
                    # Excluded after it was added (e.g. a rule restored with a backup): let it go.
                    self.remove_file(record["file_id"], source=None)
                    continue
                if record["path_key"] in seen or os.path.exists(record["path"]):
                    continue
                if record["status"] not in LOST_STATUSES:
                    self._mark_lost(record)
                    stats["missing"] += 1
        if not per_file:
            self._note("folder_added", None, source=source, root_id=root_id, path=root["path"], files=stats["new"])
        logger.info("Scanned %s: %s", root["path"], stats)
        return stats

    # ======================================================================
    # adding files and folders
    # ======================================================================

    def inspect(self, paths: Iterable[str]) -> dict[str, Any]:
        """What adding these files and folders would do, before anything is registered:
        counts by type, the images among them (also inside folders) and an estimate of the
        time and cloud calls per image mode. The UI asks how to read images from this."""
        figures_mode = getattr(self.settings, "image_understanding", "off")
        items: list[dict[str, Any]] = []
        totals: dict[str, Any] = {
            "files": 0, "indexable": 0, "unsupported": 0, "too_large": 0, "already": 0, "images": 0,
            "image_bytes": 0, "bytes": 0, "estimated_characters": 0, "estimated_figures": 0, "truncated": False,
            "by_type": {},
        }
        for raw in paths:
            path = _clean_path(raw)
            try:
                path = _absolute(raw)
                watched = self._watched_root(path)
                if watched is not None:
                    items.append({"path": path, "kind": "folder", "ok": True, "already": True,
                                  "root_id": watched["root_id"], "files": 0, "indexable": 0, "images": 0})
                elif os.path.isdir(path):
                    items.append(self._inspect_folder(path, totals, figures_mode))
                elif os.path.isfile(path):
                    items.append(self._inspect_file(path, totals))
                else:
                    raise LibraryError(f"文件或文件夹不存在：{path}")
            except LibraryError as exc:
                items.append({"path": path, "kind": "error", "ok": False, "error": str(exc)})
        preference = getattr(self.settings, "image_import_mode", "ask")
        return {
            "items": items,
            "totals": totals,
            "estimate": self._estimate(totals, figures_mode),
            "image_mode": preference,
            "ask_image_mode": bool(totals["images"]) and preference not in IMAGE_MODES,
            "default_image_mode": preference if preference in IMAGE_MODES else DEFAULT_IMAGE_MODE,
            "ocr_available": ocr_runtime.activate(self.settings.data_dir),
        }

    def _inspect_folder(self, path: str, totals: dict[str, Any], figures_mode: str) -> dict[str, Any]:
        path = self.validate_root_path(path)
        skip = (lambda item, _folder: self._decision(item, single=True).excluded) if self._rules() else None
        summary = summarize(
            path, self.extensions, self.max_bytes, self.max_files, count_figures=figures_mode != "off", skip=skip
        )
        for key in (
            "files", "indexable", "unsupported", "too_large", "bytes", "estimated_characters", "estimated_figures",
            "images", "image_bytes",
        ):
            totals[key] += getattr(summary, key)
        totals["truncated"] = totals["truncated"] or summary.truncated
        for suffix, count in summary.by_type.items():
            totals["by_type"][suffix] = totals["by_type"].get(suffix, 0) + count
        return {
            "path": path, "kind": "folder", "ok": True, "already": False, "files": summary.files,
            "indexable": summary.indexable, "images": summary.images, "by_type": summary.by_type,
            "truncated": summary.truncated,
        }

    def _watched_root(self, path: str) -> dict[str, Any] | None:
        key = path_key(path)
        return next((root for root in self.registry.list_roots() if root["path_key"] == key), None)

    def _inspect_file(self, path: str, totals: dict[str, Any]) -> dict[str, Any]:
        facts = file_facts(path)
        supported, reason = classify(path, facts.size, self.extensions, self.max_bytes)
        existing = self.registry.get_by_path(path)
        suffix = Path(path).suffix.lower()
        image = supported and is_image(path)
        totals["files"] += 1
        totals["bytes"] += facts.size
        if existing:
            totals["already"] += 1
        elif supported:
            totals["indexable"] += 1
            totals["by_type"][suffix] = totals["by_type"].get(suffix, 0) + 1
            if image:
                totals["images"] += 1
                totals["image_bytes"] += facts.size
            else:
                totals["estimated_characters"] += estimate_characters(path, facts.size)
        else:
            totals["unsupported"] += 1
            if "大小上限" in reason:
                totals["too_large"] += 1
        return {
            "path": path, "kind": "file", "ok": supported, "supported": supported, "error": reason or None,
            "image": image, "size": facts.size, "already": bool(existing),
            "file_id": existing["file_id"] if existing else None,
        }

    def _estimate(self, totals: dict[str, Any], figures_mode: str) -> dict[str, Any]:
        documents = max(0, totals["indexable"] - totals["images"])
        chunks = math.ceil(totals["estimated_characters"] / CHARACTERS_PER_CHUNK)
        requests = math.ceil(chunks / 32) + documents  # children in batches of 32, plus one file entry each
        images = totals["images"]
        by_mode = {}
        for mode in IMAGE_MODES:
            seconds = estimate_seconds(mode, images)
            by_mode[mode] = {"seconds": round(seconds), "cloud_calls": images if mode == "cloud" else 0}
        return {
            "documents": documents,
            "chunks": chunks,
            "embedding_requests": requests + 2 * images,
            "text_seconds": round(
                documents * SECONDS_PER_FILE
                + totals["estimated_characters"] / CHARACTERS_PER_SECOND
                + requests * SECONDS_PER_EMBEDDING_REQUEST
            ),
            "images": by_mode,
            # Figures inside PDFs follow the separate "图片理解" setting.
            "pdf_figures": totals["estimated_figures"],
            "pdf_figure_mode": figures_mode,
            "pdf_figure_cloud_calls": totals["estimated_figures"] if figures_mode == "cloud" else 0,
        }

    def import_paths(self, paths: Iterable[str], image_mode: str | None = None) -> dict[str, Any]:
        """Add files (registered in place) and folders (watched) the user picked or dropped.
        ``image_mode`` is how their images are read; every path gets its own result."""
        if image_mode is not None and image_mode not in IMAGE_MODES:
            raise LibraryError("未知的图片识别方式。")
        results: list[dict[str, Any]] = []
        for raw in paths:
            path = _clean_path(raw)
            try:
                path = _absolute(raw)
                watched = self._watched_root(path)
                if watched is not None:
                    # Added again (e.g. dropped twice): look for changes instead of failing.
                    results.append({"path": path, "kind": "folder", "ok": True, "already": True,
                                    "root": self.rescan_root(watched["root_id"])})
                elif os.path.isdir(path):
                    root = self.add_root(path, image_mode=image_mode)
                    results.append({"path": path, "kind": "folder", "ok": True, "already": False, "root": root})
                elif os.path.isfile(path):
                    facts = file_facts(path)
                    supported, reason = classify(path, facts.size, self.extensions, self.max_bytes)
                    if not supported:
                        raise LibraryError(reason)
                    already = self.registry.get_by_path(path) is not None
                    record = self.import_path(path, image_mode=image_mode, source="user")
                    results.append({"path": path, "kind": "file", "ok": True, "already": already,
                                    "file": self.file_view(record)})
                else:
                    raise LibraryError(f"文件或文件夹不存在：{path}")
            except LibraryError as exc:
                results.append({"path": path, "kind": "error", "ok": False, "error": str(exc)})
        return {"results": results, "added": sum(1 for item in results if item["ok"] and not item.get("already"))}

    def import_upload(self, staged: Path, filename: str, image_mode: str | None = None) -> dict[str, Any]:
        """Register a browser upload. The browser has no file path, so this one is copied."""
        # The copy's folder is the app's own: only the name counts for the rules.
        decision = self._decision(filename, single=True)
        if decision.excluded:
            Path(staged).unlink(missing_ok=True)
            raise LibraryError(f"{filename} 在排除范围里（{decision.rule['value']}），没有添加。")
        file_id = new_id()
        destination = Path(self.settings.upload_dir) / file_id / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), destination)
        facts = file_facts(destination)
        supported, reason = classify(str(destination), facts.size, self.extensions, self.max_bytes)
        with self._mutex:
            record = self.registry.insert_file(
                str(destination),
                root_id=None,
                origin="upload",
                file_id=file_id,
                status="pending" if supported else "unsupported",
                error=reason or None,
                image_mode=self._image_mode_for(str(destination), None, image_mode),
                content_mode="name" if decision.name_only else None,
                **self._facts_fields(facts),
            )
        if supported:
            self._note("new", record, source="user")
            self.enqueue(file_id, urgent=True)
        return record

    def import_path(self, raw: str, image_mode: str | None = None, source: str = "user") -> dict[str, Any]:
        """Register an existing local file in place. It is findable by name at once; its content
        is read by the background queue. Importing a known file again re-reads it (with the new
        ``image_mode`` if one is given)."""
        path = normalize_path(raw)
        if not os.path.isfile(path):
            raise LibraryError(f"文件不存在：{path}")
        facts = file_facts(path)
        supported, reason = classify(path, facts.size, self.extensions, self.max_bytes)
        with self._mutex:
            existing = self.registry.get_by_path(path)
            if existing:
                if image_mode in IMAGE_MODES and is_image(path) and existing.get("image_mode") != image_mode:
                    return self.set_image_mode(existing["file_id"], image_mode)
                self.enqueue(existing["file_id"], urgent=source == "user")
                return existing
            root = self.registry.root_for_path(path)
            decision = self._decision(path, root, single=root is None)
            if decision.excluded:
                raise LibraryError(
                    f"这个文件在排除范围里（{decision.rule['value']}），没有添加。要加进来，请先取消这条排除。"
                )
            record = self.registry.insert_file(
                path,
                root_id=root["root_id"] if root else None,
                origin="watch" if root else "import",
                status="pending" if supported else "unsupported",
                error=reason or None,
                image_mode=self._image_mode_for(path, root, image_mode),
                content_mode="name" if decision.name_only else None,
                **self._facts_fields(facts),
            )
        if supported:
            self._note("new", record, source=source)
            self.enqueue(record["file_id"], urgent=source == "user")
        return record

    def set_image_mode(self, file_id: str, mode: str) -> dict[str, Any]:
        """Change how one file's images are read and read it again: an image file
        (local / cloud / name), or a PDF's scanned pages (local / cloud, "off" = no thanks)."""
        with self._mutex:
            record = self.registry.get_file(file_id)
            if record is None:
                raise KeyError(file_id)
            suffix = Path(record["path"]).suffix.lower()
            allowed = IMAGE_MODES if is_image(record["path"]) else PDF_IMAGE_MODES if suffix == ".pdf" else ()
            if mode not in allowed:
                raise LibraryError("这个文件不能这样识别。")
            previous, notice = self._version(record), record.get("notice")
            self.registry.update_file(file_id, image_mode=mode, notice=None)
            record = self.registry.get_file(file_id)
            if record.get("content_mode") == "name":
                return record  # kept by name only: its pictures are not read either way
            unchanged = self._version(record) == previous and (mode == "off" or not notice)
            if record["status"] in LOST_STATUSES or unchanged and record["status"] == "indexed":
                return record
        return self.reindex(file_id)

    def _image_mode_for(self, path: str, root: dict[str, Any] | None, chosen: str | None = None) -> str | None:
        """The image mode a new file is registered with: the user's choice for this import, the
        folder's, the remembered preference, else local OCR (nothing is uploaded)."""
        if not is_image(path):
            return None
        for mode in (chosen, (root or {}).get("image_mode"), getattr(self.settings, "image_import_mode", "ask")):
            if mode in IMAGE_MODES:
                return mode
        return DEFAULT_IMAGE_MODE

    # ======================================================================
    # what may be read: name-only files and exclusions (library/privacy.py)
    # ======================================================================

    def _rules(self) -> list[dict[str, Any]]:
        rules = self._rules_cache
        if rules is None:
            rules = self._rules_cache = self.registry.list_rules()
        return rules

    def _rules_changed(self) -> None:
        self._rules_cache = None

    def _decision(
        self, path: str, root: dict[str, Any] | None = None, *, single: bool = False,
        rules: list[dict[str, Any]] | None = None,
    ) -> privacy.Decision:
        """What the rules (the saved ones, or ``rules``) say about ``path``: inside ``root``
        (looked up when not given), or, with ``single``, as a file outside every watched folder
        (only name patterns apply)."""
        rules = self._rules() if rules is None else rules
        if not rules:
            return privacy.FULL
        if root is None and not single:
            root = self.registry.root_for_path(path)
        inner = privacy.relative_parts(path, root["path"]) if root is not None else None
        if inner is None:
            patterns = [rule for rule in rules if rule["kind"] == "pattern" and rule["root_id"] is None]
            return privacy.decide(None, (), patterns, outer_parts=privacy.split_parts(path))
        path_rules = [rule for rule in rules if rule["kind"] == "path" and rule["root_id"] == root["root_id"]]
        patterns = [rule for rule in rules if rule["kind"] == "pattern" and rule["root_id"] in (None, root["root_id"])]
        return privacy.decide(inner, path_rules, patterns, outer_parts=privacy.split_parts(root["path"]))

    def _skip_for(self, root: dict[str, Any]) -> Callable[[str, bool], bool] | None:
        """What a walk of ``root`` leaves out: whatever the user excluded."""
        if not self._rules():
            return None
        return lambda path, _folder: self._decision(path, root).excluded

    def _switch_to_name(self, record: dict[str, Any]) -> None:
        """Keep a file by name only from now on: stop reading it and drop its text everywhere."""
        file_id = record["file_id"]
        with self._wake:
            self._queue.discard(file_id)
        self._forced.discard(file_id)
        self._retries.pop(file_id, None)
        self._stop_reading(file_id)
        fields: dict[str, Any] = {
            "content_mode": "name", "chunks": 0, "characters": None, "indexed_sha256": None,
            "parser_version": None, "notice": None, "error": None,
        }
        if record["status"] not in (*LOST_STATUSES, "unsupported"):
            fields["status"] = "named"
        self.registry.update_file(file_id, **fields)
        self._drop_index(file_id, record=record, purge=True)

    def _read_again(self, file_id: str) -> None:
        """A file the user wants read in full again (it was kept by name only)."""
        self.registry.update_file(file_id, content_mode=None)
        record = self.registry.get_file(file_id)
        if record and record["status"] not in (*LOST_STATUSES, "unsupported"):
            self.registry.update_file(file_id, status="pending", error=None)
            self.enqueue(file_id)

    def set_content_mode(self, file_id: str, mode: str, dry_run: bool = False) -> dict[str, Any]:
        """Keep one file by name only ("name") or read it in full again ("full").

        Returns what changes: ``changed`` (0/1) and ``indexed`` (1 when text of it is in the
        index now and would be dropped); ``dry_run`` only reports, for the confirmation."""
        if mode not in ("name", "full"):
            raise LibraryError("未知的收录方式。")
        with self._mutex:
            record = self.registry.get_file(file_id)
            if record is None:
                raise KeyError(file_id)
            current = "name" if record.get("content_mode") == "name" else "full"
            impact = {
                "files": 1,
                "changed": int(current != mode),
                "indexed": int(current != mode and mode == "name" and bool(record.get("chunks") or record.get("indexed_sha256"))),
            }
            if not dry_run and current != mode:
                if mode == "name":
                    self._switch_to_name(record)
                else:
                    self._read_again(file_id)
                record = self.registry.get_file(file_id) or record
        if not dry_run and mode == "full" and record["status"] == "pending":
            self.enqueue(file_id, urgent=True)  # the user is waiting for it
        return {**impact, "file": self.file_view(record)}

    def _folder_target(self, raw: str) -> tuple[str, dict[str, Any], list[str]]:
        path = _absolute(raw)
        root = self.registry.root_for_path(path)
        parts = privacy.relative_parts(path, root["path"]) if root else None
        if root is None or parts is None:
            raise LibraryError("这个文件夹不在知识库的文件夹里。")
        return path, root, parts

    def set_folder_mode(self, raw: str, mode: str, dry_run: bool = False) -> dict[str, Any]:
        """Keep a watched folder or one of its sub-folders by name only ("name"), or read it in
        full ("full"). The choice covers everything inside, files added later included; deeper
        choices made before are replaced by it."""
        if mode not in ("name", "full"):
            raise LibraryError("未知的收录方式。")
        path, root, parts = self._folder_target(raw)
        value = os.path.join(*parts) if parts else ""
        with self._mutex:
            files = self.registry.files_under(path)
            changed = [record for record in files if (record.get("content_mode") == "name") != (mode == "name")]
            impact = {
                "files": len(files),
                "changed": len(changed),
                "indexed": sum(1 for record in changed if mode == "name" and (record.get("chunks") or record.get("indexed_sha256"))),
            }
            if dry_run:
                return impact
            key = [os.path.normcase(part) for part in parts]
            for rule in self._rules():
                rule_key = [os.path.normcase(part) for part in privacy.split_parts(rule["value"])]
                if (
                    rule["root_id"] == root["root_id"] and rule["kind"] == "path" and rule["action"] in ("name", "full")
                    and rule_key[: len(key)] == key
                ):
                    self.registry.soft_delete_rule(rule["rule_id"])
            self._rules_changed()
            if self._decision(path, root).mode != mode:
                try:
                    inode = int(os.stat(path).st_ino) or None
                except OSError:
                    inode = None
                self.registry.add_rule(root_id=root["root_id"], kind="path", value=value, action=mode, inode=inode)
                self._rules_changed()
            for record in changed:
                if mode == "name":
                    self._switch_to_name(record)
                else:
                    self._read_again(record["file_id"])
        return impact

    def add_rule(
        self, *, action: str, path: str = "", pattern: str = "", root_id: str | None = None, dry_run: bool = False
    ) -> dict[str, Any]:
        """Exclude a folder or file inside a watched folder (``path``), or exclude / keep by name
        only every file or folder with this name (``pattern``, in folder ``root_id`` or, None,
        in all of them). Files now excluded leave the library; their text is purged."""
        if pattern:
            if action not in privacy.PATTERN_ACTIONS:
                raise LibraryError("未知的规则。")
            try:
                value = privacy.clean_pattern(pattern)
            except ValueError as exc:
                raise LibraryError(str(exc)) from exc
            if root_id is not None and self.registry.get_root(root_id) is None:
                raise KeyError(root_id)
            rule = {"kind": "pattern", "value": value, "action": action, "root_id": root_id}
            if any(
                other["kind"] == "pattern" and other["root_id"] == root_id and other["action"] == action
                and os.path.normcase(other["value"]) == os.path.normcase(value)
                for other in self._rules()
            ):
                raise LibraryError("已经有这条规则了。")
            candidates = self.registry.list_files(root_id=root_id) if root_id else self.registry.list_files()
        else:
            if action != "exclude":
                raise LibraryError("文件夹请用“只按文件名收录”。")
            target, root, parts = self._folder_target(path)
            if not parts:
                raise LibraryError("要排除整个文件夹，直接从知识库移除它就行。")
            try:
                inode = int(os.stat(target).st_ino) or None
            except OSError:
                inode = None
            rule = {"kind": "path", "value": os.path.join(*parts), "action": "exclude", "root_id": root["root_id"], "inode": inode}
            single = self.registry.get_by_path(target)
            candidates = [single] if single else self.registry.files_under(target)
        affected = [record for record in candidates if self._decision_with(record, rule)]
        impact = {
            "files": len(affected),
            "indexed": sum(1 for record in affected if record.get("chunks") or record.get("indexed_sha256")),
        }
        if dry_run:
            return impact
        with self._mutex:
            row = self.registry.add_rule(
                root_id=rule["root_id"], kind=rule["kind"], value=rule["value"], action=rule["action"], inode=rule.get("inode"),
            )
            self._rules_changed()
            for record in affected:
                current = self.registry.get_file(record["file_id"])
                if current is None:
                    continue
                if action == "exclude":
                    self.remove_file(current["file_id"], source=None)
                elif current.get("content_mode") != "name":
                    self._switch_to_name(current)
        root = self.registry.get_root(row["root_id"]) if row["root_id"] else None
        return {**impact, "rule": privacy.describe_rule(row, root["path"] if root else None)}

    def _decision_with(self, record: dict[str, Any], rule: dict[str, Any]) -> bool:
        """Would ``rule`` (not saved yet) change this file: exclude it, or keep it by name only?"""
        root = self.registry.get_root(record["root_id"]) if record.get("root_id") else None
        trial = {**rule, "rule_id": "new"}
        if root is None:
            if rule["kind"] != "pattern" or rule["root_id"] is not None:
                return False
            decision = privacy.decide(None, (), [trial], outer_parts=privacy.split_parts(record["path"]))
        else:
            inner = privacy.relative_parts(record["path"], root["path"])
            if inner is None:
                return False
            decision = privacy.decide(
                inner, [trial] if rule["kind"] == "path" else (), [trial] if rule["kind"] == "pattern" else (),
                outer_parts=privacy.split_parts(root["path"]),
            )
        if rule["action"] == "exclude":
            return decision.excluded
        return decision.name_only and record.get("content_mode") != "name"

    def remove_rule(self, rule_id: str, dry_run: bool = False) -> dict[str, Any]:
        """Take a rule back. A folder kept by name only goes back to what its parent folder
        does; files no longer excluded or kept by name are added and read (``files`` says how
        many would be read in full: the confirmation says they will be uploaded)."""
        with self._mutex:
            rule = self.registry.get_rule(rule_id)
            if rule is None:
                raise KeyError(rule_id)
            root = self.registry.get_root(rule["root_id"]) if rule["root_id"] else None
            if rule["kind"] == "path" and rule["action"] in ("name", "full"):
                if root is None:
                    raise KeyError(rule_id)
                folder = os.path.join(root["path"], rule["value"]) if rule["value"] else root["path"]
                others = [item for item in self._rules() if item["rule_id"] != rule_id]
                inherited = privacy.decide(
                    privacy.split_parts(rule["value"]),
                    [item for item in others if item["kind"] == "path" and item["root_id"] == root["root_id"]],
                    [item for item in others if item["kind"] == "pattern" and item["root_id"] in (None, root["root_id"])],
                    outer_parts=privacy.split_parts(root["path"]),
                )
                if inherited.excluded:
                    inherited = privacy.FULL
                return self.set_folder_mode(folder, inherited.mode, dry_run=dry_run)
            others = [item for item in self._rules() if item["rule_id"] != rule_id]
            if rule["action"] == "name":
                # A name pattern: its files are read again, unless another rule keeps them by name.
                to_read = [
                    record for record in self.registry.list_files()
                    if record.get("content_mode") == "name" and self._matches(record, rule)
                    and (rule["root_id"] is None or record.get("root_id") == rule["root_id"])
                    and not self._decision(record["path"], rules=others).name_only
                ]
                impact = {"files": len(to_read), "changed": len(to_read), "indexed": 0}
                if not dry_run:
                    self.registry.soft_delete_rule(rule_id)
                    self._rules_changed()
                    for record in to_read:
                        self._read_again(record["file_id"])
                return impact
        # An exclusion: what comes back is found by walking the folders it covered.
        roots = [root] if root else self.registry.list_roots()
        back = 0
        for item in roots:
            if item["status"] != "active":
                continue
            target = item["path"]
            if rule["kind"] == "path":
                target = os.path.join(item["path"], rule["value"])
                if os.path.isfile(target):
                    back += 1
                    continue
            if not os.path.isdir(target):
                continue
            skip = lambda path, _folder, where=item: self._decision(path, where, rules=others).excluded  # noqa: E731
            for entry in walk(target, self.extensions, self.max_bytes, max_files=self.max_files, skip=skip):
                if entry.supported and self._decision(entry.facts.path, item).excluded:
                    back += 1
        impact = {"files": back, "changed": back, "indexed": 0}
        if dry_run:
            return impact
        with self._mutex:
            self.registry.soft_delete_rule(rule_id)
            self._rules_changed()
        for item in roots:
            if item["status"] == "active":
                self.rescan_root(item["root_id"])
        return impact

    def _matches(self, record: dict[str, Any], rule: dict[str, Any]) -> bool:
        """Does a pattern rule name this file or one of its folders?"""
        names = privacy.split_parts(record["path"])
        return any(privacy.pattern_matches(rule["value"], name) for name in names)

    def privacy_summary(self) -> dict[str, Any]:
        """For the settings' "数据去向": every rule, and how many files are kept by name only."""
        roots = {root["root_id"]: root for root in self.registry.list_roots()}
        rules = []
        for rule in self._rules():
            root = roots.get(rule["root_id"]) if rule["root_id"] else None
            if rule["root_id"] and root is None:
                continue
            rules.append(privacy.describe_rule(rule, root["path"] if root else None))
        return {"rules": rules, "named_files": self.registry.count_name_only()}

    def _follow_rules(self, root: dict[str, Any]) -> int:
        """Path rules whose folder or file was renamed or moved while nobody watched: found
        again by NTFS file ID, so an excluded folder stays excluded under its new name."""
        moved = 0
        for rule in self._rules():
            if rule["root_id"] != root["root_id"] or rule["kind"] != "path" or not rule["value"] or not rule.get("inode"):
                continue
            path = os.path.join(root["path"], rule["value"])
            if os.path.exists(path):
                continue
            found = resolve_file_id(path, rule["inode"])
            if not _located(found) or winfs.in_recycle_bin(found):
                continue
            with self._mutex:
                self._move_rules(path, normalize_path(found))
            moved += 1
        return moved

    def _move_rules(self, src: str, dest: str) -> int:
        """Re-point path rules on ``src`` or inside it at ``dest`` (a rename or move)."""
        if not any(rule["kind"] == "path" for rule in self._rules()):
            return 0
        source_root = self.registry.root_for_path(src)
        if source_root is None:
            return 0
        source_parts = privacy.relative_parts(src, source_root["path"])
        if not source_parts:
            return 0
        source_key = [os.path.normcase(part) for part in source_parts]
        target_root = self.registry.root_for_path(dest)
        target_parts = privacy.relative_parts(dest, target_root["path"]) if target_root else None
        moved = 0
        for rule in self._rules():
            if rule["root_id"] != source_root["root_id"] or rule["kind"] != "path":
                continue
            parts = privacy.split_parts(rule["value"])
            if [os.path.normcase(part) for part in parts[: len(source_key)]] != source_key:
                continue
            if target_root is None or not target_parts:
                self.registry.soft_delete_rule(rule["rule_id"])  # left every watched folder
            else:
                value = os.path.join(*target_parts, *parts[len(source_key):])
                self.registry.update_rule(rule["rule_id"], root_id=target_root["root_id"], value=value)
            moved += 1
        if moved:
            self._rules_changed()
        return moved

    # ======================================================================
    # single files
    # ======================================================================

    def remove_file(self, file_id: str, source: str | None = "user") -> None:
        """Tombstone the record and drop its index; the original file is never touched. Its text
        also leaves the database's free pages and the backups (``on_purged``)."""
        with self._mutex:
            record = self.registry.get_file(file_id)
            if record is None:
                return
            self.registry.soft_delete_file(file_id)
            self._queue.discard(file_id)
            self._forced.discard(file_id)
            self._stop_reading(file_id)
        self._drop_index(file_id, record=record, purge=True)
        if record["origin"] == "upload":
            shutil.rmtree(Path(self.settings.upload_dir) / file_id, ignore_errors=True)
        if source:
            self._note("removed", record, source=source)

    def relocate(self, file_id: str, raw: str) -> dict[str, Any]:
        """User-supplied new location for a missing file."""
        record = self.registry.get_file(file_id)
        if record is None:
            raise KeyError(file_id)
        text = _clean_path(raw)
        if not text or not os.path.isabs(text):
            raise LibraryError("请输入文件的完整路径。")
        path = normalize_path(text)
        if not os.path.isfile(path):
            raise LibraryError(f"文件不存在：{path}")
        if winfs.in_recycle_bin(path):
            raise LibraryError("这个文件在回收站里，请先还原。")
        with self._mutex:
            other = self.registry.get_by_path(path)
            if other and other["file_id"] != file_id:
                raise LibraryError("这个路径已经对应知识库中的另一个文件。")
            facts = file_facts(path)
            self._relocate(record, facts, sha256=sha256_file(path))
            return self.registry.get_file(file_id)

    def reindex(self, file_id: str) -> dict[str, Any]:
        """Parse and embed a file again although its content did not change (asked by the user).
        A file whose original is gone, offline or in the Recycle Bin has nothing to read."""
        with self._mutex:
            record = self.registry.get_file(file_id)
            if record is None:
                raise KeyError(file_id)
            if record["status"] in LOST_STATUSES:
                return record
            self._forced.add(file_id)
            self.registry.update_file(file_id, status="pending", error=None)
        self.enqueue(file_id, urgent=True)
        return self.registry.get_file(file_id)

    def reveal(self, file_id: str) -> str:
        """Open the system file manager at the file (or its folder when the file is gone)."""
        record = self.registry.get_file(file_id)
        if record is None:
            raise KeyError(file_id)
        path = record["path"]
        target = path if os.path.isfile(path) else os.path.dirname(path)
        if not os.path.isdir(target) and not os.path.isfile(target):
            raise FileNotFoundError(path)
        if sys.platform == "win32":
            if target == path:
                # explorer parses "/select," itself; Windows paths cannot contain quotes.
                subprocess.Popen(f'explorer /select,"{path}"')
            else:
                subprocess.Popen(["explorer", target])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", target] if target == path else ["open", target])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(target) if target == path else target])
        return target

    def describe_files(self, doc_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Status and current location for citation doc ids."""
        result: dict[str, dict[str, Any]] = {}
        for doc_id in set(doc_ids):
            record = self.registry.get_file(doc_id)
            if record:
                result[doc_id] = {"status": record["status"], "path": record["path"], "name": record["name"]}
        return result

    def file_view(self, record: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "file_id", "root_id", "origin", "path", "name", "size", "mtime", "status", "error",
            "chunks", "characters", "created_at", "updated_at", "image_mode", "notice",
        )
        view = {key: record.get(key) for key in keys}
        view["content_mode"] = "name" if record.get("content_mode") == "name" else "full"
        view["doc_id"] = record["file_id"]
        view["source"] = record["name"]
        view["searchable"] = record["status"] in SEARCHABLE_STATUSES or (
            bool(record.get("indexed_sha256")) and record["status"] != "recycled"
        )
        view["image"] = is_image(record["path"])
        return view

    def lost_files(self, limit: int = 500) -> list[dict[str, Any]]:
        """The "找不到的文件" list: missing, offline and in the Recycle Bin."""
        return [self.file_view(record) for record in self.registry.list_files(statuses=LOST_STATUSES, limit=limit)]

    def hidden_ids(self) -> frozenset[str]:
        """Files left out of answers although their index is kept (in the Recycle Bin); cached briefly."""
        now = time.monotonic()
        if self._hidden is None or now - self._hidden[0] > 5:
            ids = frozenset(record["file_id"] for record in self.registry.list_files(statuses=("recycled",)))
            self._hidden = (now, ids)
        return self._hidden[1]

    # ======================================================================
    # background indexing queue
    # ======================================================================

    def enqueue(self, file_id: str, urgent: bool = False) -> None:
        """Queue a file for indexing. ``urgent``: the user is waiting on it (just added it,
        or opened it), so it goes before everything else."""
        with self._wake:
            if file_id in self._queue and not urgent:
                return
        record = self.registry.get_file(file_id)
        cost, kind = self._job_estimate(record) if record else (0.0, LIGHT)
        with self._wake:
            if not self._queue and not self._running:
                self._batch = {"done": 0, "failed": 0, "skipped": 0}
            if self._queue.add(file_id, cost, kind, urgent):
                self._wake.notify_all()

    def _job_estimate(self, record: dict[str, Any]) -> tuple[float, str]:
        try:
            return estimate_job(record, pdf_image_mode(self.settings, record.get("image_mode")))
        except Exception:
            return 1.0, LIGHT

    def status(self, since: int | None = None) -> dict[str, Any]:
        with self._wake:
            queued = len(self._queue)
            running = [dict(job) for job in self._running.values()]
            batch = dict(self._batch)
        finished = batch["done"] + batch["failed"] + batch["skipped"]
        total = finished + queued + len(running)
        counts = self.registry.status_counts()
        running.sort(key=lambda job: job.get("started", 0))
        payload = {
            "busy": bool(queued or running or self._scanning),
            "queued": queued,
            "current": running[0] if running else None,
            "running": len(running),
            # Every file being read now; "pages" = [read, to read] for scanned PDFs.
            "jobs": running,
            "paused": bool(self._answering and (queued or running)),
            "batch": {**batch, "total": total, "finished": finished},
            "progress": round(finished / total, 3) if total else 1.0,
            "scanning": sorted(self._scanning),
            "counts": counts,
            "lost": {status: counts.get(status, 0) for status in LOST_STATUSES},
            "device_id": self.registry.device_id,
        }
        if since is not None:
            payload["changes"] = self.changes(since)
        return payload

    def process_next(self) -> str | None:
        """Index the next queued file synchronously; returns the outcome (tests/CLI)."""
        with self._wake:
            job = self._queue.pop()
        return None if job is None else self._run_job(job.file_id, job.kind)

    def run_until_idle(self) -> list[str]:
        outcomes = []
        while (outcome := self.process_next()) is not None:
            outcomes.append(outcome)
        return outcomes

    def _worker_loop(self) -> None:
        """One of several workers: take the best job that may start (slow recognition jobs are
        capped, see jobs.Slots), never a file another worker is indexing. Runs below normal
        priority: a question being answered gets the processor first."""
        lower_current_thread(BELOW_NORMAL)
        while not self._stopping.is_set():
            self.pause_for_questions()
            with self._wake:
                job = None
                while not self._stopping.is_set():
                    running = Counter(entry["kind"] for entry in self._running.values())
                    job = self._queue.pop(
                        lambda candidate: candidate.file_id not in self._running
                        and self._slots.allows(candidate.kind, running)
                    )
                    if job is not None:
                        # Reserved here, under the lock, so two workers cannot both take a slow slot.
                        self._running[job.file_id] = {"file_id": job.file_id, "kind": job.kind, "started": time.time()}
                        break
                    self._wake.wait(timeout=5)
                    if not self._queue:
                        self._requeue_due()
                if job is None:
                    return
            try:
                self._run_job(job.file_id, job.kind)
            except Exception:
                logger.exception("Index job crashed for %s", job.file_id)
                with self._wake:
                    self._running.pop(job.file_id, None)
            finally:
                with self._wake:
                    self._wake.notify_all()

    def _run_job(self, file_id: str, kind: str = LIGHT) -> str:
        record = self.registry.get_file(file_id)
        if record is None:
            with self._wake:
                self._running.pop(file_id, None)
            return "gone"
        with self._wake:
            entry = self._running.setdefault(file_id, {"file_id": file_id, "kind": kind, "started": time.time()})
            entry.update(name=record["name"], path=record["path"], pages=None)
        try:
            outcome = self._index(record)
        finally:
            with self._wake:
                self._running.pop(file_id, None)
        key = {"indexed": "done", "failed": "failed"}.get(outcome, "skipped")
        with self._wake:
            self._batch[key] += 1
        return outcome

    # ---------------- yielding to questions ----------------

    @contextmanager
    def answering(self) -> Iterator[None]:
        """Held while a question is answered: the queue waits, so answers keep their speed
        (model rate limits, the store's lock) while thousands of files are being indexed."""
        with self._answer_cond:
            self._answering += 1
        try:
            yield
        finally:
            with self._answer_cond:
                self._answering -= 1
                self._answer_cond.notify_all()

    def pause_for_questions(self) -> None:
        """Wait (at most ``index_yield_max_seconds``) while a question is being answered.
        Called between files and between a file's embedding requests."""
        limit = float(getattr(self.settings, "index_yield_max_seconds", 120.0))
        deadline = time.monotonic() + limit
        with self._answer_cond:
            while self._answering and not self._stopping.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._answer_cond.wait(min(remaining, 1.0))

    # ---------------- one job ----------------

    def _index(self, record: dict[str, Any]) -> str:
        file_id = record["file_id"]
        path = record["path"]
        with self._mutex:
            forced = file_id in self._forced
            self._forced.discard(file_id)
            self._cancelled.discard(file_id)
            record = self.registry.get_file(file_id)
            if record is None:
                return "gone"
            path = record["path"]
            if record["status"] == "recycled":
                return "skipped"  # never read from the Recycle Bin
            if not os.path.isfile(path):
                outcome = self._mark_lost(record)
                return outcome if outcome in ("moved", "recycled", "removed") else "missing"
            try:
                facts = file_facts(path)
            except OSError as exc:
                self.registry.update_file(file_id, status="failed", error=f"无法读取文件：{exc}")
                return "failed"
            supported, reason = classify(path, facts.size, self.extensions, self.max_bytes)
            if not supported:
                self.registry.update_file(file_id, status="unsupported", error=reason, **self._facts_fields(facts))
                return "unsupported"
            if record.get("content_mode") == "name":
                # The one place a file's content would be read: a file kept by name only is
                # never opened, hashed or parsed, and nothing about it is sent anywhere.
                return self._keep_name_only(record, facts)
            if is_image(path) and not record.get("image_mode"):
                record["image_mode"] = self._image_mode_for(path, self.registry.get_root(record["root_id"] or ""))
                self.registry.update_file(file_id, image_mode=record["image_mode"])
        try:
            digest = sha256_file(path)
        except OSError as exc:
            self.registry.update_file(file_id, status="failed", error=f"无法读取文件：{exc}")
            return "failed"
        if (
            not forced
            and digest == record.get("indexed_sha256")
            and self.store.has_document(file_id)
            and not self._parser_outdated(record)
        ):
            # Content unchanged (touched, re-saved, restored from missing): keep the vectors.
            self.registry.update_file(file_id, status="indexed", error=None, sha256=digest, **self._facts_fields(facts))
            return "skipped"
        if self.indexer is None:
            raise RuntimeError("no indexer configured")
        self.registry.update_file(file_id, status="indexing", error=None, sha256=digest, **self._facts_fields(facts))
        try:
            result = self._call_indexer(record)
        except Exception as exc:
            with self._mutex:
                current = self.registry.get_file(file_id)
                if current is None or current.get("content_mode") == "name" or isinstance(exc, ReadingStopped):
                    # Switched to name-only or removed while being read; a scan may have
                    # committed its first pages already.
                    return self._stopped(file_id, current)
                if current and not os.path.isfile(current["path"]):
                    self._mark_lost(current)
                    return "missing"
                logger.warning("Indexing %s failed: %s", path, exc)
                fields: dict[str, Any] = {"status": "failed", "error": str(exc)[:500] or type(exc).__name__}
                if _account_refused(exc):
                    pass  # tried again at the next start, not every few minutes
                elif _transient(exc):
                    self._schedule_retry(file_id)
                elif not record.get("indexed_sha256"):
                    # Never indexed and the file itself is the problem: remember the parser
                    # that failed, so only an upgrade retries it. Timeouts and service errors are
                    # retried after RETRY_DELAYS, then at the next start; so is a file still
                    # searchable from an older parser.
                    fields["parser_version"] = self._version(record)
                self.registry.update_file(file_id, **fields)
            return "failed"
        chunks, characters = int(result[0]), result[1]
        info = result[2] if len(result) > 2 and isinstance(result[2], dict) else {}
        with self._mutex:
            current = self.registry.get_file(file_id)
            if current is None or current.get("content_mode") == "name" or file_id in self._cancelled:
                # Removed, or kept by name only, while we were embedding: no text stays behind.
                return self._stopped(file_id, current)
            if current["name"] != record["name"]:
                self._rename_index(file_id, current["name"])
            fields = {
                "indexed_sha256": digest,
                "chunks": chunks,
                "characters": characters,
                "error": None,
                "parser_version": self._version(record),
                "notice": info.get("notice"),
            }
            if current["status"] == "indexing":
                fields["status"] = "indexed"
            self.registry.update_file(file_id, **fields)
            self._retries.pop(file_id, None)
        self._notify(self.on_indexed, file_id)
        return "indexed"

    def _schedule_retry(self, file_id: str) -> None:
        attempts = self._retries.get(file_id, (0, 0.0))[0]
        if attempts < len(RETRY_DELAYS):
            self._retries[file_id] = (attempts + 1, time.monotonic() + RETRY_DELAYS[attempts])
        else:
            self._retries.pop(file_id, None)

    def _requeue_due(self) -> int:
        """Queue the transient failures whose retry time has come (called by the idle worker)."""
        now = time.monotonic()
        due = [file_id for file_id, (_, when) in self._retries.items() if when <= now]
        queued = 0
        for file_id in due:
            record = self.registry.get_file(file_id)
            if record is None or record["status"] != "failed":
                self._retries.pop(file_id, None)
                continue
            # Keep the attempt count; the next failure schedules the next, longer delay.
            self._retries[file_id] = (self._retries[file_id][0], float("inf"))
            self.enqueue(file_id)
            queued += 1
        return queued

    def _job_progress(self, file_id: str, done: int, total: int) -> None:
        with self._wake:
            if file_id in self._running:
                self._running[file_id]["pages"] = [int(done), int(total)]

    def _call_indexer(self, record: dict[str, Any]) -> tuple:
        file_id = record["file_id"]
        if record.get("content_mode") == "name":
            raise ReadingStopped(file_id)  # never: _index keeps these by name before getting here
        options = _indexer_options(self.indexer, {
            "image_mode": record.get("image_mode"),
            "pause": lambda: self._pause_or_stop(file_id),
            "progress": lambda done, total: self._job_progress(file_id, done, total),
        })
        return tuple(self.indexer(record["path"], file_id, record["name"], **options))

    def _pause_or_stop(self, file_id: str) -> None:
        """Between pages and embedding requests: wait while a question is answered, and stop
        reading a file that was switched to name-only or removed meanwhile."""
        if file_id in self._cancelled:
            raise ReadingStopped(file_id)
        self.pause_for_questions()
        if file_id in self._cancelled:
            raise ReadingStopped(file_id)

    def _stop_reading(self, file_id: str) -> None:
        with self._wake:
            if file_id in self._running:
                self._cancelled.add(file_id)

    def _stopped(self, file_id: str, current: dict[str, Any] | None) -> str:
        """The reading of a file stopped half-way (switched to name-only or removed): drop
        whatever it committed."""
        self._cancelled.discard(file_id)
        self._retries.pop(file_id, None)
        if current is None:
            self._drop_index(file_id, purge=True)
            return "gone"
        self._switch_to_name(current)
        return "named"

    def _keep_name_only(self, record: dict[str, Any], facts: FileFacts) -> str:
        """A file kept by name only came up in the queue: note its size and dates, nothing more."""
        had_text = bool(record.get("chunks") or record.get("indexed_sha256")) or self.store.has_document(record["file_id"])
        self.registry.update_file(
            record["file_id"], status="named", error=None, notice=None, chunks=0, characters=None,
            indexed_sha256=None, parser_version=None, **self._facts_fields(facts),
        )
        self._retries.pop(record["file_id"], None)
        if had_text:
            self._drop_index(record["file_id"], record=record, purge=True)
        return "named"

    # ======================================================================
    # watcher events (already debounced by the watcher)
    # ======================================================================

    def handle_path(self, path: str) -> None:
        """A path was created, modified or deleted: sync it with what is on disk now."""
        path = normalize_path(path)
        root = self.registry.root_for_path(path)
        if os.path.isdir(path):
            if root is None or self._decision(path, root).excluded:
                return
            # A folder moved in from outside the watched tree arrives as one "created" event.
            for entry in walk(path, self.extensions, self.max_bytes, max_files=self.max_files, skip=self._skip_for(root)):
                with self._mutex:
                    self._upsert_entry(entry.facts, entry.supported, entry.reason, root, None)
            return
        if os.path.isfile(path):
            if root is None or path_excluded(path, root["path"]) or self._decision(path, root).excluded:
                return
            try:
                facts = file_facts(path)
            except OSError:
                return
            supported, reason = classify(path, facts.size, self.extensions, self.max_bytes)
            with self._mutex:
                self._upsert_entry(facts, supported, reason, root, None)
            return
        with self._mutex:
            record = self.registry.get_by_path(path)
            records = [record] if record else self.registry.files_under(path)
            for item in records:
                item = self.registry.get_file(item["file_id"]) or item  # a sibling's follow may have moved it
                if item["status"] not in LOST_STATUSES and not os.path.exists(item["path"]):
                    self._mark_lost(item)

    def handle_move(self, src: str, dest: str) -> None:
        src, dest = normalize_path(src), normalize_path(dest)
        # Rules on the moved file or folder (or inside it) move with it first: an excluded
        # folder that is renamed stays excluded.
        with self._mutex:
            self._move_rules(src, dest)
        root = self.registry.root_for_path(dest)
        if root is not None and self._decision(dest, root).excluded:
            # The watcher already knows the destination. Purge immediately rather
            # than relying on NTFS file-ID lookup (unavailable on other hosts).
            with self._mutex:
                record = self.registry.get_by_path(src)
                records = [record] if record else self.registry.files_under(src)
            # Each removal takes the lock itself; purging a whole folder must not hold it
            # for the watcher and the queue the whole time.
            for item in records:
                self.remove_file(item["file_id"], source="auto")
            return
        if root is None or path_excluded(dest, root["path"]):
            self.handle_path(src)
            return
        with self._mutex:
            record = self.registry.get_by_path(src)
            if record is None:
                children = self.registry.files_under(src)
                if not children:
                    # e.g. an editor's temp file renamed over the real document.
                    self.handle_path(dest)
                    return
                for child in children:
                    new_path = dest + child["path"][len(src):]
                    if os.path.isfile(new_path):
                        self._relocate(child, file_facts(new_path))
                return
            if not os.path.isfile(dest):
                self.handle_path(src)
                return
            occupant = self.registry.get_by_path(dest)
            if occupant and occupant["file_id"] != record["file_id"]:
                # Renamed over another indexed file: that file's content is gone.
                self.remove_file(occupant["file_id"], source="auto")
            facts = file_facts(dest)
            supported, reason = classify(dest, facts.size, self.extensions, self.max_bytes)
            if not supported:
                self._relocate(record, facts)
                self.registry.update_file(record["file_id"], status="unsupported", error=reason)
                return
            was_unsupported = record["status"] == "unsupported"
            self._relocate(record, facts)
            if was_unsupported:
                self.registry.update_file(record["file_id"], status="pending", error=None)
                self.enqueue(record["file_id"])

    # ======================================================================
    # following files and folders (file IDs, Recycle Bin)
    # ======================================================================

    def _follow(self, record: dict[str, Any]) -> str | None:
        """The record's path is gone: look its file up by NTFS file ID.

        Returns "moved" (re-pointed, index kept), "recycled", "removed" (it was in the
        Recycle Bin and is now deleted for good) or None (not found: the caller marks it lost).
        """
        root = self.registry.get_root(record["root_id"]) if record.get("root_id") else None
        if root and root["status"] == "active" and not os.path.isdir(root["path"]):
            # The watched folder itself moved: re-point it (and every file in it) first.
            self.check_root(root)
            current = self.registry.get_file(record["file_id"])
            if current is None:
                return "removed"
            if os.path.exists(current["path"]):
                return "moved"
            if current["status"] == "recycled":
                return "recycled"
            record = current
        if not record.get("inode") or not volume_available(record["path"], record.get("volume_id")):
            return None
        found = resolve_file_id(record["path"], record["inode"])
        if found == winfs.UNSUPPORTED:
            return None
        if found == winfs.NOT_FOUND:
            if record["status"] == "recycled":
                # Deleted from the Recycle Bin: the user threw it away for good.
                self.remove_file(record["file_id"], source="auto")
                return "removed"
            return None
        if winfs.in_recycle_bin(found):
            if record["status"] != "recycled":
                self.registry.update_file(record["file_id"], status="recycled")
                self._queue.discard(record["file_id"])
                self._hidden = None
                self._note("recycled", record)
            return "recycled"
        if path_key(found) == record["path_key"] or not os.path.isfile(found):
            return None
        if Path(found).suffix.lower() != Path(record["path"]).suffix.lower():
            return None  # renamed to another type (a backup, an editor's temp file): treat as gone
        target_root = self.registry.root_for_path(found)
        if target_root and path_excluded(found, target_root["path"]):
            return None
        if self._decision(found, target_root, single=target_root is None).excluded:
            # Moved into a folder the user excluded: the library lets go of it.
            self.remove_file(record["file_id"], source="auto")
            return "removed"
        if not self._free_path(found):
            return None
        was = record["status"]
        self._relocate(record, file_facts(found), note=False)
        self._note("restored" if was == "recycled" else "moved", self.registry.get_file(record["file_id"]) or record,
                   old_path=record["path"])
        return "moved"

    def check_root(self, root: dict[str, Any], scan: bool = True) -> str:
        """Is a watched folder where the registry says? Follow it when it was renamed, moved,
        sent to the Recycle Bin or restored. Returns its status ("moved" when re-pointed)."""
        root = self.registry.get_root(root["root_id"]) or root
        path = root["path"]
        if not volume_available(path, root["volume_id"]):
            return self._root_lost(root, "offline")
        if os.path.isdir(path):
            try:
                inode = int(os.stat(path).st_ino) or None
            except OSError:
                inode = None
            if inode and root.get("inode") and inode != root["inode"]:
                # Another folder took the name; the watched one may have moved on.
                found = resolve_file_id(path, root["inode"])
                if _located(found) and not winfs.in_recycle_bin(found) and os.path.isdir(found):
                    return self._root_found(root, found, scan)
            if inode and inode != root.get("inode"):
                self.registry.update_root(root["root_id"], inode=inode)
            if root["status"] != "active":
                return self._root_back(root, scan)
            return "active"
        found = resolve_file_id(path, root.get("inode"))
        if _located(found) and (winfs.in_recycle_bin(found) or os.path.isdir(found)):
            return self._root_found(root, found, scan)
        if found == winfs.NOT_FOUND and root["status"] == "recycled":
            self.remove_root(root["root_id"], source="auto")
            return "removed"
        return self._root_lost(root, "missing")

    def check_root_by_id(self, root_id: str) -> str | None:
        """The parent folder watcher saw this watched folder renamed, moved or removed."""
        root = self.registry.get_root(root_id)
        return self.check_root(root) if root else None

    def _root_found(self, root: dict[str, Any], found: str, scan: bool) -> str:
        if winfs.in_recycle_bin(found):
            return self._root_lost(root, "recycled")
        if path_key(found) == root["path_key"]:
            return self._root_back(root, scan)
        return self._move_root(root, found, scan)

    def _move_root(self, root: dict[str, Any], new_path: str, scan: bool) -> str:
        """The folder was renamed or moved on its volume: re-point the folder and its files
        (no re-indexing), then watch it at its new place."""
        old_path = root["path"]
        new_path = normalize_path(new_path)
        if self.watcher:
            self.watcher.unwatch(root["root_id"])
            self.watcher.forget_parent(root["root_id"])
        with self._mutex:
            moved = self.registry.move_root(root["root_id"], new_path)
            container = next(
                (other for other in self.registry.list_roots()
                 if other["root_id"] != root["root_id"]
                 and path_key(new_path).startswith(other["path_key"].rstrip(os.sep) + os.sep)),
                None,
            )
            if container is not None:
                # Moved into another watched folder: its files (and rules) belong to that folder now.
                for record in self.registry.list_files(root_id=root["root_id"]):
                    self.registry.update_file(record["file_id"], root_id=container["root_id"])
                prefix = privacy.relative_parts(new_path, container["path"]) or []
                for rule in self._rules():
                    if rule["root_id"] != root["root_id"]:
                        continue
                    if rule["kind"] == "path":
                        self.registry.update_rule(
                            rule["rule_id"], root_id=container["root_id"],
                            value=os.path.join(*prefix, *privacy.split_parts(rule["value"])) if prefix else rule["value"],
                        )
                    else:
                        self.registry.update_rule(rule["rule_id"], root_id=container["root_id"])
                self._rules_changed()
                self.registry.soft_delete_root(root["root_id"])
            else:
                self.registry.update_root(root["root_id"], status="active")
            for record in self.registry.list_files(root_id=(container or root)["root_id"]):
                # Lost while the folder's old path was stale, and there again at the new path.
                if record["status"] in LOST_STATUSES and os.path.exists(record["path"]):
                    self.registry.update_file(record["file_id"], status="indexed" if record.get("indexed_sha256") else "pending")
                    if not record.get("indexed_sha256"):
                        self.enqueue(record["file_id"])
            self._hidden = None
        self._note("folder_moved", None, root_id=root["root_id"], path=new_path, old_path=old_path, files=moved)
        logger.info("Watched folder moved: %s -> %s (%d files)", old_path, new_path, moved)
        live = self.registry.get_root(root["root_id"])
        if live and self.watcher:
            self.watcher.watch(live)
        if scan:
            # Catch whatever changed between the move and now (stat only; unchanged files are skipped).
            self._spawn(self._scan_in_background, f"scan-{root['root_id'][:8]}", (container or root)["root_id"])
        return "moved"

    def _root_back(self, root: dict[str, Any], scan: bool) -> str:
        was = root["status"]
        self.registry.update_root(root["root_id"], status="active")
        live = self.registry.get_root(root["root_id"])
        with self._mutex:
            for record in self.registry.list_files(root_id=root["root_id"], statuses=("recycled",)):
                if os.path.exists(record["path"]):
                    self.registry.update_file(record["file_id"], status="indexed" if record.get("indexed_sha256") else "pending")
            self._hidden = None
        if was == "recycled":
            self._note("folder_restored", None, root_id=root["root_id"], path=root["path"])
        if self.watcher and live:
            self.watcher.watch(live)
        if scan:
            self.scan_root(root["root_id"])
        return "active"

    def _root_lost(self, root: dict[str, Any], status: str) -> str:
        if root["status"] == status:
            return status
        if self.watcher:
            self.watcher.unwatch(root["root_id"])
            self.watcher.watch_parent(root)
        with self._mutex:
            # Files first, then the folder: whoever sees the folder's new state sees its files' too.
            for record in self.registry.list_files(root_id=root["root_id"]):
                if os.path.exists(record["path"]) or record["status"] == status:
                    continue
                if status == "recycled":
                    self.registry.update_file(record["file_id"], status="recycled")
                    self._queue.discard(record["file_id"])
                else:
                    self._set_lost(record, note=False)
            self._hidden = None
            self.registry.update_root(root["root_id"], status=status)
        self._note(f"folder_{'missing' if status == 'offline' else status}", None, root_id=root["root_id"],
                   path=root["path"], status=status)
        return status

    def check_file(self, file_id: str) -> dict[str, Any]:
        """Check one file against the disk now (it is about to be shown or opened)."""
        with self._mutex:
            record = self.registry.get_file(file_id)
            if record is None:
                raise KeyError(file_id)
            self._check_record(record)
        if file_id in self._queue:
            self.enqueue(file_id, urgent=True)  # opened while waiting: read it next
        return self.registry.get_file(file_id) or {**record, "deleted_at": "now"}

    def sync(self, min_interval: float = 3.0) -> dict[str, Any]:
        """Check watched folders and single files now (the window regained focus).
        Calls closer together than ``min_interval`` seconds return the last result."""
        now = time.monotonic()
        if self._last_sync and now - self._last_sync[0] < min_interval:
            return {**self._last_sync[1], "cached": True}
        before = self._change_seq
        started = time.perf_counter()
        stats = self.sweep()
        result = {**stats, "elapsed_ms": round((time.perf_counter() - started) * 1000), "changes": self.changes(before)}
        self._last_sync = (now, result)
        return result

    def sweep(self) -> dict[str, int]:
        """Stat every single imported file and look lost files up again by file ID; check each
        watched folder is still where it was. Cheap: one stat (or one ID lookup) per file."""
        self._last_sweep = time.monotonic()
        stats: Counter = Counter()
        for root in self.registry.list_roots():
            try:
                stats[f"folders_{self.check_root(root)}"] += 1
            except Exception:
                logger.exception("Checking watched folder %s failed", root["path"])
        with self._mutex:
            records = self.registry.list_single_files() + self.registry.list_files(statuses=LOST_STATUSES)
            seen: set[str] = set()
            for record in records:
                if record["file_id"] in seen:
                    continue
                seen.add(record["file_id"])
                record = self.registry.get_file(record["file_id"])
                if record is None:
                    continue
                stats[self._check_record(record)] += 1
            waiting = self.registry.files_with_notice("ocr_unavailable")
            if waiting and ocr_runtime.activate(self.settings.data_dir):
                # Images read by name only because local OCR was not downloaded yet.
                for record in waiting:
                    if record["status"] == "indexed" and record["file_id"] not in self._queue:
                        self.reindex(record["file_id"])
                        stats["ocr_requeued"] += 1
        return dict(stats)

    def _check_record(self, record: dict[str, Any]) -> str:
        """One file against the disk: changed, moved, recycled, restored, lost or the same."""
        path = record["path"]
        if record["origin"] == "upload":
            return "same"
        if record["status"] in LOST_STATUSES:
            if os.path.isfile(path) and record["status"] != "recycled":
                return self._refresh_single(record, back=True)
            outcome = self._follow(record)
            if outcome is None and os.path.isfile(path):
                return self._refresh_single(record, back=True)  # restored to where it was
            if outcome is None and record["status"] != self._lost_status(record) and record["status"] != "recycled":
                self._set_lost(record)
                return record["status"]
            return outcome or "same"
        if not os.path.isfile(path):
            return self._mark_lost(record)
        if record.get("root_id"):
            return "same"  # watched: the folder watcher reports its changes
        return self._refresh_single(record)

    def _refresh_single(self, record: dict[str, Any], back: bool = False) -> str:
        try:
            facts = file_facts(record["path"])
        except OSError:
            return "same"
        changed = facts.size != record["size"] or abs(facts.mtime - (record["mtime"] or 0)) >= 1e-3
        if back:
            # Back where it was (restored, drive plugged in): re-check the content, keep the index if same.
            self.registry.update_file(record["file_id"], status="pending", error=None, **self._facts_fields(facts))
            self._hidden = None
            self._note("restored", record)
            self.enqueue(record["file_id"])
            return "restored"
        if changed or facts.inode != record["inode"]:
            # Edited (an editor's "save" may also make a new file at the same path).
            self.registry.update_file(record["file_id"], **self._facts_fields(facts))
            if record["status"] in ("indexed", "failed", "pending"):
                if record["status"] == "indexed" and changed:
                    self._note("changed", record)
                self.enqueue(record["file_id"])
            return "changed" if changed else "same"
        return "same"

    # ======================================================================
    # recent changes
    # ======================================================================

    def _note(self, kind: str, record: dict[str, Any] | None, source: str = "auto", **extra: Any) -> None:
        item = {"kind": kind, "source": source, "at": time.time(), **extra}
        if record is not None:
            item.update(file_id=record["file_id"], name=record.get("name"), path=extra.get("path") or record.get("path"))
        with self._changes_lock:
            self._change_seq += 1
            item["seq"] = self._change_seq
            self._changes.append(item)

    def changes(self, since: int = 0, limit: int = 30) -> dict[str, Any]:
        """What changed after ``since`` (a ``seq`` from an earlier answer): counts per kind for
        changes the app found by itself, and the latest items. The UI turns them into quiet toasts."""
        with self._changes_lock:
            items = [item for item in self._changes if item["seq"] > since]
            oldest = self._changes[0]["seq"] if self._changes else self._change_seq + 1
            seq = self._change_seq
        counts = Counter(item["kind"] for item in items if item["source"] == "auto")
        files = Counter()
        for item in items:
            if item["source"] == "user" and item["kind"] == "folder_added":
                files["folder_added"] += item.get("files", 0)
        return {
            "seq": seq,
            "since": since,
            "counts": {kind: counts.get(kind, 0) for kind in CHANGE_KINDS if counts.get(kind)},
            "user": dict(Counter(item["kind"] for item in items if item["source"] == "user")),
            "items": items[-limit:],
            "truncated": since + 1 < oldest and since < seq,
        }

    # ======================================================================
    # startup reconciliation
    # ======================================================================

    def reconcile(self) -> dict[str, int]:
        """Check every record against the disk after the app was not watching.

        Watched folders are followed by file ID first. Missing paths are then looked up by
        file ID (anywhere on their volume), then by size + sha256 inside the watched
        folders; files on an unmounted volume become ``offline``.
        """
        stats = {"lost": 0, "offline": 0, "queued": 0, "reparse": 0, "moved": 0, "recycled": 0}
        with self._mutex:
            for record in self.registry.list_files(statuses=("indexing",)):
                self.registry.update_file(record["file_id"], status="pending")
            for root in self.registry.list_roots():
                self.check_root(root, scan=False)
            for record in self.registry.list_files():
                record = self.registry.get_file(record["file_id"])
                if record is None or os.path.exists(record["path"]):
                    continue
                if record["status"] in LOST_STATUSES and record["status"] == self._lost_status(record) and not record.get("inode"):
                    continue
                outcome = self._mark_lost(record)
                if outcome in ("moved", "recycled"):
                    stats[outcome] += 1
                elif outcome in ("missing", "offline"):
                    stats["offline" if outcome == "offline" else "lost"] += 1
        for root in self.registry.list_roots():
            if root["status"] == "active":
                self.scan_root(root["root_id"])
        with self._mutex:
            for record in self.registry.list_files():
                if record["status"] in (*LOST_STATUSES, "unsupported") or not os.path.isfile(record["path"]):
                    continue
                if record["status"] == "named" and record.get("content_mode") == "name":
                    continue  # nothing to read
                # Failed files are retried only once they change; pending ones always run.
                # Files parsed by an older parser version are re-indexed either way.
                if record["status"] in ("indexed", "failed") and self._unchanged(record):
                    if not self._parser_outdated(record):
                        continue
                    stats["reparse"] += 1
                self.enqueue(record["file_id"])
                stats["queued"] += 1
        logger.info("Reconciled file registry: %s", stats)
        return stats

    def _monitor_loop(self) -> None:
        """Bring offline folders back when their drive is plugged in again; sweep single files."""
        interval = float(getattr(self.settings, "library_monitor_seconds", 30.0))
        sweep_every = float(getattr(self.settings, "library_sweep_seconds", 180.0))
        while not self._stopping.wait(interval):
            try:
                if time.monotonic() - self._last_sweep >= sweep_every:
                    self.sweep()
                else:
                    self.check_volumes()
            except Exception:
                logger.exception("Volume monitor failed")

    def check_volumes(self) -> None:
        for root in self.registry.list_roots():
            self.check_root(root)

    # ======================================================================
    # legacy migration (content-hash doc ids -> registry file ids)
    # ======================================================================

    def migrate_legacy(self, search_dirs: Iterable[str | Path] = ()) -> list[dict[str, Any]]:
        """Give documents indexed before the file registry a file_id and a registry row.

        Chunks, parents and the parsed snapshot are re-keyed in place (no re-embedding).
        The original is looked for in ``upload_dir/<old id>/`` and in ``search_dirs``
        by name + legacy content hash; when it cannot be found the row is ``missing``
        so the user can relocate it. Already migrated rows that are still missing are
        looked up again, so the script can be re-run with more search folders.
        """
        reassign = getattr(self.store, "reassign_document", None)
        candidates = self._legacy_candidates(search_dirs)
        migrated: list[dict[str, Any]] = []
        with self._mutex:
            for document in self.store.list_documents():
                doc_id = str(document["doc_id"])
                if self.registry.get_file(doc_id, include_deleted=True):
                    continue
                source = str(document["source"])
                if len(doc_id) == _LEGACY_ID_LENGTH:
                    if reassign is None:
                        logger.warning("Store %s cannot re-key legacy documents", type(self.store).__name__)
                        break
                    old_id, file_id = doc_id, new_id()
                    path, origin = self._locate_legacy(old_id, source, file_id, candidates)
                else:
                    # Already re-keyed but never registered (an interrupted migration).
                    old_id, file_id = None, doc_id
                    path = next((found for (name, _), found in candidates.items() if name == source.lower()), None)
                    origin = "import"
                fields: dict[str, Any] = {"legacy_doc_id": old_id}
                if path and os.path.isfile(path) and self._free_path(path):
                    facts = file_facts(path)
                    digest = sha256_file(path)
                    fields.update(status="indexed", sha256=digest, indexed_sha256=digest, **self._facts_fields(facts))
                else:
                    path = str(Path(self.settings.upload_dir) / file_id / source)
                    fields.update(status="missing", error="迁移时未找到原文件，可通过“重新定位”指定。")
                root = self.registry.root_for_path(path)
                # Register first: if re-keying is interrupted, the next run finds the row.
                self.registry.insert_file(
                    path,
                    root_id=root["root_id"] if root else None,
                    origin="watch" if root else origin,
                    file_id=file_id,
                    **fields,
                )
                if old_id:
                    reassign(old_id, file_id)
                    self.parent_store.reassign_document(old_id, file_id)
                    self._reassign_parsed(old_id, file_id)
                self.registry.update_file(file_id, chunks=self._chunk_count(file_id))
                migrated.append(self.registry.get_file(file_id))
            for record in self.registry.list_files(statuses=("missing",)):
                old_id = record.get("legacy_doc_id")
                if not old_id or os.path.exists(record["path"]):
                    continue
                path = candidates.get((record["name"].lower(), old_id))
                if path and self._free_path(path):
                    # The legacy hash matched, so the indexed text is this file's content.
                    digest = sha256_file(path)
                    self.registry.update_file(record["file_id"], indexed_sha256=digest)
                    self._relocate({**record, "indexed_sha256": digest}, file_facts(path), sha256=digest, note=False)
                    if self.registry.root_for_path(path) is None:
                        self.registry.update_file(record["file_id"], origin="import")
                    migrated.append(self.registry.get_file(record["file_id"]))
        if migrated:
            logger.info("Migrated %d legacy documents into the file registry", len(migrated))
        return migrated

    def _free_path(self, path: str) -> bool:
        """Make room for a migrated record at ``path``: a row there that was only
        registered by a scan (never indexed) is replaced; an indexed one wins."""
        occupant = self.registry.get_by_path(path)
        if occupant is None:
            return True
        if occupant.get("indexed_sha256"):
            return False
        self.registry.soft_delete_file(occupant["file_id"])
        self._queue.discard(occupant["file_id"])
        return True

    def _legacy_candidates(self, search_dirs: Iterable[str | Path]) -> dict[tuple[str, str], str]:
        found: dict[tuple[str, str], str] = {}
        for directory in search_dirs:
            for entry in walk(normalize_path(directory), self.extensions, self.max_bytes):
                path = Path(entry.facts.path)
                try:
                    found.setdefault((path.name.lower(), stable_doc_id(path)), str(path))
                except OSError:
                    continue
        return found

    def _locate_legacy(
        self, old_id: str, source: str, file_id: str, candidates: dict[tuple[str, str], str]
    ) -> tuple[str | None, str]:
        upload_dir = Path(self.settings.upload_dir)
        uploaded = upload_dir / old_id / source
        if uploaded.is_file():
            destination = upload_dir / file_id / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            uploaded.replace(destination)
            shutil.rmtree(upload_dir / old_id, ignore_errors=True)
            return str(destination), "upload"
        return candidates.get((source.lower(), old_id)), "import"

    def _reassign_parsed(self, old_id: str, file_id: str) -> None:
        parsed_dir = Path(self.settings.parsed_dir)
        old_path = parsed_dir / f"{old_id}.json"
        if not old_path.is_file():
            return
        payload = json.loads(old_path.read_text(encoding="utf-8"))
        payload["doc_id"] = file_id
        (parsed_dir / f"{file_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        old_path.unlink()

    def _chunk_count(self, file_id: str) -> int:
        counts = getattr(self.store, "chunk_counts", None)
        return int(counts().get(file_id, 0)) if counts else 0

    # ======================================================================
    # internals
    # ======================================================================

    @staticmethod
    def _facts_fields(facts: FileFacts) -> dict[str, Any]:
        return {"size": facts.size, "mtime": facts.mtime, "volume_id": facts.volume_id, "inode": facts.inode}

    def _version(self, record: dict[str, Any]) -> int:
        mode = record.get("image_mode")
        if mode:
            return parser_version(record["path"], self.settings, mode)
        return parser_version(record["path"], self.settings)

    def _parser_outdated(self, record: dict[str, Any]) -> bool:
        """Indexed (or failed) by another parser/chunker version or image mode than the current one."""
        return (record.get("parser_version") or 1) != self._version(record)

    def requeue_outdated(self) -> int:
        """Queue every indexed file whose index version changed (e.g. image understanding switched)."""
        queued = 0
        with self._mutex:
            for record in self.registry.list_files(statuses=("indexed", "failed")):
                if os.path.isfile(record["path"]) and self._parser_outdated(record):
                    self.enqueue(record["file_id"])
                    queued += 1
        return queued

    @staticmethod
    def _unchanged(record: dict[str, Any]) -> bool:
        try:
            facts = file_facts(record["path"])
        except OSError:
            return False
        return facts.size == record["size"] and abs(facts.mtime - (record["mtime"] or 0)) < 1e-3

    def _lost_status(self, record: dict[str, Any]) -> str:
        return "missing" if volume_available(record["path"], record.get("volume_id")) else "offline"

    def _mark_lost(self, record: dict[str, Any]) -> str:
        """The record's path is gone: follow the file by its ID, else mark it missing / offline."""
        outcome = self._follow(record)
        if outcome is not None:
            return outcome
        current = self.registry.get_file(record["file_id"])
        if current is None:
            return "removed"
        return self._set_lost(current)

    def _set_lost(self, record: dict[str, Any], note: bool = True) -> str:
        status = self._lost_status(record)
        if record["status"] == "unsupported" and status == "missing":
            # Nothing was indexed; an unsupported file that disappears is just forgotten.
            self.registry.soft_delete_file(record["file_id"])
            return "removed"
        if record["status"] != status:
            self.registry.update_file(record["file_id"], status=status)
            if record["status"] == "recycled":
                self._hidden = None
            if note:
                self._note(status, record)
        self._queue.discard(record["file_id"])
        return status

    def _lost_records_by_size(self) -> dict[int, list[dict[str, Any]]]:
        lost: dict[int, list[dict[str, Any]]] = {}
        for record in self.registry.list_files(statuses=("missing", "offline")):
            if record["size"] is not None and record["sha256"]:
                lost.setdefault(int(record["size"]), []).append(record)
        return lost

    def _upsert_entry(
        self,
        facts: FileFacts,
        supported: bool,
        reason: str,
        root: dict[str, Any],
        lost_by_size: dict[int, list[dict[str, Any]]] | None,
        note: bool = True,
        urgent: bool = False,
    ) -> str:
        record = self.registry.get_by_path(facts.path)
        if record is not None:
            return self._refresh(record, facts, supported, reason, root, note)
        moved = self._find_moved(facts, supported, lost_by_size)
        if moved is not None:
            if self._relocate(moved, facts, note=note) and moved["root_id"] != root["root_id"]:
                self.registry.update_file(moved["file_id"], root_id=root["root_id"], origin="watch")
            return "moved"
        decision = self._decision(facts.path, root)
        if decision.excluded:
            return "excluded"  # walk() skips these; a watcher event may still name one
        record = self.registry.insert_file(
            facts.path,
            root_id=root["root_id"],
            status="pending" if supported else "unsupported",
            error=reason or None,
            image_mode=self._image_mode_for(facts.path, root) if supported else None,
            content_mode="name" if decision.name_only else None,
            **self._facts_fields(facts),
        )
        if supported:
            if note:
                self._note("new", record)
            self.enqueue(record["file_id"], urgent=urgent)
        return "new"

    def _refresh(
        self, record: dict[str, Any], facts: FileFacts, supported: bool, reason: str, root: dict[str, Any],
        note: bool = True,
    ) -> str:
        changed = facts.size != record["size"] or abs(facts.mtime - (record["mtime"] or 0)) >= 1e-3
        fields = self._facts_fields(facts)
        if record["root_id"] != root["root_id"]:
            fields["root_id"] = root["root_id"]
        if not supported:
            if record["status"] != "unsupported" or record["error"] != reason or changed:
                self.registry.update_file(record["file_id"], status="unsupported", error=reason, **fields)
            return "changed" if changed else "same"
        if record["status"] in (*LOST_STATUSES, "unsupported"):
            self.registry.update_file(record["file_id"], status="pending", error=None, **fields)
            if record["status"] == "recycled":
                self._hidden = None
                if note:
                    self._note("restored", record)
            self.enqueue(record["file_id"])
            return "changed"
        if changed or record["status"] == "pending":
            # The worker hashes the file and skips re-embedding when the content is the same.
            self.registry.update_file(record["file_id"], **fields)
            if note and changed and record["status"] == "indexed":
                self._note("changed", record)
            self.enqueue(record["file_id"])
            return "changed"
        if (facts.volume_id, facts.inode) != (record["volume_id"], record["inode"]):
            self.registry.update_file(record["file_id"], volume_id=facts.volume_id, inode=facts.inode)
        return "same"

    def _find_moved(
        self, facts: FileFacts, supported: bool, lost_by_size: dict[int, list[dict[str, Any]]] | None
    ) -> dict[str, Any] | None:
        """A known record whose file now lives at ``facts.path``: same volume + file ID,
        else (for indexable files) same size + sha256."""
        suffix = Path(facts.path).suffix.lower()
        if facts.inode:
            for record in self.registry.find_by_identity(facts.volume_id, facts.inode):
                # File IDs can be reused after a delete; also require the same file type.
                if Path(record["path"]).suffix.lower() == suffix and not os.path.exists(record["path"]):
                    return record
        if not supported:
            return None
        if lost_by_size is None:
            lost_by_size = self._lost_records_by_size()
        candidates = lost_by_size.get(facts.size)
        if not candidates:
            return None
        digest = sha256_file(facts.path)
        for record in candidates:
            if record["sha256"] == digest and not os.path.exists(record["path"]):
                candidates.remove(record)
                return record
        return None

    def _relocate(self, record: dict[str, Any], facts: FileFacts, sha256: str | None = None, note: bool = True) -> bool:
        """Point a record at its new path; the index is reused, never rebuilt for a move.

        A file moved into an excluded folder is let go of (returns False); one moved into a
        folder kept by name only is kept by name only from now on. Moving out of such a folder
        changes nothing: the file is never read again without the user saying so."""
        file_id = record["file_id"]
        fields: dict[str, Any] = {"path": facts.path, **self._facts_fields(facts)}
        root = self.registry.root_for_path(facts.path)
        decision = self._decision(facts.path, root, single=root is None)
        if decision.excluded:
            self.remove_file(file_id, source="auto" if note else None)
            return False
        entering_name = (
            decision.name_only and record.get("content_mode") != "name"
            and not self._decision(record["path"]).name_only
        )
        fields["root_id"] = root["root_id"] if root else None
        if record.get("origin") != "upload":
            # A file that left every watched folder is kept as a single file (and swept, not watched).
            fields["origin"] = "watch" if root else "import"
        unchanged = facts.size == record["size"] and abs(facts.mtime - (record["mtime"] or 0)) < 1e-3
        if sha256 is not None:
            fields["sha256"] = sha256
            unchanged = sha256 == record.get("indexed_sha256")
        if record["status"] == "unsupported":
            pass
        elif unchanged and record.get("indexed_sha256"):
            fields.update(status="indexed", error=None)
        else:
            fields.update(status="pending", error=None)
        if record["status"] == "recycled":
            self._hidden = None
        self.registry.update_file(file_id, **fields)
        new_name = os.path.basename(facts.path)
        if new_name != record["name"]:
            self._rename_index(file_id, new_name)
        if note and path_key(facts.path) != record.get("path_key", path_key(record["path"])):
            self._note("moved", {**record, "name": new_name}, path=facts.path, old_path=record["path"])
        if entering_name:
            self._switch_to_name(self.registry.get_file(file_id) or record)
        elif fields.get("status") == "pending":
            self.enqueue(file_id)
        return True

    def _rename_index(self, file_id: str, name: str) -> None:
        rename = getattr(self.store, "rename_source", None)
        if rename is not None:
            rename(file_id, name)
        try:
            self.parent_store.rename_source(file_id, name)
        except (OSError, ValueError) as exc:
            logger.warning("Could not rename parents of %s: %s", file_id, exc)
        parsed = Path(self.settings.parsed_dir) / f"{file_id}.json"
        if parsed.is_file():
            payload = json.loads(parsed.read_text(encoding="utf-8"))
            payload["source"] = name
            parsed.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _drop_index(self, file_id: str, record: dict[str, Any] | None = None, purge: bool = False) -> None:
        """Delete a file's chunks, vectors, parents and parsed text. ``purge``: for good (removed,
        excluded or kept by name only): the store overwrites the deleted rows, an image's cached
        reading goes too, and ``on_purged`` cleans the free pages and the backups."""
        delete = getattr(self.store, "purge_document", None) if purge else None
        try:
            (delete or self.store.delete_document)(file_id)
        except ValueError:
            pass
        try:
            self.parent_store.delete_document(file_id)
        except ValueError:
            pass
        (Path(self.settings.parsed_dir) / f"{file_id}.json").unlink(missing_ok=True)
        if purge and record is not None and is_image(record["path"]):
            try:
                forget_readings(record["path"], self.settings)
            except Exception as exc:  # the cache is local; a failure must not keep the file
                logger.warning("Cached reading of %s not removed: %s", record["name"], exc)
        self._notify(self.on_dropped, file_id)
        if purge and self.on_purged is not None:
            try:
                self.on_purged([file_id])
            except Exception as exc:
                logger.warning("Purge of %s not scheduled: %s", file_id, exc)

    @staticmethod
    def _notify(listener: Callable[[str], None] | None, file_id: str) -> None:
        if listener is None:
            return
        try:
            listener(file_id)
        except Exception as exc:
            logger.warning("Library listener failed for %s: %s", file_id, exc)
