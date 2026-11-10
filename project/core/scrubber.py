"""Make dropped text really gone: from the index file's free pages and from the backups.

When a file is removed from the library, excluded, or switched to "只按文件名收录", the library
deletes its chunks at once (``LibraryService._drop_index``). Words can still sit in the
database file afterwards (old full-text segments, free pages, the write-ahead log) and in the
daily backups, which are copies of that file. A few seconds after the last such change the
scrubber rebuilds the index file and cleans every backup (``DataSafety.scrub_backups``), so a
whole folder switched at once costs one rebuild, about a second for this size of library
(design/PRIVACY.md).

The files still to scrub are written to ``scrub-pending.json``: a scrub cut short by quitting
the app is finished at the next start.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

PENDING_FILE = "scrub-pending.json"
# Seconds without a new purge before the scrub runs, and before a failed one is tried again.
DELAY_SECONDS = 3.0
RETRY_SECONDS = 60.0
MAX_TRIES = 5


class Scrubber:
    def __init__(
        self,
        settings: Any,
        safety: Any,
        store: Any,
        registry: Any,
        delay: float = DELAY_SECONDS,
    ):
        self.settings = settings
        self.safety = safety
        self.store = store
        self.registry = registry
        self.delay = delay
        self.path = Path(settings.data_dir) / PENDING_FILE
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._run_lock = threading.Lock()  # one scrub at a time
        self._pending: set[str] = self._load()
        self._last_add = 0.0
        self._thread: threading.Thread | None = None
        self.last_scrub: str = ""
        self.last_error: str = ""
        # Tests: called after every scrub with the ids it covered.
        self.on_scrubbed: Callable[[list[str]], None] | None = None

    # ---------------- queue ----------------

    def add(self, file_ids: Iterable[str]) -> None:
        ids = {str(file_id) for file_id in file_ids if file_id}
        if not ids:
            return
        with self._wake:
            self._pending |= ids
            self._last_add = time.monotonic()
            self._save()
            self._wake.notify_all()
        self._start()

    def resume(self) -> None:
        """At start-up: finish a scrub the last run did not get to."""
        if self._pending:
            self._start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"pending": len(self._pending), "last": self.last_scrub, "error": self.last_error}

    def _start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, name="privacy-scrub", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        failures = 0
        while True:
            with self._wake:
                if not self._pending:
                    self._thread = None
                    return
                # Wait until purges stop coming (a folder switched file by file).
                while (wait := self._last_add + self.delay - time.monotonic()) > 0:
                    self._wake.wait(wait)
            if self.run_now():
                failures = 0
                continue
            failures += 1
            if failures >= MAX_TRIES:
                # Left in scrub-pending.json for the next start (or the next purge).
                with self._lock:
                    self._thread = None
                return
            time.sleep(RETRY_SECONDS)

    # ---------------- the work ----------------

    def run_now(self) -> bool:
        """Scrub what is pending; True when done (or nothing to do)."""
        with self._run_lock:
            return self._run()

    def _run(self) -> bool:
        with self._lock:
            ids = sorted(self._pending)
        if not ids:
            return True
        started = time.perf_counter()
        try:
            scrub = getattr(self.store, "scrub", None)
            if scrub is not None:
                scrub()
            rows = {file_id: self.registry.get_file(file_id, include_deleted=True) for file_id in ids}
            rules = self.registry.list_rules(include_deleted=True)
            backups = self.safety.scrub_backups(ids, rows, rules)
        except Exception as exc:
            logger.warning("Scrubbing %d purged files failed, will retry: %s", len(ids), exc)
            with self._lock:
                self.last_error = str(exc)[:300]
            return False
        with self._wake:
            self._pending -= set(ids)
            self._save()
            self.last_scrub = datetime.now().isoformat(timespec="seconds")
            self.last_error = ""
        logger.info(
            "Scrubbed %d purged files from the index and %d backups in %.1f s",
            len(ids), backups, time.perf_counter() - started,
        )
        if self.on_scrubbed is not None:
            self.on_scrubbed(ids)
        return True

    # ---------------- persistence ----------------

    def _load(self) -> set[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return set()
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable %s", self.path.name)
            return set()
        return {str(item) for item in data.get("files", [])} if isinstance(data, dict) else set()

    def _save(self) -> None:
        """Under the lock. Written to a temporary file first so a crash never leaves half a list."""
        if not self._pending:
            self.path.unlink(missing_ok=True)
            return
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"files": sorted(self._pending)}), encoding="utf-8")
        temporary.replace(self.path)
