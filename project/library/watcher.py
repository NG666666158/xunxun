"""watchdog bridge: turns file-system events into debounced ``LibraryService`` calls.

Every call runs on one dispatcher thread, in due-time order, so a move is never
handled concurrently with a change to the same file. Create/modify/delete events are
collapsed per path into one "sync this path" call that fires ``debounce`` seconds after
the last event; it then looks at the disk, so an editor's delete-and-recreate save is
seen as a modification, not a deletion. Moves are dispatched right away.

Each watched folder's parent is watched too (not recursively): renaming, moving or
deleting the watched folder itself is reported there, and the service then finds the
folder again by its file ID.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import os
import threading
import time
from typing import Any, Callable

from .registry import normalize_path, path_key
from .scanner import path_excluded

logger = logging.getLogger(__name__)


class Debouncer:
    def __init__(self, delay: float):
        self.delay = delay
        self._heap: list[tuple[float, int, Any]] = []
        self._pending: dict[Any, tuple[float, Callable[[], None]]] = {}
        self._counter = itertools.count()
        self._cond = threading.Condition()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="library-watch-dispatch", daemon=True)
        self._thread.start()

    def submit(self, key: Any, action: Callable[[], None], delay: float | None = None) -> None:
        due = time.monotonic() + (self.delay if delay is None else delay)
        with self._cond:
            self._pending[key] = (due, action)
            heapq.heappush(self._heap, (due, next(self._counter), key))
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._stopped:
                    if self._heap:
                        due, _, key = self._heap[0]
                        wait = due - time.monotonic()
                        if wait <= 0:
                            heapq.heappop(self._heap)
                            pending = self._pending.get(key)
                            # A later event for the same key pushed its due time back.
                            if pending and pending[0] == due:
                                del self._pending[key]
                                action = pending[1]
                                break
                            continue
                        self._cond.wait(wait)
                    else:
                        self._cond.wait()
                else:
                    return
            try:
                action()
            except Exception:
                logger.exception("Handling a file-system event failed")


class EventRouter:
    """Maps watchdog events for one watched folder to service calls (testable without watchdog)."""

    def __init__(self, service: Any, root: dict[str, Any], submit: Callable[..., None]):
        self.service = service
        self.root_path = root["path"]
        self.submit = submit

    def dispatch(self, event: Any) -> None:
        kind = event.event_type
        if kind in ("opened", "closed", "closed_no_write"):
            return
        src = normalize_path(event.src_path)
        if kind == "modified" and event.is_directory:
            return
        if kind == "moved":
            dest = normalize_path(event.dest_path)
            src_out = path_excluded(src, self.root_path)
            dest_out = path_excluded(dest, self.root_path)
            if src_out and dest_out:
                return
            if src_out:
                self._sync(dest)
            elif dest_out:
                self._sync(src)
            elif self._save_swap(event, src, dest):
                # Word and others save by renaming the document to a temp name and writing a
                # new file in its place: look at both paths once the save has settled, so the
                # record (and its vectors) stays with the document's path.
                self._sync(src)
                self._sync(dest)
            else:
                self.submit(("move", path_key(src), path_key(dest)), lambda: self.service.handle_move(src, dest), 0)
            return
        if not path_excluded(src, self.root_path):
            self._sync(src)

    def _sync(self, path: str) -> None:
        self.submit(("sync", path_key(path)), lambda: self.service.handle_path(path))

    def _save_swap(self, event: Any, src: str, dest: str) -> bool:
        """A document renamed to a name the library does not index (``报告.docx`` -> ``~WRL0001.tmp``)."""
        extensions = getattr(self.service, "extensions", ())
        return (
            not event.is_directory
            and os.path.splitext(src)[1].lower() in extensions
            and os.path.splitext(dest)[1].lower() not in extensions
        )


class ParentRouter:
    """Events in the folder that contains watched folders: a watched folder renamed, moved
    away, sent to the Recycle Bin or restored. Only events naming a watched folder count."""

    # Seconds to let a move settle before the folder is looked up by its file ID.
    delay = 0.3

    def __init__(self, service: Any, roots: Callable[[], dict[str, str]], submit: Callable[..., None]):
        self.service = service
        self.roots = roots  # path_key -> root_id of the watched folders in this parent
        self.submit = submit

    def dispatch(self, event: Any) -> None:
        if event.event_type not in ("moved", "deleted", "created"):
            return
        roots = self.roots()
        keys = {path_key(event.src_path)}
        if event.event_type == "moved":
            keys.add(path_key(event.dest_path))
        for key in keys:
            root_id = roots.get(key)
            if root_id:
                self.submit(("root", root_id), lambda root_id=root_id: self.service.check_root_by_id(root_id), self.delay)


class FolderWatcher:
    def __init__(self, service: Any, debounce: float = 2.0):
        from watchdog.observers import Observer

        self.service = service
        self.debouncer = Debouncer(debounce)
        self.observer = Observer()
        self._watches: dict[str, Any] = {}
        # Parent folder (path_key) -> its non-recursive watch, and the watched folders inside it.
        self._parent_watches: dict[str, Any] = {}
        self._parent_roots: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        self.observer.start()

    def stop(self) -> None:
        self.debouncer.stop()
        self.observer.stop()

    def watch(self, root: dict[str, Any]) -> None:
        router = EventRouter(self.service, root, self.debouncer.submit)
        with self._lock:
            if root["root_id"] not in self._watches:
                try:
                    self._watches[root["root_id"]] = self.observer.schedule(_handler(router), root["path"], recursive=True)
                except (OSError, FileNotFoundError) as exc:
                    logger.warning("Cannot watch %s: %s", root["path"], exc)
        self.watch_parent(root)

    def unwatch(self, root_id: str) -> None:
        with self._lock:
            watch = self._watches.pop(root_id, None)
            if watch is not None:
                self._unschedule(watch)

    def watch_parent(self, root: dict[str, Any]) -> None:
        """Watch the folder containing ``root`` (not recursively), for renames of ``root`` itself."""
        parent = os.path.dirname(normalize_path(root["path"]))
        if not parent or path_key(parent) == path_key(root["path"]):
            return  # a drive root has no parent to watch
        key = path_key(parent)
        with self._lock:
            for roots in self._parent_roots.values():
                for path, root_id in list(roots.items()):
                    if root_id == root["root_id"] and path != path_key(root["path"]):
                        del roots[path]
            self._parent_roots.setdefault(key, {})[path_key(root["path"])] = root["root_id"]
            if key in self._parent_watches:
                return
            router = ParentRouter(self.service, lambda key=key: dict(self._parent_roots.get(key, {})), self.debouncer.submit)
            try:
                self._parent_watches[key] = self.observer.schedule(_handler(router), parent, recursive=False)
            except (OSError, FileNotFoundError) as exc:
                logger.warning("Cannot watch %s: %s", parent, exc)

    def forget_parent(self, root_id: str) -> None:
        """Stop watching the parent of a removed (or re-pointed) folder, once no other folder needs it."""
        with self._lock:
            for key in list(self._parent_roots):
                roots = self._parent_roots[key]
                for path in [path for path, rid in roots.items() if rid == root_id]:
                    del roots[path]
                if not roots:
                    del self._parent_roots[key]
                    watch = self._parent_watches.pop(key, None)
                    if watch is not None:
                        self._unschedule(watch)

    def _unschedule(self, watch: Any) -> None:
        try:
            self.observer.unschedule(watch)
        except (KeyError, OSError):
            pass


def _handler(router: Any):
    from watchdog.events import FileSystemEventHandler

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            router.dispatch(event)

    return Handler()
