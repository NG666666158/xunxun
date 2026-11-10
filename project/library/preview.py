"""Original-file preview: which renderer shows a file, which files may be read, and Word → PDF.

The reader panel shows the original file rather than the parsed text whenever it can:
PDF.js for PDF, SheetJS for spreadsheets, the browser for Markdown/TXT/images. Word files
are converted to PDF by the Word installed on this machine (COM automation) and shown by
PDF.js; without Word, or when the conversion fails, the UI falls back to docx-preview.
PowerPoint decks go the same way, with the slides drawn from the deck's own XML as the fallback.

Only files the library itself tracks may be read: a registry record whose path lies in a
live watched folder (or the upload directory, for browser uploads).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable
import uuid

from ingestion.office import (
    PowerPointBusy,
    export_with_powerpoint,
    kill_if_hidden,
    kill_process,
    powerpoint_installed,
    process_alive as _process_alive,
    process_ids,
)

from .registry import path_key

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff")
# Images the webview cannot show itself: the reader gets a PNG of the first page.
CONVERTED_IMAGES = (".tif", ".tiff")
KIND_BY_EXTENSION = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".pptx": "pptx",
    ".ppt": "ppt",
    ".xlsx": "sheet",
    ".xls": "sheet",
    ".csv": "sheet",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    **{ext: "image" for ext in IMAGE_EXTENSIONS},
}
WORD_TIMEOUT_SECONDS = 90.0
WD_EXPORT_FORMAT_PDF = 17
WD_EXPORT_OPTIMIZE_FOR_ONSCREEN = 1
WD_DO_NOT_SAVE_CHANGES = 0
WD_ALERTS_NONE = 0
MSO_AUTOMATION_SECURITY_FORCE_DISABLE = 3


def preview_kind(ext: str) -> str:
    """File family by extension: pdf, docx, doc, sheet, markdown, text, image or other."""
    return KIND_BY_EXTENSION.get((ext or "").lower(), "other")


def preview_plan(ext: str, file_available: bool, word_available: bool, powerpoint_available: bool = False) -> dict[str, str]:
    """How the reader shows the original.

    ``renderer`` is what to try first; ``fallback`` is used when it fails (a Word
    conversion error). ``parsed`` means the original cannot be read and the indexed text is
    shown instead; ``external`` offers "open with the default program".
    """
    kind = preview_kind(ext)
    if not file_available:
        return {"kind": kind, "renderer": "parsed", "fallback": "parsed"}
    if kind == "pdf":
        return {"kind": kind, "renderer": "pdfjs", "fallback": "external"}
    if kind == "docx":
        return {"kind": kind, "renderer": "word-pdf" if word_available else "docx-preview", "fallback": "docx-preview"}
    if kind == "doc":
        # docx-preview only reads OOXML; a legacy .doc needs Word.
        return {"kind": kind, "renderer": "word-pdf" if word_available else "external", "fallback": "external"}
    if kind == "pptx":
        return {"kind": kind, "renderer": "slides-pdf" if powerpoint_available else "slides", "fallback": "slides"}
    if kind == "ppt":
        # A legacy .ppt has no XML to draw from.
        return {"kind": kind, "renderer": "slides-pdf" if powerpoint_available else "external", "fallback": "external"}
    if kind == "sheet":
        return {"kind": kind, "renderer": "sheetjs", "fallback": "external"}
    if kind in ("markdown", "text", "image"):
        return {"kind": kind, "renderer": kind, "fallback": "external"}
    return {"kind": kind, "renderer": "external", "fallback": "external"}


def _within(path: str, directory: str) -> bool:
    key, base = path_key(path), path_key(directory).rstrip(os.sep)
    return key == base or key.startswith(base + os.sep)


def readable_original(record: dict[str, Any] | None, registry: Any, upload_dir: str | os.PathLike[str]) -> Path | None:
    """The original of a registry record, if it may be read: it exists and lies in a live
    watched folder (uploads: in the upload directory). The real path (links resolved) must
    stay inside as well, so a link in a watched folder cannot expose other files.

    A file the user added on its own (``origin = "import"``, no folder) may be read while
    it is still the file that was added: same volume and file ID as the registry recorded,
    so a stale or edited row cannot point the reader at some other file."""
    if not record or record.get("deleted_at"):
        return None
    path = str(record.get("path") or "")
    if not path or not os.path.isfile(path):
        return None
    real = os.path.realpath(path)
    if record.get("origin") == "upload":
        base = os.path.realpath(os.fspath(upload_dir))
        return Path(path) if _within(path, os.fspath(upload_dir)) and _within(real, base) else None
    root = registry.root_for_path(path)
    if root is None:
        return Path(path) if record.get("origin") == "import" and _same_file(path, record) else None
    return Path(path) if _within(real, os.path.realpath(root["path"])) else None


def _same_file(path: str, record: dict[str, Any]) -> bool:
    if not record.get("inode"):
        return False
    try:
        info = os.stat(path)
    except OSError:
        return False
    return int(info.st_ino) == int(record["inode"]) and int(info.st_dev) == int(record.get("volume_id") or 0)


# ----------------------------------------------------------------------------
# Word → PDF
# ----------------------------------------------------------------------------


class PreviewError(Exception):
    """A conversion that did not produce a PDF; ``reason`` is shown to the user."""

    status = 422

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class WordUnavailable(PreviewError):
    status = 409


class ConversionTimeout(PreviewError):
    status = 504


class ApplicationBusy(PreviewError):
    """The user has the application open; try again later, nothing is wrong with the file."""

    status = 409


Engine = Callable[[str, str, Callable[[int], None]], None]


def word_installed() -> bool:
    """Word is registered for COM automation (and pywin32 can drive it)."""
    if sys.platform != "win32":
        return False
    try:
        import winreg

        import pythoncom  # noqa: F401 - pywin32 must be importable
        import win32com.client  # noqa: F401

        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Word.Application\CLSID"):
            return True
    except (ImportError, OSError):
        return False


def _word_pids() -> set[int]:
    """Process ids of running WINWORD.EXE instances."""
    return process_ids("winword.exe")


def export_with_word(source: str, target: str, on_pid: Callable[[int], None]) -> None:
    """Open ``source`` read-only in a new, hidden Word instance and save it as PDF.

    A separate instance (CoCreateInstanceEx) never touches documents the user has open in
    Word. ``on_pid`` receives that instance's process id, so a caller that gives up can kill
    it. Macros are disabled and no dialog may appear: a password-protected file fails with
    the dummy password instead of prompting.
    """
    import pythoncom
    import pywintypes
    import win32com.client.dynamic
    import win32gui
    import win32process

    pythoncom.CoInitialize()
    word = document = None
    pid = 0
    try:
        before = _word_pids()
        clsid = pywintypes.IID("Word.Application")
        dispatch = pythoncom.CoCreateInstanceEx(
            clsid, None, pythoncom.CLSCTX_LOCAL_SERVER, None, (pythoncom.IID_IDispatch,)
        )[0]
        word = win32com.client.dynamic.Dispatch(dispatch)
        # Find our instance by a unique caption on its (hidden) main window; fall back to
        # the one WINWORD.EXE that was not running before (conversions are serialized).
        caption = f"ekb-preview-{uuid.uuid4().hex}"
        try:
            word.Caption = caption
            hwnd = win32gui.FindWindow("OpusApp", caption)
            if hwnd:
                pid = win32process.GetWindowThreadProcessId(hwnd)[1]
        except Exception:
            pid = 0
        if not pid:
            new = _word_pids() - before
            pid = new.pop() if len(new) == 1 else 0
        if pid:
            on_pid(pid)
        word.Visible = False
        word.DisplayAlerts = WD_ALERTS_NONE
        word.ScreenUpdating = False
        try:
            word.AutomationSecurity = MSO_AUTOMATION_SECURITY_FORCE_DISABLE
        except Exception:
            pass
        document = word.Documents.Open(
            FileName=source,
            ConfirmConversions=False,
            ReadOnly=True,
            AddToRecentFiles=False,
            PasswordDocument="ekb-no-password",
            Visible=False,
            OpenAndRepair=False,
            NoEncodingDialog=True,
        )
        document.ExportAsFixedFormat(
            OutputFileName=target,
            ExportFormat=WD_EXPORT_FORMAT_PDF,
            OpenAfterExport=False,
            OptimizeFor=WD_EXPORT_OPTIMIZE_FOR_ONSCREEN,
        )
    finally:
        try:
            if document is not None:
                document.Close(SaveChanges=WD_DO_NOT_SAVE_CHANGES)
        except Exception:
            pass
        try:
            if word is not None:
                word.Quit(SaveChanges=WD_DO_NOT_SAVE_CHANGES)
        except Exception:
            pass
        document = word = None
        pythoncom.CoUninitialize()
        # Quit normally ends the process within a moment; never leave it behind.
        if pid:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and _process_alive(pid):
                time.sleep(0.1)
            if _process_alive(pid):
                kill_process(pid)


class WordPdfConverter:
    """Word → PDF with a cache keyed by path + modification time + size.

    Conversions run one at a time (Word automation is not built for concurrency) and each is
    bounded by ``timeout``: past it the Word process is killed. A failure is remembered for
    that file version, so a broken file does not start Word on every click.
    """

    application = "Word"
    missing = "本机没有安装 Microsoft Word"
    default_engine = staticmethod(export_with_word)
    default_available = staticmethod(word_installed)
    default_kill = staticmethod(kill_process)

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        timeout: float = WORD_TIMEOUT_SECONDS,
        engine: Engine | None = None,
        available: Callable[[], bool] | None = None,
        kill: Callable[[int], None] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.engine = engine or self.default_engine
        self._available = available or self.default_available
        self._available_result: bool | None = None
        self.kill = kill or self.default_kill
        self._lock = threading.Lock()

    def available(self) -> bool:
        if self._available_result is None:
            self._available_result = bool(self._available())
        return self._available_result

    @staticmethod
    def cache_name(path: str | os.PathLike[str]) -> str:
        """``<path hash>-<version hash>.pdf``: a new version of the same file replaces the old."""
        stat = os.stat(path)
        file_part = hashlib.sha1(path_key(path).encode("utf-8")).hexdigest()[:16]
        version = hashlib.sha1(f"{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:16]
        return f"{file_part}-{version}.pdf"

    def cached(self, path: str | os.PathLike[str]) -> Path | None:
        target = self.cache_dir / self.cache_name(path)
        return target if target.is_file() else None

    def failure(self, path: str | os.PathLike[str]) -> str | None:
        """Why this version of the file could not be converted, if an attempt failed."""
        failed = (self.cache_dir / self.cache_name(path)).with_suffix(".failed")
        return (failed.read_text(encoding="utf-8") or f"{self.application} 无法转换这个文件") if failed.is_file() else None

    def convert(self, path: str | os.PathLike[str]) -> tuple[Path, bool]:
        """PDF for ``path`` and whether it came from the cache."""
        hit = self.cached(path)
        if hit:
            return hit, True
        if not self.available():
            raise WordUnavailable(self.missing)
        with self._lock:
            # Another request may have converted it while this one waited.
            name = self.cache_name(path)
            target = self.cache_dir / name
            if target.is_file():
                return target, True
            failed = target.with_suffix(".failed")
            if failed.is_file():
                raise PreviewError(failed.read_text(encoding="utf-8") or f"{self.application} 无法转换这个文件")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            partial = self.cache_dir / f"{name}.{uuid.uuid4().hex[:8]}.part.pdf"
            try:
                self._run(os.fspath(path), os.fspath(partial))
            except ConversionTimeout:
                partial.unlink(missing_ok=True)
                raise
            except PowerPointBusy as exc:
                partial.unlink(missing_ok=True)
                raise ApplicationBusy(str(exc)) from exc
            except Exception as exc:
                partial.unlink(missing_ok=True)
                logger.warning("%s conversion failed for %s: %s", self.application, path, exc)
                reason = f"{self.application} 无法转换这个文件（可能有密码保护或已损坏）"
                failed.write_text(reason, encoding="utf-8")
                raise PreviewError(reason) from exc
            if not partial.is_file() or partial.stat().st_size == 0:
                partial.unlink(missing_ok=True)
                raise PreviewError(f"{self.application} 没有生成 PDF")
            os.replace(partial, target)
            self._drop_old_versions(name)
            return target, False

    def _run(self, source: str, target: str) -> None:
        pids: list[int] = []
        errors: list[BaseException] = []

        def work() -> None:
            try:
                self.engine(source, target, pids.append)
            except BaseException as exc:  # reported to the waiting request
                errors.append(exc)

        worker = threading.Thread(target=work, name="word-to-pdf", daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            for pid in pids:
                self.kill(pid)
            worker.join(5)
            raise ConversionTimeout(f"{self.application} 转换超过 {int(self.timeout)} 秒，已停止")
        if errors:
            raise errors[0]

    def _drop_old_versions(self, name: str) -> None:
        prefix = name.split("-", 1)[0] + "-"
        for item in self.cache_dir.glob(prefix + "*"):
            if item.name != name and not item.name.endswith(".part.pdf"):
                item.unlink(missing_ok=True)


class PowerPointPdfConverter(WordPdfConverter):
    """A deck → PDF, one page per slide. The cache and limits are the same as for Word;
    the engine stays away from a PowerPoint the user has open (``ingestion.office``)."""

    application = "PowerPoint"
    missing = "本机没有安装 Microsoft PowerPoint"
    default_engine = staticmethod(export_with_powerpoint)
    default_available = staticmethod(powerpoint_installed)
    default_kill = staticmethod(kill_if_hidden)


class PreviewPrefetcher:
    """Word files converted to PDF before anyone opens them, so opening one rarely waits.

    One background worker converts queued files one at a time: a file being opened goes to
    the front, files the UI is about to show (answer cards, find-file rows) queue behind it,
    and once nothing has been asked for ``idle_after`` seconds, recently added Word files are
    converted too, a few per pass. ``resolve(doc_id)`` is the readable original or None;
    ``recent()`` lists doc ids worth converting while idle. The conversions themselves go
    through ``WordPdfConverter``: serialized, cached, bounded by its timeout.
    """

    def __init__(
        self,
        converter: WordPdfConverter,
        resolve: Callable[[str], Path | None],
        recent: Callable[[], list[str]] | None = None,
        *,
        idle_after: float = 120.0,
        idle_batch: int = 3,
        tick: float = 30.0,
        pause: float = 1.0,
        kinds: tuple[str, ...] = ("docx", "doc"),
    ):
        self.converter = converter
        self.kinds = kinds
        self.resolve = resolve
        self.recent = recent
        self.idle_after = idle_after
        self.idle_batch = idle_batch
        self.tick = tick
        self.pause = pause
        self._cond = threading.Condition()
        self._queue: list[tuple[str, Path]] = []
        self._pending: dict[str, threading.Event] = {}
        # Background conversions that failed (a timeout leaves no marker): not retried in
        # the background; opening the file still tries again.
        self._gave_up: set[str] = set()
        self._last_request = time.monotonic()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        with self._cond:
            if self._thread is None or not self._thread.is_alive():
                self._stopping.clear()
                self._thread = threading.Thread(target=self._run, name="preview-prefetch", daemon=True)
                self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        with self._cond:
            self._cond.notify_all()

    def request(self, doc_id: str, *, urgent: bool = False) -> str:
        """Queue a Word original for conversion: "ready", "queued", or "skip" (not Word, no
        Word on this machine, unreadable, or this version already failed)."""
        with self._cond:
            self._last_request = time.monotonic()
        state = self._enqueue(doc_id, urgent)
        if state == "queued":
            self.start()
        return state

    def wait(self, path: str | os.PathLike[str], timeout: float) -> Path | None:
        """The cached PDF, waiting up to ``timeout`` seconds for a conversion under way."""
        with self._cond:
            event = self._pending.get(path_key(path))
        if event is not None:
            event.wait(timeout)
        try:
            return self.converter.cached(path)
        except OSError:
            return None

    def pending(self) -> int:
        with self._cond:
            return len(self._pending)

    def _enqueue(self, doc_id: str, urgent: bool) -> str:
        if not self.converter.available():
            return "skip"
        path = self.resolve(doc_id)
        if path is None or preview_kind(Path(path).suffix) not in self.kinds:
            return "skip"
        try:
            if self.converter.cached(path):
                return "ready"
            if self.converter.failure(path):
                return "skip"
        except OSError:
            return "skip"
        key = path_key(path)
        with self._cond:
            if not urgent and key in self._gave_up:
                return "skip"
            if key in self._pending:
                if urgent:
                    # Already waiting its turn: someone is opening it now, so it goes first.
                    for i, item in enumerate(self._queue):
                        if item[0] == key:
                            self._queue.insert(0, self._queue.pop(i))
                            break
                return "queued"
            self._pending[key] = threading.Event()
            self._queue.insert(0 if urgent else len(self._queue), (key, Path(path)))
            self._cond.notify()
        return "queued"

    def _idle_pass(self) -> None:
        """While nobody is asking: queue a few recently added Word files not converted yet."""
        if self.recent is None:
            return
        try:
            candidates = self.recent()
        except Exception:
            logger.warning("Listing recent Word files for preview failed", exc_info=True)
            return
        added = 0
        for doc_id in candidates:
            if added >= self.idle_batch:
                break
            if self._enqueue(doc_id, urgent=False) == "queued":
                added += 1

    def _run(self) -> None:
        while not self._stopping.is_set():
            with self._cond:
                if not self._queue:
                    self._cond.wait(self.tick)
                idle = not self._queue and time.monotonic() - self._last_request >= self.idle_after
            if self._stopping.is_set():
                return
            if idle:
                self._idle_pass()
            with self._cond:
                if not self._queue:
                    continue
                key, path = self._queue.pop(0)
            try:
                self.converter.convert(path)
            except Exception as exc:
                logger.info("Preview prefetch skipped %s: %s", path.name, exc)
                with self._cond:
                    self._gave_up.add(key)
            finally:
                with self._cond:
                    event = self._pending.pop(key, None)
                if event is not None:
                    event.set()
            # Low priority: leave the machine (and Word) a moment between files.
            self._stopping.wait(self.pause)
