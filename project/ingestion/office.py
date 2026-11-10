"""PowerPoint on this machine, driven through COM: a deck to PDF (the reader) and .ppt to .pptx.

PowerPoint, unlike Word, keeps one process per user: asking for a new instance connects to the
one already open. So nothing here runs while the user has PowerPoint open
(``PowerPointBusy``, tried again later), and an instance is only ever quit or killed when this
code started it and it still shows no window: a deck the user opened meanwhile would land in
the same process.

Conversions run one at a time. PowerPoint refuses long target paths, so output goes to a short
temporary folder first.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

PPT_NEEDS_POWERPOINT = "旧版 .ppt 需要本机装有 PowerPoint 才能读取，也可以另存为 .pptx"

PPT_SAVE_AS_PDF = 32
PPT_SAVE_AS_PPTX = 24
PPT_ALERTS_NONE = 1
MSO_AUTOMATION_SECURITY_FORCE_DISABLE = 3
MSO_FALSE = 0
PPT_TIMEOUT_SECONDS = 180.0
DUMMY_PASSWORD = "ekb-no-password"

_lock = threading.Lock()


class PowerPointBusy(TimeoutError):
    """The user has PowerPoint open; a TimeoutError so the library retries the file later."""


def powerpoint_installed() -> bool:
    """PowerPoint is registered for COM automation (and pywin32 can drive it)."""
    if sys.platform != "win32":
        return False
    try:
        import winreg

        import pythoncom  # noqa: F401 - pywin32 must be importable
        import win32com.client  # noqa: F401

        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"PowerPoint.Application\CLSID"):
            return True
    except (ImportError, OSError):
        return False


def process_ids(image: str) -> set[int]:
    """Process ids of running instances of an executable, e.g. ``winword.exe``."""
    import win32api
    import win32con
    import win32process

    pids = set()
    for pid in win32process.EnumProcesses():
        try:
            handle = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        except Exception:
            continue
        try:
            name = win32process.GetModuleFileNameEx(handle, 0)
        except Exception:
            name = ""
        finally:
            win32api.CloseHandle(handle)
        if os.path.basename(name).lower() == image:
            pids.add(pid)
    return pids


def kill_process(pid: int) -> None:
    """Terminate one process that this app started."""
    if sys.platform != "win32":
        os.kill(pid, 9)
        return
    import win32api
    import win32con

    try:
        handle = win32api.OpenProcess(win32con.PROCESS_TERMINATE, False, pid)
    except Exception:
        return  # already gone
    try:
        win32api.TerminateProcess(handle, 1)
    finally:
        win32api.CloseHandle(handle)


def process_alive(pid: int) -> bool:
    import win32api
    import win32con
    import win32event

    try:
        handle = win32api.OpenProcess(win32con.SYNCHRONIZE, False, pid)
    except Exception:
        return False
    try:
        return win32event.WaitForSingleObject(handle, 0) == win32event.WAIT_TIMEOUT
    finally:
        win32api.CloseHandle(handle)


def has_visible_window(pid: int) -> bool:
    import win32gui
    import win32process

    found: list[int] = []

    def check(hwnd: int, _: object) -> bool:
        if win32gui.IsWindowVisible(hwnd) and win32process.GetWindowThreadProcessId(hwnd)[1] == pid:
            found.append(hwnd)
        return True

    win32gui.EnumWindows(check, None)
    return bool(found)


def kill_if_hidden(pid: int) -> None:
    """Stop a PowerPoint this app started, unless the user has since opened a deck in it."""
    if has_visible_window(pid):
        logger.info("PowerPoint %s now shows a window; left running", pid)
        return
    kill_process(pid)


def _with_powerpoint(source: str, on_pid: Callable[[int], None], work: Callable[[object], None]) -> None:
    import pythoncom
    import pywintypes
    import win32com.client.dynamic

    with _lock:
        running = process_ids("powerpnt.exe")
        if running:
            raise PowerPointBusy("PowerPoint 正在使用中，关闭后再试")
        pythoncom.CoInitialize()
        application = presentation = dispatch = None
        pid = 0
        try:
            clsid = pywintypes.IID("PowerPoint.Application")
            dispatch = pythoncom.CoCreateInstanceEx(
                clsid, None, pythoncom.CLSCTX_LOCAL_SERVER, None, (pythoncom.IID_IDispatch,)
            )[0]
            application = win32com.client.dynamic.Dispatch(dispatch)
            started = process_ids("powerpnt.exe")
            if len(started) == 1:
                pid = started.pop()
                on_pid(pid)
            application.DisplayAlerts = PPT_ALERTS_NONE
            try:
                application.AutomationSecurity = MSO_AUTOMATION_SECURITY_FORCE_DISABLE
            except Exception:
                pass
            # A dummy password makes a protected deck fail instead of asking for one.
            presentation = application.Presentations.Open(
                FileName=f"{source}::{DUMMY_PASSWORD}::{DUMMY_PASSWORD}",
                ReadOnly=True,
                Untitled=False,
                WithWindow=False,
            )
            work(presentation)
        finally:
            try:
                if presentation is not None:
                    presentation.Close()
            except Exception:
                pass
            other_decks = True
            try:
                other_decks = application is not None and application.Presentations.Count > 0
                if application is not None and not other_decks:
                    application.Quit()
            except Exception:
                pass
            presentation = application = dispatch = None
            pythoncom.CoUninitialize()
            # Quit normally ends the process within a moment; never leave it behind.
            if pid and not other_decks:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and process_alive(pid):
                    time.sleep(0.1)
                if process_alive(pid):
                    kill_if_hidden(pid)


def _short_target(target: str, work: Callable[[str], None]) -> None:
    """Run ``work`` with a short temporary path, then move what it wrote to ``target``."""
    folder = tempfile.mkdtemp(prefix="ekb")
    try:
        temporary = os.path.join(folder, "out" + Path(target).suffix)
        work(temporary)
        if not os.path.isfile(temporary):
            raise OSError("PowerPoint 没有生成文件")
        shutil.move(temporary, target)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def export_with_powerpoint(source: str, target: str, on_pid: Callable[[int], None]) -> None:
    """Open ``source`` read-only, without a window, and save it as PDF, one page per slide.

    Hidden slides are included: a citation's slide number has to be the PDF's page number."""

    def run(temporary: str) -> None:
        def work(presentation) -> None:
            # In memory only: the deck is open read-only and only the PDF is written.
            for number in range(1, presentation.Slides.Count + 1):
                transition = presentation.Slides(number).SlideShowTransition
                if transition.Hidden:
                    transition.Hidden = MSO_FALSE
            presentation.SaveAs(temporary, PPT_SAVE_AS_PDF)

        _with_powerpoint(source, on_pid, work)

    _short_target(target, run)


def save_as_pptx(source: str, target: str, on_pid: Callable[[int], None]) -> None:
    """A legacy .ppt saved as .pptx (``target``); the original is opened read-only."""

    def run(temporary: str) -> None:
        _with_powerpoint(source, on_pid, lambda presentation: presentation.SaveAs(temporary, PPT_SAVE_AS_PPTX))

    _short_target(target, run)


def ppt_to_pptx(source: str, timeout: float = PPT_TIMEOUT_SECONDS) -> str:
    """Convert a .ppt for parsing; returns a temporary .pptx the caller removes with its folder."""
    folder = tempfile.mkdtemp(prefix="ekbppt")
    target = os.path.join(folder, "deck.pptx")
    pids: list[int] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            save_as_pptx(source, target, pids.append)
        except BaseException as exc:  # reported to the waiting caller
            errors.append(exc)

    worker = threading.Thread(target=work, name="ppt-to-pptx", daemon=True)
    worker.start()
    worker.join(timeout)
    try:
        if worker.is_alive():
            for pid in pids:
                kill_if_hidden(pid)
            worker.join(5)
            raise TimeoutError(f"PowerPoint 读取这个文件超过 {int(timeout)} 秒，已停止")
        if errors:
            raise errors[0]
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return target
