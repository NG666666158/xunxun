"""Background work yields the processor to a question being answered.

Indexing threads (the library's workers, local OCR) run below normal priority, so when
both want the CPU the answer's threads go first; when nothing else runs, the background
threads still get the whole machine. Windows only; elsewhere this does nothing.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, TypeVar

BELOW_NORMAL = -1   # THREAD_PRIORITY_BELOW_NORMAL: the library's workers
LOWEST = -2         # THREAD_PRIORITY_LOWEST: recognition, the heaviest work

_done = threading.local()
T = TypeVar("T")


def lower_current_thread(level: int = BELOW_NORMAL) -> None:
    """Lower the calling thread's scheduling priority (once per thread and level)."""
    if os.name != "nt" or getattr(_done, "level", None) == level:
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    if kernel32.SetThreadPriority(ctypes.c_void_p(kernel32.GetCurrentThread()), level):
        _done.level = level


def lower_new_threads(make: Callable[[], T], level: int = LOWEST) -> tuple[T, int]:
    """Call ``make()``; the native threads it starts (an ONNX Runtime session's thread pool)
    get ``level``. Python threads started meanwhile (a request being handled) are left
    alone. Returns make's result and how many threads were lowered."""
    if os.name != "nt":
        return make(), 0
    before = _process_threads()
    result = make()
    python = {thread.native_id for thread in threading.enumerate()}
    lowered = sum(_set_priority(tid, level) for tid in _process_threads() - before - python)
    return result, lowered


def _process_threads() -> set[int]:
    import ctypes
    from ctypes import wintypes

    class ThreadEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
                    ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", ctypes.c_long),
                    ("tpDeltaPri", ctypes.c_long), ("dwFlags", wintypes.DWORD)]

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    snapshot = kernel32.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return set()
    found: set[int] = set()
    entry = ThreadEntry()
    entry.dwSize = ctypes.sizeof(ThreadEntry)
    pid = os.getpid()
    try:
        more = kernel32.Thread32First(ctypes.c_void_p(snapshot), ctypes.byref(entry))
        while more:
            if entry.th32OwnerProcessID == pid:
                found.add(int(entry.th32ThreadID))
            more = kernel32.Thread32Next(ctypes.c_void_p(snapshot), ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(snapshot))
    return found


def _set_priority(thread_id: int, level: int) -> bool:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.OpenThread.restype = ctypes.c_void_p
    handle = kernel32.OpenThread(0x0020 | 0x0040, False, thread_id)  # SET_ / QUERY_INFORMATION
    if not handle:
        return False
    try:
        return bool(kernel32.SetThreadPriority(ctypes.c_void_p(handle), level))
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
