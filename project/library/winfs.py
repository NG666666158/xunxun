"""Find a file or folder again by its NTFS file ID after it was renamed or moved.

``os.stat`` gives every file and folder its volume serial (``st_dev``) and NTFS file ID
(``st_ino``). Both survive a rename, a move within the volume and a trip to the Recycle
Bin (which is a move into ``<drive>\\$Recycle.Bin\\<SID>\\``). ``OpenFileById`` opens the
item by that ID with a handle on the volume root as the hint, and
``GetFinalPathNameByHandleW`` returns where it is now. Neither needs administrator
rights (unlike the USN change journal). A move to another volume makes a new file with
a new ID; that case is not found here.

Elsewhere (and on FAT/exFAT, network shares, other systems) ``resolve`` reports
``UNSUPPORTED`` and the library falls back to its path- and content-based matching.
"""

from __future__ import annotations

import os
from pathlib import Path

# resolve() outcomes besides a path.
NOT_FOUND = "not_found"      # the volume answered: no item has this ID (deleted for good)
UNSUPPORTED = "unsupported"  # cannot tell (no volume, not NTFS, access denied, not Windows)

_RECYCLE_BIN = "$recycle.bin"

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _FileIdUnion(ctypes.Union):
        _fields_ = [("FileId", ctypes.c_longlong), ("ExtendedFileId", ctypes.c_ubyte * 16)]

    class _FileIdDescriptor(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("dwSize", wintypes.DWORD), ("Type", ctypes.c_int), ("u", _FileIdUnion)]

    _INVALID_HANDLE = ctypes.c_void_p(-1).value
    _FILE_READ_ATTRIBUTES = 0x80
    _SHARE_ALL = 0x1 | 0x2 | 0x4
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000  # needed to open folders
    _FILE_ID_TYPE = 0
    # OpenFileById errors meaning "the volume has no item with this ID".
    _GONE = {2, 3, 87, 1168}  # file / path not found, invalid parameter (measured for a deleted ID), not found

    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _kernel32.OpenFileById.restype = wintypes.HANDLE
    _kernel32.OpenFileById.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_FileIdDescriptor), wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    _kernel32.GetFinalPathNameByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


def _dos_path(path: str) -> str:
    if path.startswith("\\\\?\\UNC\\"):
        return "\\" + path[7:]
    if path.startswith("\\\\?\\"):
        return path[4:]
    return path


def resolve(volume_path: str, file_id: int | None) -> str:
    """Where the item with ``file_id`` on the volume holding ``volume_path`` is now.

    Returns its current path, ``NOT_FOUND`` or ``UNSUPPORTED``. ``volume_path`` is any
    path on that volume (the item's last known path is fine; only its drive is used).
    """
    if os.name != "nt" or not file_id:
        return UNSUPPORTED
    anchor = Path(volume_path).anchor
    if not anchor:
        return UNSUPPORTED
    hint = _kernel32.CreateFileW(anchor, 0, _SHARE_ALL, None, _OPEN_EXISTING, _FILE_FLAG_BACKUP_SEMANTICS, None)
    if hint in (None, _INVALID_HANDLE):
        return UNSUPPORTED
    try:
        descriptor = _FileIdDescriptor()
        descriptor.dwSize = ctypes.sizeof(_FileIdDescriptor)
        descriptor.Type = _FILE_ID_TYPE
        descriptor.FileId = ctypes.c_longlong(int(file_id) & 0xFFFFFFFFFFFFFFFF).value
        handle = _kernel32.OpenFileById(
            hint, ctypes.byref(descriptor), _FILE_READ_ATTRIBUTES, _SHARE_ALL, None, _FILE_FLAG_BACKUP_SEMANTICS
        )
        if handle in (None, _INVALID_HANDLE):
            return NOT_FOUND if ctypes.get_last_error() in _GONE else UNSUPPORTED
        try:
            size = 1024
            while True:
                buffer = ctypes.create_unicode_buffer(size)
                length = _kernel32.GetFinalPathNameByHandleW(handle, buffer, size, 0)
                if length == 0:
                    return UNSUPPORTED
                if length < size:
                    return _dos_path(buffer.value)
                size = length + 1
        finally:
            _kernel32.CloseHandle(handle)
    finally:
        _kernel32.CloseHandle(hint)


def in_recycle_bin(path: str) -> bool:
    """``D:\\$Recycle.Bin\\S-1-5-…\\$R4X2.pdf`` (or anything below a recycled folder)."""
    parts = Path(path).parts
    return len(parts) > 1 and parts[1].lower() == _RECYCLE_BIN
