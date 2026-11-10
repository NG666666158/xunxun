"""Walk watched folders with the default exclusion rules and estimate indexing work."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import stat as stat_module
from typing import Callable, Iterator
import zipfile

from ingestion.images import is_image
from ingestion.office import PPT_NEEDS_POWERPOINT
from ingestion.parsers import TEXT_LIKE
from ingestion.slides import text_size_estimate

EXCLUDED_DIR_NAMES = frozenset(
    name.lower()
    for name in (
        ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv",
        "$recycle.bin", "system volume information",
    )
)
_HIDDEN_ATTRIBUTES = getattr(stat_module, "FILE_ATTRIBUTE_HIDDEN", 2) | getattr(stat_module, "FILE_ATTRIBUTE_SYSTEM", 4)


@dataclass
class FileFacts:
    path: str
    size: int
    mtime: float
    volume_id: int
    inode: int


@dataclass
class ScanEntry:
    facts: FileFacts
    supported: bool
    reason: str = ""


@dataclass
class ScanSummary:
    files: int = 0
    indexable: int = 0
    unsupported: int = 0
    too_large: int = 0
    bytes: int = 0
    indexable_bytes: int = 0
    estimated_characters: int = 0
    estimated_figures: int = 0
    # Image files among the indexable ones (read by OCR / a vision model, not by characters).
    images: int = 0
    image_bytes: int = 0
    truncated: bool = False
    by_type: dict[str, int] = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def file_facts(path: str | os.PathLike[str]) -> FileFacts:
    """``os.stat`` facts; on Windows st_dev/st_ino are the volume serial and NTFS file ID.

    ``DirEntry.stat()`` leaves both at zero on Windows, so always stat the path.
    """
    info = os.stat(path)
    return FileFacts(os.fspath(path), int(info.st_size), float(info.st_mtime), int(info.st_dev), int(info.st_ino))


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def volume_available(path: str, volume_id: int | None = None) -> bool:
    """Whether the drive/volume holding ``path`` is mounted (and is the same volume).

    A drive letter can come back as a different USB stick; when the recorded volume
    serial is known, the mounted volume must match it.
    """
    anchor = Path(path).anchor
    if not anchor:
        return True
    try:
        info = os.stat(anchor)
    except OSError:
        return False
    return volume_id is None or os.name != "nt" or int(info.st_dev) == int(volume_id)


def _hidden(entry: os.DirEntry) -> bool:
    if entry.name.startswith("."):
        return True
    try:
        attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return True
    return bool(attributes & _HIDDEN_ATTRIBUTES)


def excluded_dir(entry: os.DirEntry) -> bool:
    return entry.name.lower() in EXCLUDED_DIR_NAMES or _hidden(entry) or entry.is_symlink()


def excluded_file(entry: os.DirEntry) -> bool:
    # "~$报告.docx" is Office's lock file for an open document.
    return entry.name.startswith("~$") or _hidden(entry) or entry.is_symlink()


def path_excluded(path: str, root: str) -> bool:
    """Same rules for a single path reported by the watcher (no DirEntry available)."""
    try:
        relative = Path(path).relative_to(root)
    except ValueError:
        return True
    parts = relative.parts
    for directory in parts[:-1]:
        if directory.startswith(".") or directory.lower() in EXCLUDED_DIR_NAMES:
            return True
    name = parts[-1] if parts else ""
    return not name or name.startswith(".") or name.startswith("~$")


@dataclass(frozen=True)
class SizeLimits:
    """How big a file registered in place may be, in bytes. Text files have their own, lower
    limit: everything in them gets embedded, where most of a big PDF or deck is pictures."""

    files: int
    text: int

    def for_suffix(self, suffix: str) -> int:
        return self.text if suffix in TEXT_LIKE else self.files


def classify(path: str, size: int, extensions: tuple[str, ...], max_bytes: "int | SizeLimits") -> tuple[bool, str]:
    suffix = Path(path).suffix.lower()
    if suffix not in extensions:
        if suffix == ".ppt":
            return False, PPT_NEEDS_POWERPOINT
        return False, f"不支持的格式：{suffix or '(无扩展名)'}"
    limit = max_bytes.for_suffix(suffix) if isinstance(max_bytes, SizeLimits) else max_bytes
    if size > limit:
        return False, f"超过 {limit // (1024 * 1024)} MB 大小上限"
    if size == 0:
        return False, "空文件"
    return True, ""


def walk(
    root: str,
    extensions: tuple[str, ...],
    max_bytes: "int | SizeLimits",
    max_files: int | None = None,
    skip: Callable[[str, bool], bool] | None = None,
) -> Iterator[ScanEntry]:
    """Yield every non-excluded file below ``root``; unreadable folders are skipped.

    ``skip(path, is_folder)`` adds the user's exclusions (library/privacy.py): a skipped
    folder is not entered at all, a skipped file is not even stat-ed."""
    stack = [root]
    seen = 0
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name.lower())
        except OSError:
            continue
        subdirectories = []
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if not excluded_dir(entry) and not (skip and skip(entry.path, True)):
                        subdirectories.append(entry.path)
                    continue
                if not entry.is_file(follow_symlinks=False) or excluded_file(entry):
                    continue
                if skip and skip(entry.path, False):
                    continue
                facts = file_facts(entry.path)
            except OSError:
                continue
            supported, reason = classify(entry.path, facts.size, extensions, max_bytes)
            yield ScanEntry(facts, supported, reason)
            seen += 1
            if max_files is not None and seen >= max_files:
                return
        stack.extend(reversed(subdirectories))


def estimate_characters(path: str, size: int) -> int:
    """Cheap estimate of extractable characters, without running the full parser."""
    suffix = Path(path).suffix.lower()
    try:
        if suffix in (".txt", ".md"):
            with open(path, "rb") as handle:
                return len(handle.read().decode("utf-8", errors="ignore"))
        if suffix == ".pdf":
            import pymupdf

            with pymupdf.open(path) as pdf:
                return pdf.page_count * 600
        if suffix in (".docx", ".xlsx"):
            # The main XML part's size, minus markup overhead (~5 bytes of XML per character).
            part = "word/document.xml" if suffix == ".docx" else "xl/sharedStrings.xml"
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo(part) if part in archive.namelist() else None
                xml_bytes = info.file_size if info else sum(item.file_size for item in archive.infolist())
            return xml_bytes // 5
        if suffix == ".pptx":
            return text_size_estimate(path)
        if suffix == ".ppt":
            return size // 4000  # binary: 8 thousand characters in a real 38 MB deck
    except Exception:
        pass
    return size // 3


_CAPTION = re.compile(r"(?m)^\s*(?:图|Figure|Fig\.)\s*\d+(?:\s*[-.．]\s*\d+)?")


def estimate_figures(path: str) -> int:
    """Figures (captioned) plus text-less pages of a PDF: what image understanding would read."""
    try:
        import pymupdf

        with pymupdf.open(path) as pdf:
            count = 0
            for page in pdf:
                text = page.get_text()
                count += len(_CAPTION.findall(text)) if len(text.strip()) >= 5 else 1
            return count
    except Exception:
        return 0


def summarize(
    root: str,
    extensions: tuple[str, ...],
    max_bytes: "int | SizeLimits",
    max_files: int,
    sample_limit: int = 8,
    count_figures: bool = False,
    skip: Callable[[str, bool], bool] | None = None,
) -> ScanSummary:
    summary = ScanSummary()
    for entry in walk(root, extensions, max_bytes, max_files=max_files + 1, skip=skip):
        if summary.files >= max_files:
            summary.truncated = True
            break
        summary.files += 1
        summary.bytes += entry.facts.size
        if entry.supported:
            summary.indexable += 1
            summary.indexable_bytes += entry.facts.size
            if is_image(entry.facts.path):
                summary.images += 1
                summary.image_bytes += entry.facts.size
            else:
                summary.estimated_characters += estimate_characters(entry.facts.path, entry.facts.size)
            if count_figures and entry.facts.path.lower().endswith(".pdf"):
                summary.estimated_figures += estimate_figures(entry.facts.path)
            suffix = Path(entry.facts.path).suffix.lower()
            summary.by_type[suffix] = summary.by_type.get(suffix, 0) + 1
            if len(summary.samples) < sample_limit:
                summary.samples.append(os.path.relpath(entry.facts.path, root))
        else:
            summary.unsupported += 1
            if "大小上限" in entry.reason:
                summary.too_large += 1
    return summary
