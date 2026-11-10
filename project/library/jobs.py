"""The indexing queue: files the user is waiting on first, then the cheapest; slow jobs
(OCR, vision) capped so they never hold up everything else.

Several workers take jobs from one queue. A job's *kind* decides whether it may start:

* ``light`` -- no recognition (text PDFs, Word, Excel, Markdown, images read by name):
  always, up to the number of workers.
* ``ocr``   -- recognition that ends soon (an image, a PDF with a few scanned pages):
  at most ``ocr_jobs`` of these and ``long`` ones together.
* ``long``  -- recognition of many pages (a scanned PDF): at most ``long_jobs`` at once.

Its *cost* (estimated seconds) orders the queue: a 3-second image is not stuck behind a
5-minute scan that is still waiting.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import heapq
import itertools
import os
from pathlib import Path
import threading
from typing import Any, Callable

from ingestion.images import is_image

LIGHT, OCR, LONG = "light", "ocr", "long"
# A PDF with at least this many scanned pages to recognise is a long job.
LONG_PAGES = 8
# Seconds, for ordering only (design/LIBRARY_SYNC_AND_IMAGES.md §6): recognition per scanned
# page and per image, parsing + embedding per MB of text documents.
SECONDS_PER_PAGE = {"local": 2.3, "cloud": 10.0}
SECONDS_PER_IMAGE = {"local": 2.0, "cloud": 25.0, "name": 0.6}
SECONDS_PER_MB = 2.0
PROBE_PAGES = 12


@dataclass(order=True)
class Job:
    rank: int                                  # 0: the user is waiting on it; 1: everything else
    cost: float
    seq: int
    file_id: str = field(compare=False)
    kind: str = field(compare=False, default=LIGHT)


class JobQueue:
    """A priority queue of file ids with set-like membership (``in``, ``discard``)."""

    def __init__(self) -> None:
        self._heap: list[Job] = []
        self._jobs: dict[str, Job] = {}
        self._seq = itertools.count()

    def add(self, file_id: str, cost: float, kind: str = LIGHT, urgent: bool = False) -> bool:
        """Queue a file, or move it forward when it becomes urgent. True when anything changed."""
        rank = 0 if urgent else 1
        current = self._jobs.get(file_id)
        if current is not None and current.rank <= rank:
            return False
        job = Job(rank, float(cost), next(self._seq), file_id, kind)
        self._jobs[file_id] = job
        heapq.heappush(self._heap, job)
        return True

    def pop(self, allowed: Callable[[Job], bool] = lambda job: True) -> Job | None:
        """The best job that may start now (the others keep their place)."""
        skipped: list[Job] = []
        found = None
        while self._heap:
            job = heapq.heappop(self._heap)
            if self._jobs.get(job.file_id) is not job:
                continue  # replaced (moved forward) or discarded
            if not allowed(job):
                skipped.append(job)
                continue
            found = job
            del self._jobs[job.file_id]
            break
        for job in skipped:
            heapq.heappush(self._heap, job)
        return found

    def discard(self, file_id: str) -> None:
        self._jobs.pop(file_id, None)

    def kinds(self) -> Counter:
        return Counter(job.kind for job in self._jobs.values())

    def __contains__(self, file_id: object) -> bool:
        return file_id in self._jobs

    def __len__(self) -> int:
        return len(self._jobs)

    def __bool__(self) -> bool:
        return bool(self._jobs)


class Slots:
    """Which kinds may start, given the jobs running now."""

    def __init__(self, ocr_jobs: int = 2, long_jobs: int = 1):
        self.ocr_jobs = max(1, int(ocr_jobs))
        self.long_jobs = max(1, min(int(long_jobs), self.ocr_jobs))

    def allows(self, kind: str, running: Counter) -> bool:
        if kind == LIGHT:
            return True
        busy = running[OCR] + running[LONG]
        if kind == LONG:
            return running[LONG] < self.long_jobs and busy < self.ocr_jobs
        return busy < self.ocr_jobs


_probe_cache: dict[tuple[str, int, float], tuple[int, int]] = {}
_probe_lock = threading.Lock()


def scanned_pages(path: str) -> tuple[int, int]:
    """(pages, pages without a text layer) of a PDF, from a sample of up to PROBE_PAGES pages.
    Cheap (no rendering); cached per file version."""
    try:
        info = os.stat(path)
    except OSError:
        return 0, 0
    key = (path, int(info.st_size), float(info.st_mtime))
    with _probe_lock:
        if key in _probe_cache:
            return _probe_cache[key]
    try:
        import pymupdf

        with pymupdf.open(path) as document:
            total = document.page_count
            step = max(1, total // PROBE_PAGES)
            sample = list(range(0, total, step))[:PROBE_PAGES]
            textless = sum(1 for n in sample if len(document[n].get_text().strip()) < 5)
        result = (total, round(total * textless / max(1, len(sample))))
    except Exception:
        result = (0, 0)
    with _probe_lock:
        if len(_probe_cache) > 5000:
            _probe_cache.clear()
        _probe_cache[key] = result
    return result


def estimate(record: dict[str, Any], pdf_mode: str) -> tuple[float, str]:
    """(seconds, kind) of indexing a registry record. ``pdf_mode`` is how this PDF's scanned
    pages and figures are read ("off" | "local" | "cloud")."""
    path = record["path"]
    megabytes = (record.get("size") or 0) / 1e6
    if is_image(path):
        mode = record.get("image_mode") or "local"
        return SECONDS_PER_IMAGE.get(mode, 2.0), LIGHT if mode == "name" else OCR
    base = 0.3 + megabytes * SECONDS_PER_MB
    if Path(path).suffix.lower() != ".pdf" or pdf_mode not in SECONDS_PER_PAGE:
        return base, LIGHT
    _, scanned = scanned_pages(path)
    cost = base + scanned * SECONDS_PER_PAGE[pdf_mode]
    if scanned >= LONG_PAGES:
        return cost, LONG
    if scanned or (pdf_mode == "cloud" and megabytes > 2):
        return cost, OCR
    return base, LIGHT
