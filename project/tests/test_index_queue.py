"""Indexing side by side: cheap and urgent files first, slow recognition capped, a long scan
committed as it is read, and model calls retried when the connection drops."""

from collections import Counter
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import httpx
import openai
import pymupdf
import pytest

from db.parent_store_manager import ParentStoreManager
from library import FileRegistry, LibraryService
from library import jobs
from library.jobs import LIGHT, LONG, OCR, JobQueue, Slots, estimate
from services import model_clients


# ---------------------------------------------------------------- the queue


def test_queue_takes_urgent_then_cheapest_and_keeps_skipped_jobs_in_place():
    queue = JobQueue()
    queue.add("big-scan", 140, LONG)
    queue.add("photo", 2, OCR)
    queue.add("notes", 0.5, LIGHT)
    queue.add("report", 6, LIGHT)
    assert queue.pop().file_id == "notes"
    # Opened by the user: goes first even though it costs more.
    assert queue.add("report", 6, LIGHT, urgent=True) is True
    assert queue.add("report", 6, LIGHT) is False  # never demoted
    assert queue.pop().file_id == "report"
    # Recognition slots full: the light work behind it is taken, the scan keeps its place.
    queue.add("readme", 9, LIGHT)
    assert queue.pop(lambda job: job.kind == LIGHT).file_id == "readme"
    assert queue.pop().file_id == "photo"
    queue.discard("big-scan")
    assert "big-scan" not in queue and len(queue) == 0 and queue.pop() is None


def test_slots_cap_long_and_all_recognition_jobs():
    slots = Slots(ocr_jobs=2, long_jobs=1)
    assert slots.allows(LIGHT, Counter({LONG: 1, OCR: 1}))
    assert slots.allows(OCR, Counter({LONG: 1})) and not slots.allows(LONG, Counter({LONG: 1}))
    assert not slots.allows(OCR, Counter({LONG: 1, OCR: 1}))
    assert slots.allows(LONG, Counter({OCR: 1})) and not slots.allows(LONG, Counter({OCR: 2}))


def _scanned_pdf(path: Path, pages: int, text_pages: int = 0) -> Path:
    document = pymupdf.open()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pixmap.clear_with(200)
    for number in range(pages):
        page = document.new_page()
        if number < text_pages:
            page.insert_text((72, 72), f"Page {number} has a text layer.", fontsize=12)
        else:
            page.insert_image(page.rect, pixmap=pixmap)
    document.save(path)
    return path


def test_estimate_orders_by_cost_and_tells_scans_from_text(tmp_path):
    scan = _scanned_pdf(tmp_path / "scan.pdf", 20)
    text = _scanned_pdf(tmp_path / "text.pdf", 6, text_pages=6)
    few = _scanned_pdf(tmp_path / "few.pdf", 10, text_pages=8)
    record = lambda path, **extra: {"path": str(path), "size": path.stat().st_size, **extra}
    assert jobs.scanned_pages(str(scan)) == (20, 20)
    assert estimate(record(scan), "local")[1] == LONG
    assert estimate(record(scan), "off")[1] == LIGHT  # not read: nothing slow about it
    assert estimate(record(text), "local")[1] == LIGHT
    assert estimate(record(few), "local")[1] == OCR
    assert estimate(record(scan), "local")[0] > estimate(record(few), "local")[0] > estimate(record(text), "local")[0]
    image = {"path": str(tmp_path / "a.jpg"), "size": 5_000_000}
    assert estimate({**image, "image_mode": "local"}, "off") == (jobs.SECONDS_PER_IMAGE["local"], OCR)
    assert estimate({**image, "image_mode": "name"}, "off")[1] == LIGHT


# ---------------------------------------------------------------- several workers


class SlowIndexer:
    """Sleeps per file by name ("scan" takes long); records how many ran at once, per kind."""

    def __init__(self, store):
        self.store = store
        self.lock = threading.Lock()
        self.running: Counter = Counter()
        self.peak: Counter = Counter()
        self.finished: list[tuple[str, float]] = []

    def __call__(self, path, file_id, source, image_mode=None, pause=None, progress=None):
        kind = "scan" if "scan" in source else "photo" if source.endswith(".png") else "text"
        with self.lock:
            self.running[kind] += 1
            self.peak[kind] = max(self.peak[kind], self.running[kind])
            self.peak["all"] = max(self.peak["all"], sum(self.running.values()))
        try:
            if kind == "scan":
                for page in range(1, 11):
                    time.sleep(0.12)
                    if progress:
                        progress(page, 10)
            else:
                time.sleep(0.05)
        finally:
            with self.lock:
                self.running[kind] -= 1
                self.finished.append((source, time.monotonic()))
        self.store.documents[file_id] = source
        return 1, 10, {}


class Store:
    def __init__(self):
        self.documents = {}

    def has_document(self, doc_id):
        return doc_id in self.documents

    def delete_document(self, doc_id):
        self.documents.pop(doc_id, None)

    def list_documents(self):
        return []


@pytest.fixture
def lib(tmp_path):
    settings = SimpleNamespace(
        allowed_extensions=(".pdf", ".md", ".txt", ".png"), max_upload_mb=5, max_local_file_mb=5, max_local_text_mb=5, data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads", parsed_dir=tmp_path / "data" / "parsed",
        parent_store_dir=tmp_path / "data" / "parents", watch_max_files=1000, image_import_mode="local",
        image_understanding="local", vision_concurrency=6, index_workers=4, index_ocr_jobs=2, index_long_jobs=1,
        index_yield_max_seconds=5, library_monitor_seconds=3600, library_sweep_seconds=3600,
    )
    for path in (settings.upload_dir, settings.parsed_dir, settings.parent_store_dir):
        path.mkdir(parents=True)
    store = Store()
    indexer = SlowIndexer(store)
    library = LibraryService(settings, FileRegistry(tmp_path / "data" / "r.sqlite3"), store,
                             ParentStoreManager(settings), indexer)
    library.slow = indexer
    yield library
    library.stop()
    library.registry.close()


def _png(path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    return path


def _wait(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_small_files_are_not_stuck_behind_a_long_scan(lib, tmp_path):
    scans = [lib.import_path(str(_scanned_pdf(tmp_path / f"scan{n}.pdf", 12)))["file_id"] for n in range(2)]
    small = [lib.import_path(str(tmp_path / f"note{n}.md"), source="auto")["file_id"]
             for n in range(6) if (tmp_path / f"note{n}.md").write_text("内容", encoding="utf-8") or True]
    photos = [lib.import_path(str(_png(tmp_path / f"p{n}.png")), source="auto")["file_id"] for n in range(3)]
    started = time.monotonic()
    lib.start(watch=False)
    done = lambda: all(lib.registry.get_file(f)["status"] == "indexed" for f in scans + small + photos)
    assert _wait(done)
    finished = dict(lib.slow.finished)
    first_scan = min(finished[f"scan{n}.pdf"] for n in range(2))
    # Every note and photo was done before the first scan finished: none waited for it.
    assert all(finished[f"note{n}.md"] < first_scan for n in range(6))
    assert all(finished[f"p{n}.png"] < first_scan for n in range(3))
    # One long scan at a time, at most two recognition jobs, never more than four workers.
    assert lib.slow.peak["scan"] == 1 and lib.slow.peak["scan"] + lib.slow.peak["photo"] <= 2 + 1
    assert lib.slow.peak["all"] <= 4
    assert time.monotonic() - started < 5


def test_status_lists_every_file_being_read_with_page_progress(lib, tmp_path):
    scan = lib.import_path(str(_scanned_pdf(tmp_path / "scan.pdf", 12)))["file_id"]
    lib.start(watch=False)
    seen = []

    def has_pages():
        jobs_now = lib.status()["jobs"]
        seen.extend(job.get("pages") for job in jobs_now if job["file_id"] == scan and job.get("pages"))
        return bool(seen)

    assert _wait(has_pages)
    assert seen[0][1] == 10 and 1 <= seen[0][0] <= 10
    assert lib.status()["jobs"][0]["kind"] == LONG


def test_a_file_the_user_opens_jumps_the_queue(lib, tmp_path):
    for n in range(5):
        (tmp_path / f"n{n}.md").write_text("内容", encoding="utf-8")
        lib.import_path(str(tmp_path / f"n{n}.md"), source="auto")
    big = tmp_path / "big.md"
    big.write_text("大" * 200_000, encoding="utf-8")
    record = lib.import_path(str(big), source="auto")
    lib.check_file(record["file_id"])  # shown in the reader
    assert lib.process_next() == "indexed"
    assert lib.slow.finished[0][0] == "big.md"


def test_one_file_is_never_indexed_twice_at_once(lib, tmp_path):
    scan = lib.import_path(str(_scanned_pdf(tmp_path / "scan.pdf", 12)))["file_id"]
    lib.start(watch=False)
    assert _wait(lambda: scan in {job["file_id"] for job in lib.status()["jobs"]})
    lib.enqueue(scan)  # it changed again while being read
    time.sleep(0.4)
    assert lib.slow.peak["scan"] == 1
    assert _wait(lambda: lib.status()["busy"] is False)


# ---------------------------------------------------------------- model calls


def _connection_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "https://api.siliconflow.cn/v1/embeddings"))


def _status_error(code):
    request = httpx.Request("POST", "https://api.siliconflow.cn/v1/embeddings")
    response = httpx.Response(code, request=request, json={"message": "x"})
    cls = {429: openai.RateLimitError, 401: openai.AuthenticationError, 500: openai.InternalServerError}[code]
    return cls("x", response=response, body={"message": "x"})


def test_transient_errors_are_retried_and_bad_requests_are_not():
    assert model_clients.transient_error(_connection_error())
    assert model_clients.transient_error(_status_error(429)) and model_clients.transient_error(_status_error(500))
    assert not model_clients.transient_error(_status_error(401))
    assert not model_clients.transient_error(ValueError("bad json"))


def test_embedding_a_document_survives_a_dropped_connection(monkeypatch):
    settings = SimpleNamespace(embedding_dimension=2, embedding_model="m", siliconflow_base_url="http://x",
                               request_timeout_seconds=5, require_api_key=lambda: "k")
    embeddings = model_clients.SiliconFlowEmbeddings(settings)
    calls = []

    def flaky(texts):
        calls.append(len(texts))
        if len(calls) == 1:
            raise _connection_error()  # "Connection error.": a pooled connection the server had closed
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(embeddings, "_embed_batch", flaky)
    monkeypatch.setattr(model_clients.SiliconFlowEmbeddings._embed_for_index.retry, "sleep", lambda seconds: None)
    assert embeddings.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert calls == [2, 2]

    def refused(texts):
        calls.append("401")
        raise _status_error(401)

    monkeypatch.setattr(embeddings, "_embed_batch", refused)
    with pytest.raises(openai.AuthenticationError):
        embeddings.embed_documents(["a"])
    assert calls[-1] == "401" and calls.count("401") == 1  # a bad key is not hammered


def test_background_calls_are_limited_across_threads():
    limiter = model_clients.CallLimiter(2)
    active, peak = [0], [0]
    lock = threading.Lock()

    def call():
        with limiter.slot():
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak[0] == 2
    limiter.resize(3)
    assert limiter.limit == 3


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows thread priorities")
def test_background_threads_run_below_normal_priority():
    import ctypes

    from ingestion.figures import LocalReader
    from ingestion.priority import BELOW_NORMAL, LOWEST, lower_current_thread

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    seen = []

    def work(level):
        lower_current_thread(level)
        seen.append(kernel32.GetThreadPriority(ctypes.c_void_p(kernel32.GetCurrentThread())))

    for level in (BELOW_NORMAL, LOWEST):
        thread = threading.Thread(target=work, args=(level,))
        thread.start()
        thread.join()
    assert seen == [BELOW_NORMAL, LOWEST]
    engines, threads = LocalReader.layout()
    assert threads == 2 and engines >= 1


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows thread priorities")
def test_an_ocr_engines_own_threads_run_at_the_lowest_priority():
    pytest.importorskip("rapidocr_onnxruntime")
    from rapidocr_onnxruntime import RapidOCR

    from ingestion.priority import lower_new_threads

    started = threading.Event()
    bystander = threading.Thread(target=started.wait)  # a Python thread started meanwhile
    engine, lowered = lower_new_threads(
        lambda: (bystander.start(), RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1))[1]
    )
    started.set()
    bystander.join()
    assert engine is not None and lowered >= 3  # one pool thread per model (detect, classify, read)


# ---------------------------------------------------------------- a scan committed as it is read


def test_long_scan_is_committed_in_batches_and_reuses_vectors(tmp_path, monkeypatch):
    pytest.importorskip("sqlite_vec")
    from config import Settings
    from ingestion.pipeline import IngestionPipeline
    from retrieval.embedded_store import EmbeddedStore

    settings = Settings(_env_file=None, data_dir=tmp_path / "data", embedding_dimension=4, image_understanding="local")
    settings.ensure_directories()
    store = EmbeddedStore(settings)
    store.ensure_collection()

    class PageReader:
        mode, model, workers = "local", "fake", 3

        def read(self, figure):
            return f"第 {figure.page} 页的文字：编号 P{figure.page:03d}"

    class Embeddings:
        def __init__(self):
            self.texts = 0

        def embed_documents(self, texts):
            self.texts += len(texts)
            return [[1.0, 0.0, 0.0, float(len(t))] for t in texts]

    monkeypatch.setattr("ingestion.parsers.build_reader", lambda settings, mode=None: PageReader())
    embeddings = Embeddings()
    pipeline = IngestionPipeline(store, embeddings, settings, parent_store=ParentStoreManager(settings))
    pipeline.ocr_pages_per_commit = 4
    commits = []
    real_commit = pipeline._commit

    def watch(doc_id, document, pause, report):
        result = real_commit(doc_id, document, pause, report)
        commits.append(store.chunk_counts().get(doc_id, 0))
        return result

    pipeline._commit = watch
    pdf = _scanned_pdf(tmp_path / "scan.pdf", 10)
    progress, report = [], {}
    doc = "f" * 32
    chunks, _ = pipeline.index_document(pdf, doc, "scan.pdf", progress=lambda done, total: progress.append((done, total)), report=report)
    assert report["commits"] == 3  # pages 1-4, 5-8, 9-10
    assert commits[0] > 0  # searchable after the first batch
    assert progress[-1] == (10, 10) and progress[0][1] == 10
    assert embeddings.texts == report["embedded"] and report["reused"] > 0  # earlier pages kept their vectors
    text = " ".join(row[0] for row in store._conn.execute("SELECT text FROM chunks WHERE doc_id = ?", (doc,)))
    assert "P001" in text and "P010" in text and chunks == commits[-1]
    store.close()
