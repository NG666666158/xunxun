"""Original-file preview: which files may be read, renderer choice, Word → PDF cache and timeout."""

import json
import os
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import main
from core.rag_system import RAGSystem
from library import FileRegistry
from library.preview import (
    ConversionTimeout,
    PreviewError,
    PreviewPrefetcher,
    WordPdfConverter,
    WordUnavailable,
    preview_kind,
    preview_plan,
    readable_original,
)


# ---------------- renderer choice ----------------


@pytest.mark.parametrize(
    ("ext", "word", "renderer", "fallback"),
    [
        (".pdf", False, "pdfjs", "external"),
        (".PDF", True, "pdfjs", "external"),
        (".docx", True, "word-pdf", "docx-preview"),
        (".docx", False, "docx-preview", "docx-preview"),
        (".doc", True, "word-pdf", "external"),
        (".doc", False, "external", "external"),
        (".xlsx", False, "sheetjs", "external"),
        (".md", False, "markdown", "external"),
        (".txt", False, "text", "external"),
        (".png", False, "image", "external"),
        (".jpeg", False, "image", "external"),
        (".pptx", True, "slides", "slides"),
        (".ppt", True, "external", "external"),
        ("", False, "external", "external"),
    ],
)
def test_preview_plan_picks_renderer_by_format(ext, word, renderer, fallback):
    plan = preview_plan(ext, True, word)
    assert (plan["renderer"], plan["fallback"]) == (renderer, fallback)
    assert plan["kind"] == preview_kind(ext)


@pytest.mark.parametrize("ext", [".pdf", ".docx", ".xlsx", ".md", ".pptx"])
def test_missing_original_falls_back_to_parsed_text(ext):
    assert preview_plan(ext, False, True)["renderer"] == "parsed"


# ---------------- which files may be read ----------------


@pytest.fixture
def registry(tmp_path):
    reg = FileRegistry(tmp_path / "registry.sqlite3")
    yield reg
    reg.close()


def _file(path, text="内容"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_readable_original_only_inside_live_watched_folders(registry, tmp_path):
    watched = tmp_path / "watched"
    inside = _file(watched / "sub" / "a.md")
    outside = _file(tmp_path / "elsewhere" / "secret.md")
    root = registry.add_root(str(watched), None)
    ok = registry.insert_file(str(inside), root_id=root["root_id"])
    # A record pointing outside every watched folder (e.g. a tampered or stale registry).
    stray = registry.insert_file(str(outside), root_id=root["root_id"])
    uploads = tmp_path / "uploads"

    assert readable_original(ok, registry, uploads) == inside
    assert readable_original(stray, registry, uploads) is None
    assert readable_original(None, registry, uploads) is None

    registry.soft_delete_root(root["root_id"])
    assert readable_original(registry.get_file(ok["file_id"]), registry, uploads) is None


def test_readable_original_needs_the_file_and_a_live_record(registry, tmp_path):
    watched = tmp_path / "watched"
    path = _file(watched / "a.md")
    root = registry.add_root(str(watched), None)
    record = registry.insert_file(str(path), root_id=root["root_id"])
    assert readable_original({**record, "deleted_at": "2026-01-01"}, registry, tmp_path / "u") is None
    path.unlink()
    assert readable_original(record, registry, tmp_path / "u") is None


def test_readable_original_uploads_must_stay_in_upload_dir(registry, tmp_path):
    uploads = tmp_path / "uploads"
    good = _file(uploads / "f1" / "a.md")
    bad = _file(tmp_path / "other" / "b.md")
    good_record = registry.insert_file(str(good), root_id=None, origin="upload")
    bad_record = registry.insert_file(str(bad), root_id=None, origin="upload")
    assert readable_original(good_record, registry, uploads) == good
    assert readable_original(bad_record, registry, uploads) is None


def test_readable_original_single_files_while_they_are_the_file_added(registry, tmp_path):
    added = _file(tmp_path / "桌面" / "笔记.md")
    info = os.stat(added)
    record = registry.insert_file(str(added), root_id=None, origin="import", volume_id=info.st_dev, inode=info.st_ino)
    assert readable_original(record, registry, tmp_path / "u") == added
    # A row whose identity does not match the file at that path (stale, or edited) reads nothing.
    assert readable_original({**record, "inode": info.st_ino + 1}, registry, tmp_path / "u") is None
    assert readable_original({**record, "inode": None}, registry, tmp_path / "u") is None
    # Only rows the user added themselves: a "watch" row outside every folder stays closed.
    assert readable_original({**record, "origin": "watch"}, registry, tmp_path / "u") is None


def test_image_files_open_in_the_reader_and_tiff_is_drawn_as_png(tmp_path, monkeypatch):
    from PIL import Image

    tiff = tmp_path / "扫描.tif"
    Image.new("RGB", (3000, 2000), (250, 250, 250)).save(tiff)
    png = tmp_path / "截图.png"
    Image.new("RGB", (40, 30), (0, 0, 0)).save(png)
    assert preview_plan(".tif", True, False)["renderer"] == "image"
    monkeypatch.setattr(main, "_original_or_404", lambda doc_id: tiff if doc_id == "t" else png)
    drawn = main.document_image("t")
    assert drawn.media_type == "image/png" and drawn.body[:8] == b"\x89PNG\r\n\x1a\n"
    assert max(Image.open(__import__("io").BytesIO(drawn.body)).size) == 2400
    assert main.document_image("p").path == png  # shown as it is


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are Windows-only")
def test_readable_original_rejects_links_out_of_the_folder(registry, tmp_path):
    import subprocess

    watched = tmp_path / "watched"
    watched.mkdir()
    secret = _file(tmp_path / "private" / "s.md")
    link = watched / "link"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(secret.parent)], check=True, capture_output=True)
    root = registry.add_root(str(watched), None)
    record = registry.insert_file(str(link / "s.md"), root_id=root["root_id"])
    assert readable_original(record, registry, tmp_path / "u") is None


def _settings(tmp_path):
    settings = SimpleNamespace(
        parsed_dir=tmp_path / "parsed",
        parent_store_dir=tmp_path / "parents",
        upload_dir=tmp_path / "uploads",
    )
    for path in (settings.parsed_dir, settings.upload_dir):
        path.mkdir()
    return settings


def _parsed(settings, doc_id, source, file_type):
    (settings.parsed_dir / f"{doc_id}.json").write_text(
        json.dumps({"doc_id": doc_id, "source": source, "file_type": file_type, "units": []}),
        encoding="utf-8",
    )


def test_file_endpoints_refuse_registered_files_outside_watched_folders(monkeypatch, registry, tmp_path):
    settings = _settings(tmp_path)
    watched = tmp_path / "watched"
    inside = _file(watched / "in.md")
    outside = _file(tmp_path / "outside.md")
    root = registry.add_root(str(watched), None)
    good = registry.insert_file(str(inside), root_id=root["root_id"])
    bad = registry.insert_file(str(outside), root_id=root["root_id"])
    for record in (good, bad):
        _parsed(settings, record["file_id"], record["name"], ".md")
    rag = RAGSystem(settings)
    rag.library = SimpleNamespace(registry=registry)
    monkeypatch.setattr(main, "system", rag)

    assert main.download_document(good["file_id"]).path == inside
    assert main.get_document(good["file_id"])["preview"]["renderer"] == "markdown"
    with pytest.raises(HTTPException) as error:
        main.download_document(bad["file_id"])
    assert error.value.status_code == 404
    assert main.get_document(bad["file_id"])["preview"]["renderer"] == "parsed"
    with pytest.raises(HTTPException) as error:
        main.document_as_pdf(bad["file_id"])
    assert error.value.status_code == 404


def test_pdf_endpoint_only_converts_word_and_reports_unavailable_word(monkeypatch, registry, tmp_path):
    settings = _settings(tmp_path)
    watched = tmp_path / "watched"
    root = registry.add_root(str(watched), None)
    md = registry.insert_file(str(_file(watched / "a.md")), root_id=root["root_id"])
    docx = registry.insert_file(str(_file(watched / "b.docx")), root_id=root["root_id"])
    for record, ext in ((md, ".md"), (docx, ".docx")):
        _parsed(settings, record["file_id"], record["name"], ext)
    rag = RAGSystem(settings)
    rag.library = SimpleNamespace(registry=registry)
    monkeypatch.setattr(main, "system", rag)
    monkeypatch.setattr(main, "_word_converter", WordPdfConverter(tmp_path / "cache", available=lambda: False))

    with pytest.raises(HTTPException) as error:
        main.document_as_pdf(md["file_id"])
    assert error.value.status_code == 400
    with pytest.raises(HTTPException) as error:
        main.document_as_pdf(docx["file_id"])
    assert error.value.status_code == 409
    assert main.get_document(docx["file_id"])["preview"]["renderer"] == "docx-preview"


# ---------------- Word → PDF ----------------


class FakeWord:
    """Engine stand-in: writes a PDF, optionally slowly, and records how many run at once."""

    def __init__(self, delay=0.0, fail=False, pid=4321):
        self.delay, self.fail, self.pid = delay, fail, pid
        self.calls = []
        self.running = 0
        self.max_running = 0
        self._lock = threading.Lock()

    def __call__(self, source, target, on_pid):
        with self._lock:
            self.calls.append(source)
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            on_pid(self.pid)
            time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("Word said no")
            with open(target, "wb") as f:
                f.write(b"%PDF-1.7 fake")
        finally:
            with self._lock:
                self.running -= 1


def _converter(tmp_path, engine, **kwargs):
    # FakeWord's pids are made up: a test must never kill a real process.
    kwargs.setdefault("kill", lambda pid: None)
    return WordPdfConverter(tmp_path / "cache", engine=engine, available=lambda: True, **kwargs)


def test_conversion_is_cached_by_path_mtime_and_size(tmp_path):
    doc = _file(tmp_path / "报告.docx", "v1")
    engine = FakeWord()
    converter = _converter(tmp_path, engine)

    pdf, hit = converter.convert(doc)
    assert not hit and pdf.read_bytes().startswith(b"%PDF")
    assert converter.convert(doc) == (pdf, True)
    assert len(engine.calls) == 1

    # A new version converts again and replaces the old cache entry.
    doc.write_text("version two", encoding="utf-8")
    os.utime(doc, (time.time() + 5, time.time() + 5))
    second, hit = converter.convert(doc)
    assert not hit and second != pdf
    assert len(engine.calls) == 2
    assert [p.name for p in (tmp_path / "cache").iterdir()] == [second.name]


def test_same_name_in_another_folder_has_its_own_cache_entry(tmp_path):
    a = _file(tmp_path / "a" / "x.docx", "same")
    b = _file(tmp_path / "b" / "x.docx", "same")
    os.utime(b, (os.stat(a).st_atime, os.stat(a).st_mtime))
    assert WordPdfConverter.cache_name(a) != WordPdfConverter.cache_name(b)


def test_conversion_timeout_kills_word(tmp_path):
    doc = _file(tmp_path / "slow.docx")
    killed = []
    engine = FakeWord(delay=1.0, pid=999)
    converter = _converter(tmp_path, engine, timeout=0.2, kill=killed.append)

    started = time.monotonic()
    with pytest.raises(ConversionTimeout) as error:
        converter.convert(doc)
    assert time.monotonic() - started < 3
    assert killed == [999]
    assert error.value.status == 504
    time.sleep(1.0)  # the fake engine finishes on its own; nothing half-written is kept as a hit
    assert converter.cached(doc) is None


def test_failed_conversion_is_remembered_for_that_version(tmp_path):
    doc = _file(tmp_path / "broken.docx")
    engine = FakeWord(fail=True)
    converter = _converter(tmp_path, engine)
    for _ in range(2):
        with pytest.raises(PreviewError):
            converter.convert(doc)
    assert len(engine.calls) == 1
    assert not list((tmp_path / "cache").glob("*.part.pdf"))


def test_conversions_run_one_at_a_time(tmp_path):
    docs = [_file(tmp_path / f"d{i}.docx", str(i)) for i in range(3)]
    engine = FakeWord(delay=0.1)
    converter = _converter(tmp_path, engine)
    # Two requests for the same file plus two others, all at once.
    threads = [threading.Thread(target=converter.convert, args=(doc,)) for doc in docs + docs[:1]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert engine.max_running == 1
    assert sorted(engine.calls) == sorted(str(doc) for doc in docs)


def test_without_word_conversion_is_refused_quickly(tmp_path):
    doc = _file(tmp_path / "a.docx")
    converter = WordPdfConverter(tmp_path / "cache", engine=FakeWord(), available=lambda: False)
    with pytest.raises(WordUnavailable):
        converter.convert(doc)


# ---------------- converting ahead of time ----------------


def _prefetcher(tmp_path, engine, files, **kwargs):
    converter = _converter(tmp_path, engine)
    options = {"idle_after": 3600, "tick": 0.05, "pause": 0, **kwargs}
    return PreviewPrefetcher(converter, files.get, **options)


def test_prefetch_converts_word_files_in_the_background(tmp_path):
    files = {"w": _file(tmp_path / "a.docx"), "m": _file(tmp_path / "b.md"), "old": _file(tmp_path / "c.doc")}
    engine = FakeWord(delay=0.05)
    prefetcher = _prefetcher(tmp_path, engine, files)
    try:
        assert prefetcher.request("w") == "queued"
        assert prefetcher.request("w") == "queued"  # once only
        assert prefetcher.request("old") == "queued"
        assert prefetcher.request("m") == "skip"
        assert prefetcher.request("gone") == "skip"
        assert prefetcher.wait(files["w"], 5) is not None
        assert prefetcher.wait(files["old"], 5) is not None
        assert prefetcher.request("w") == "ready"
        assert sorted(engine.calls) == sorted([str(files["w"]), str(files["old"])])
        assert engine.max_running == 1 and prefetcher.pending() == 0
    finally:
        prefetcher.stop()


def test_a_file_being_opened_jumps_the_queue(tmp_path):
    files = {k: _file(tmp_path / f"{k}.docx", k) for k in "abcd"}
    engine = FakeWord(delay=0.15)
    prefetcher = _prefetcher(tmp_path, engine, files)
    try:
        for key in "abc":
            prefetcher.request(key)
        time.sleep(0.05)  # "a" is converting
        prefetcher.request("d", urgent=True)
        prefetcher.request("c", urgent=True)
        assert prefetcher.wait(files["b"], 5) is not None
        order = [os.path.basename(c) for c in engine.calls]
        assert order[0] == "a.docx" and order.index("c.docx") < order.index("d.docx") < order.index("b.docx")
    finally:
        prefetcher.stop()


def test_idle_time_converts_recently_added_word_files(tmp_path):
    files = {"new": _file(tmp_path / "new.docx"), "pdf": _file(tmp_path / "x.pdf")}
    engine = FakeWord()
    prefetcher = _prefetcher(tmp_path, engine, files, recent=lambda: ["pdf", "new"], idle_after=0)
    try:
        prefetcher.start()
        deadline = time.monotonic() + 5
        while not prefetcher.converter.cached(files["new"]) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert prefetcher.converter.cached(files["new"]) is not None
        assert engine.calls == [str(files["new"])]
    finally:
        prefetcher.stop()


def test_background_failures_are_not_retried_in_the_background(tmp_path):
    files = {"bad": _file(tmp_path / "bad.docx"), "slow": _file(tmp_path / "slow.docx")}
    engine = FakeWord(fail=True)
    prefetcher = _prefetcher(tmp_path, engine, files)
    try:
        prefetcher.request("bad")
        prefetcher.wait(files["bad"], 5)
        assert prefetcher.converter.failure(files["bad"])
        assert prefetcher.request("bad") == "skip"
        # A timeout leaves no marker: skipped in the background, tried again when opened.
        prefetcher.converter.timeout = 0.1
        engine.fail, engine.delay = False, 0.5
        prefetcher.request("slow")
        prefetcher.wait(files["slow"], 5)
        time.sleep(0.6)
        assert prefetcher.request("slow") == "skip"
        assert prefetcher.request("slow", urgent=True) == "queued"
    finally:
        prefetcher.stop()


def test_opening_a_word_file_waits_briefly_then_says_it_is_still_preparing(monkeypatch, registry, tmp_path):
    settings = _settings(tmp_path)
    watched = tmp_path / "watched"
    root = registry.add_root(str(watched), None)
    docx = registry.insert_file(str(_file(watched / "长报告.docx")), root_id=root["root_id"])
    _parsed(settings, docx["file_id"], docx["name"], ".docx")
    rag = RAGSystem(settings)
    rag.library = SimpleNamespace(registry=registry)
    monkeypatch.setattr(main, "system", rag)
    engine = FakeWord(delay=0.6)
    monkeypatch.setattr(main, "_word_converter", _converter(tmp_path, engine))
    monkeypatch.setattr(main, "_prefetcher", None)
    try:
        response = main.document_as_pdf(docx["file_id"], wait=0.1)
        assert response.status_code == 202
        assert main.prepare_documents(main.PrepareRequest(doc_ids=[docx["file_id"], docx["file_id"]]))["states"] == {
            docx["file_id"]: "queued"
        }
        ready = main.document_as_pdf(docx["file_id"], wait=3)
        assert ready.media_type == "application/pdf"
        assert len(engine.calls) == 1
    finally:
        main.preview_prefetcher().stop()


def test_unparsed_registered_file_still_opens_its_original(monkeypatch, registry, tmp_path):
    settings = _settings(tmp_path)
    watched = tmp_path / "watched"
    root = registry.add_root(str(watched), None)
    scan = registry.insert_file(str(_file(watched / "扫描报告.pdf")), root_id=root["root_id"], status="failed")
    rag = RAGSystem(settings)
    rag.library = SimpleNamespace(registry=registry)
    monkeypatch.setattr(main, "system", rag)

    document = main.get_document(scan["file_id"])
    assert document["parsed"] is False and document["units"] == []
    assert document["status"] == "failed"
    assert document["preview"]["renderer"] == "pdfjs"
    assert main.download_document(scan["file_id"]).path == watched / "扫描报告.pdf"
    with pytest.raises(HTTPException):
        main.get_document("f" * 32)


def test_citations_carry_a_quote_and_stay_deduplicated():
    from rag_agent.nodes import AgentNodes

    docs = [
        {"source": "a.pdf", "locator": "page:3", "doc_id": "d1", "text": "第一段证据"},
        {"source": "a.pdf", "locator": "page:3", "doc_id": "d1", "text": "同一页的另一段"},
        {"source": "b.pdf", "locator": "page:9", "doc_id": "d2", "text": "x" * 500},
    ]
    _, citations = AgentNodes._build_evidence({"documents": docs})
    assert [(c["source"], c["quote"]) for c in citations] == [("a.pdf", "第一段证据"), ("b.pdf", "x" * 200)]
