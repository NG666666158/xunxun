"""Adding single files and images, and following files and folders that move, appear or disappear."""

import ctypes
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from db.parent_store_manager import ParentStoreManager
from library import FileRegistry, LibraryError, LibraryService
from library import service as service_module
from library import winfs
from library.watcher import EventRouter, ParentRouter

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
# Files and folders are followed by their NTFS file ID; the Recycle Bin and paths are Windows ones.
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="needs NTFS file IDs and Windows paths")


class FakeStore:
    def __init__(self):
        self.documents = {}

    def has_document(self, doc_id):
        return doc_id in self.documents

    def delete_document(self, doc_id):
        self.documents.pop(doc_id, None)

    def rename_source(self, doc_id, source):
        if doc_id in self.documents:
            self.documents[doc_id] = source

    def list_documents(self):
        return [{"doc_id": d, "source": s} for d, s in self.documents.items()]


class Indexer:
    """Takes the library's options like the app's indexer; returns the parser's notice."""

    def __init__(self, store):
        self.store = store
        self.calls = []
        self.notice = None

    def __call__(self, path, file_id, source, image_mode=None, pause=None):
        self.calls.append((Path(path).name, image_mode))
        self.store.documents[file_id] = source
        return 1, 10, {"notice": self.notice}


def _settings(tmp_path, **extra):
    settings = SimpleNamespace(
        allowed_extensions=(".pdf", ".docx", ".xlsx", ".md", ".txt", *IMAGE_EXTENSIONS),
        max_upload_mb=1, max_local_file_mb=1, max_local_text_mb=1,
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        parsed_dir=tmp_path / "data" / "parsed",
        parent_store_dir=tmp_path / "data" / "parents",
        watch_max_files=1000,
        image_import_mode="ask",
        image_understanding="off",
        vision_concurrency=6,
        index_yield_max_seconds=5,
    )
    for key, value in extra.items():
        setattr(settings, key, value)
    for path in (settings.upload_dir, settings.parsed_dir, settings.parent_store_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings


@pytest.fixture
def lib(tmp_path):
    store = FakeStore()
    indexer = Indexer(store)
    library = LibraryService(
        _settings(tmp_path), FileRegistry(tmp_path / "data" / "registry.sqlite3"), store,
        ParentStoreManager(_settings(tmp_path)), indexer,
    )
    library.indexer_calls = indexer.calls
    library.fake_indexer = indexer
    yield library
    library.stop()
    library.registry.close()


def _write(path, text="内容"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _png(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)  # the fake indexer never decodes it
    return path


@pytest.fixture
def desk(tmp_path):
    """A watched-folder candidate with an image, plus loose files elsewhere."""
    _write(tmp_path / "资料" / "制度.md", "# 制度\n内容")
    _png(tmp_path / "资料" / "截图" / "流程.png")
    _write(tmp_path / "桌面" / "笔记.txt", "单个文件")
    _png(tmp_path / "桌面" / "白板.jpg")
    return tmp_path


def _by_name(lib):
    return {record["name"]: record for record in lib.registry.list_files()}


def _settle(lib, timeout=10):
    """Wait for folder scans started in the background (adding a folder returns at once)."""
    deadline = time.time() + timeout
    while lib.status()["scanning"] and time.time() < deadline:
        time.sleep(0.02)


# ---------------------------------------------------------------- inspect & import


def test_inspect_counts_images_and_estimates_without_registering(lib, desk):
    result = lib.inspect([str(desk / "资料"), str(desk / "桌面" / "白板.jpg"), str(desk / "桌面" / "笔记.txt"),
                          str(desk / "nope.md"), "relative.md"])
    kinds = [item["kind"] for item in result["items"]]
    assert kinds == ["folder", "file", "file", "error", "error"]
    totals = result["totals"]
    assert totals["images"] == 2 and totals["indexable"] == 4 and totals["by_type"][".png"] == 1
    assert result["ask_image_mode"] is True and result["default_image_mode"] == "local"
    assert result["estimate"]["images"]["cloud"]["cloud_calls"] == 2
    assert result["estimate"]["images"]["local"]["cloud_calls"] == 0
    assert result["estimate"]["images"]["cloud"]["seconds"] >= result["estimate"]["images"]["name"]["seconds"]
    assert totals["estimated_characters"] < 100  # image bytes are not counted as text
    assert lib.registry.list_files() == [] and lib.registry.list_roots() == []

    lib.settings.image_import_mode = "name"  # "记住我的选择"
    assert lib.inspect([str(desk / "桌面" / "白板.jpg")])["ask_image_mode"] is False


def test_import_registers_files_in_place_and_watches_folders(lib, desk):
    result = lib.import_paths(
        [str(desk / "桌面" / "笔记.txt"), str(desk / "桌面" / "白板.jpg"), str(desk / "资料"), str(desk / "a.exe")],
        image_mode="cloud",
    )
    _settle(lib)
    assert [item["ok"] for item in result["results"]] == [True, True, True, False]
    assert result["added"] == 3
    files = _by_name(lib)
    note, board, flow = files["笔记.txt"], files["白板.jpg"], files["流程.png"]
    assert note["origin"] == "import" and note["root_id"] is None and note["path"] == str(desk / "桌面" / "笔记.txt")
    assert board["image_mode"] == "cloud" and flow["image_mode"] == "cloud" and note["image_mode"] is None
    assert lib.list_roots()[0]["image_mode"] == "cloud"  # images that appear later are read the same way
    assert not any(lib.settings.upload_dir.iterdir())  # nothing copied
    # Findable at once: registered before the queue has read anything.
    assert {files[name]["status"] for name in ("笔记.txt", "白板.jpg", "流程.png")} == {"pending"}

    lib.run_until_idle()
    assert ("白板.jpg", "cloud") in lib.indexer_calls and ("笔记.txt", None) in lib.indexer_calls

    again = lib.import_paths([str(desk / "桌面" / "笔记.txt"), str(desk / "资料")])
    _settle(lib)
    assert [item.get("already") for item in again["results"]] == [True, True] and again["added"] == 0
    with pytest.raises(LibraryError):
        lib.import_paths([str(desk / "桌面" / "笔记.txt")], image_mode="ocr")


def test_new_images_take_the_folders_choice_then_the_preference(lib, desk):
    root = lib.add_root(str(desk / "资料"), background=False, image_mode="name")
    lib.run_until_idle()
    later = _png(desk / "资料" / "新截图.png")
    lib.handle_path(str(later))
    assert _by_name(lib)["新截图.png"]["image_mode"] == "name"

    lib.settings.image_import_mode = "cloud"
    assert lib.import_path(str(_png(desk / "其他" / "a.png")))["image_mode"] == "cloud"
    lib.settings.image_import_mode = "ask"
    assert lib.import_path(str(_png(desk / "其他" / "b.png")))["image_mode"] == "local"  # nothing uploaded by default
    assert root["image_mode"] == "name"


def test_changing_a_files_image_mode_reads_it_again(lib, desk):
    record = lib.import_path(str(desk / "桌面" / "白板.jpg"), image_mode="local")
    lib.run_until_idle()
    lib.indexer_calls.clear()
    assert lib.set_image_mode(record["file_id"], "cloud")["status"] == "pending"
    assert lib.run_until_idle() == ["indexed"] and lib.indexer_calls == [("白板.jpg", "cloud")]

    text = lib.import_path(str(desk / "桌面" / "笔记.txt"))
    with pytest.raises(LibraryError):
        lib.set_image_mode(text["file_id"], "cloud")


def test_scanned_pdf_notice_is_stored_and_answered(lib, tmp_path):
    lib.fake_indexer.notice = "scanned_pdf"
    record = lib.import_path(str(_write(tmp_path / "扫描.pdf", "%PDF fake")))
    lib.run_until_idle()
    assert lib.registry.get_file(record["file_id"])["notice"] == "scanned_pdf"

    # "不用了": remembered, no re-reading.
    lib.fake_indexer.notice = None
    declined = lib.set_image_mode(record["file_id"], "off")
    assert declined["notice"] is None and declined["image_mode"] == "off" and declined["status"] == "indexed"
    assert lib.run_until_idle() == []
    # "识别": read again with local OCR for this file only.
    assert lib.set_image_mode(record["file_id"], "local")["status"] == "pending"
    lib.run_until_idle()
    assert lib.indexer_calls[-1] == ("扫描.pdf", "local")


def test_images_read_by_name_only_are_read_once_local_ocr_arrives(lib, desk, monkeypatch):
    lib.fake_indexer.notice = "ocr_unavailable"
    record = lib.import_path(str(desk / "桌面" / "白板.jpg"), image_mode="local")
    lib.run_until_idle()
    assert lib.registry.get_file(record["file_id"])["notice"] == "ocr_unavailable"
    monkeypatch.setattr(service_module.ocr_runtime, "activate", lambda data_dir, component="ocr": False)
    lib.sweep()
    assert lib.run_until_idle() == []
    monkeypatch.setattr(service_module.ocr_runtime, "activate", lambda data_dir, component="ocr": True)
    lib.fake_indexer.notice = None
    assert lib.sweep()["ocr_requeued"] == 1
    assert lib.run_until_idle() == ["indexed"] and lib.registry.get_file(record["file_id"])["notice"] is None


# ---------------------------------------------------------------- following single files


@windows_only
def test_moved_single_file_is_followed_by_file_id_without_reindexing(lib, desk):
    record = lib.import_path(str(desk / "桌面" / "笔记.txt"))
    lib.run_until_idle()
    seq = lib.changes()["seq"]
    target = desk / "归档" / "2024" / "笔记-终版.txt"
    target.parent.mkdir(parents=True)
    os.replace(desk / "桌面" / "笔记.txt", target)

    stats = lib.sweep()
    after = lib.registry.get_file(record["file_id"])
    assert stats.get("moved") == 1
    assert after["path"] == str(target) and after["status"] == "indexed" and after["name"] == "笔记-终版.txt"
    assert lib.store.documents[record["file_id"]] == "笔记-终版.txt"  # index renamed in place
    assert lib.run_until_idle() == []
    changes = lib.changes(seq)
    assert changes["counts"] == {"moved": 1} and changes["items"][-1]["old_path"] == str(desk / "桌面" / "笔记.txt")


def test_edited_single_file_is_reindexed_on_sweep_and_check(lib, desk):
    record = lib.import_path(str(desk / "桌面" / "笔记.txt"))
    lib.run_until_idle()
    _write(desk / "桌面" / "笔记.txt", "改过的内容，长度也变了")
    assert lib.sweep().get("changed") == 1
    assert lib.run_until_idle() == ["indexed"]
    os.utime(desk / "桌面" / "笔记.txt", (time.time() + 9, time.time() + 9))
    lib.check_file(record["file_id"])
    assert lib.run_until_idle() == ["skipped"]  # touched only: same content, vectors kept
    assert lib.sync()["elapsed_ms"] >= 0 and lib.sync()["cached"] is True


def test_deleted_single_file_is_missing_and_comes_back(lib, desk):
    record = lib.import_path(str(desk / "桌面" / "笔记.txt"))
    lib.run_until_idle()
    content = (desk / "桌面" / "笔记.txt").read_bytes()
    (desk / "桌面" / "笔记.txt").unlink()
    lib.sweep()
    assert lib.registry.get_file(record["file_id"])["status"] == "missing"
    assert [view["file_id"] for view in lib.lost_files()] == [record["file_id"]]
    (desk / "桌面" / "笔记.txt").write_bytes(content)  # put back (a new file ID)
    lib.sweep()
    assert lib.run_until_idle() == ["skipped"]
    assert lib.registry.get_file(record["file_id"])["status"] == "indexed"


@windows_only
def test_file_moved_out_of_its_watched_folder_is_kept_as_a_single_file(lib, desk):
    lib.add_root(str(desk / "资料"), background=False)
    lib.run_until_idle()
    record = _by_name(lib)["制度.md"]
    os.replace(desk / "资料" / "制度.md", desk / "桌面" / "制度.md")
    lib.handle_path(str(desk / "资料" / "制度.md"))  # the watcher's "deleted"
    after = lib.registry.get_file(record["file_id"])
    assert after["path"] == str(desk / "桌面" / "制度.md") and after["status"] == "indexed"
    assert after["root_id"] is None and after["origin"] == "import"
    assert lib.run_until_idle() == []


# ---------------------------------------------------------------- Recycle Bin


def _fake_bin(monkeypatch, places):
    """resolve_file_id answers from ``places`` (inode -> path / NOT_FOUND)."""
    monkeypatch.setattr(service_module, "resolve_file_id", lambda path, inode: places.get(inode, winfs.UNSUPPORTED))


@windows_only
def test_recycled_file_is_hidden_kept_and_restored_without_reindexing(lib, desk, monkeypatch):
    record = lib.import_path(str(desk / "桌面" / "笔记.txt"))
    lib.run_until_idle()
    original = desk / "桌面" / "笔记.txt"
    binned = desk / "$Recycle.Bin" / "S-1-5-21" / "$R1A2B3C.txt"
    places = {record["inode"]: "C:\\$Recycle.Bin\\S-1-5-21\\$R1A2B3C.txt"}
    _fake_bin(monkeypatch, places)
    binned.parent.mkdir(parents=True)
    os.replace(original, binned)

    lib.sweep()
    recycled = lib.registry.get_file(record["file_id"])
    assert recycled["status"] == "recycled" and recycled["path"] == str(original)  # where it will come back to
    assert record["file_id"] in lib.store.documents and record["file_id"] in lib.hidden_ids()
    assert lib.reindex(record["file_id"])["status"] == "recycled"  # never read from the bin
    assert lib.status()["lost"]["recycled"] == 1
    assert lib.file_view(recycled)["searchable"] is False

    os.replace(binned, original)  # restored
    places[record["inode"]] = str(original)
    lib.sweep()
    assert lib.run_until_idle() == ["skipped"]
    assert lib.registry.get_file(record["file_id"])["status"] == "indexed"
    assert record["file_id"] not in lib.hidden_ids()
    assert lib.changes()["counts"] == {"recycled": 1, "restored": 1}  # the user's own import is not a toast


@windows_only
def test_emptied_from_the_recycle_bin_removes_the_file(lib, desk, monkeypatch):
    record = lib.import_path(str(desk / "桌面" / "笔记.txt"))
    lib.run_until_idle()
    places = {record["inode"]: "C:\\$Recycle.Bin\\S-1\\$R9.txt"}
    _fake_bin(monkeypatch, places)
    (desk / "桌面" / "笔记.txt").unlink()
    lib.sweep()
    assert lib.registry.get_file(record["file_id"])["status"] == "recycled"
    places[record["inode"]] = winfs.NOT_FOUND
    lib.sweep()
    assert lib.registry.get_file(record["file_id"]) is None and record["file_id"] not in lib.store.documents
    assert lib.changes()["counts"]["removed"] == 1


@pytest.mark.skipif(os.name != "nt", reason="the Windows Recycle Bin")
def test_real_recycle_bin_round_trip(lib, tmp_path):
    """Sends a scratch file to the user's Recycle Bin, then puts it back by hand (nothing is left there)."""
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_uint16), ("aborted", wintypes.BOOL),
                    ("mappings", wintypes.LPVOID), ("title", wintypes.LPCWSTR)]

    path = _write(tmp_path / "回收测试.txt", "放进回收站再拿回来")
    record = lib.import_path(str(path))
    lib.run_until_idle()
    # FO_DELETE with FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    operation = SHFILEOPSTRUCTW(None, 3, str(path) + "\0\0", None, 0x40 | 0x10 | 0x4 | 0x400, False, None, None)
    assert ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation)) == 0
    binned = winfs.resolve(str(path), record["inode"])
    try:
        assert winfs.in_recycle_bin(binned)
        assert lib.check_file(record["file_id"])["status"] == "recycled"
    finally:
        # Restore by hand: the $R file back to its place, its $I record removed.
        os.replace(binned, path)
        Path(binned).with_name("$I" + Path(binned).name[2:]).unlink(missing_ok=True)
    lib.check_file(record["file_id"])
    assert lib.run_until_idle() == ["skipped"]
    assert lib.registry.get_file(record["file_id"])["status"] == "indexed"


# ---------------------------------------------------------------- watched folders that move


@windows_only
def test_renamed_watched_folder_is_re_pointed_without_reindexing(lib, desk):
    root = lib.add_root(str(desk / "资料"), background=False)
    lib.run_until_idle()
    before = _by_name(lib)
    seq = lib.changes()["seq"]
    os.rename(desk / "资料", desk / "资料-2024")
    # A stale event from the old path (or the parent watcher) makes the library look again.
    lib.handle_path(str(desk / "资料" / "制度.md"))
    moved = lib.registry.get_root(root["root_id"])
    assert moved["path"] == str(desk / "资料-2024") and moved["status"] == "active"
    after = _by_name(lib)
    assert after["制度.md"]["path"] == str(desk / "资料-2024" / "制度.md")
    assert after["制度.md"]["file_id"] == before["制度.md"]["file_id"] and after["制度.md"]["status"] == "indexed"
    assert lib.run_until_idle() == []
    assert lib.changes(seq)["counts"] == {"folder_moved": 1}


@windows_only
def test_watched_folder_found_again_at_start_up(lib, desk):
    root = lib.add_root(str(desk / "资料"), background=False)
    lib.run_until_idle()
    target = desk / "归档" / "旧资料"
    target.parent.mkdir()
    os.rename(desk / "资料", target)  # while the app was closed
    lib.reconcile()
    assert lib.registry.get_root(root["root_id"])["path"] == str(target)
    assert {r["status"] for r in lib.registry.list_files()} == {"indexed"}
    assert lib.run_until_idle() == []


@windows_only
def test_watched_folder_moved_into_another_one_merges(lib, desk):
    outer = lib.add_root(str(desk / "桌面"), background=False)
    inner = lib.add_root(str(desk / "资料"), background=False)
    lib.run_until_idle()
    os.rename(desk / "资料", desk / "桌面" / "资料")
    lib.check_volumes()
    assert lib.registry.get_root(inner["root_id"]) is None
    assert {r["root_id"] for r in lib.registry.list_files()} == {outer["root_id"]}
    assert _by_name(lib)["制度.md"]["path"] == str(desk / "桌面" / "资料" / "制度.md")
    assert lib.run_until_idle() == []


@windows_only
def test_recycled_watched_folder_and_its_restore(lib, desk, monkeypatch):
    root = lib.add_root(str(desk / "资料"), background=False)
    lib.run_until_idle()
    inode = lib.registry.get_root(root["root_id"])["inode"]
    assert inode == os.stat(desk / "资料").st_ino
    places = {inode: "C:\\$Recycle.Bin\\S-1\\$RAB12"}
    _fake_bin(monkeypatch, places)
    os.rename(desk / "资料", desk / "回收中")
    lib.check_volumes()
    assert lib.registry.get_root(root["root_id"])["status"] == "recycled"
    assert {r["status"] for r in lib.registry.list_files()} == {"recycled"}

    os.rename(desk / "回收中", desk / "资料")
    lib.check_volumes()
    assert lib.registry.get_root(root["root_id"])["status"] == "active"
    lib.run_until_idle()
    assert {r["status"] for r in lib.registry.list_files()} == {"indexed"}
    kinds = lib.changes()["counts"]
    assert kinds["folder_recycled"] == 1 and kinds["folder_restored"] == 1


@windows_only
def test_parent_router_reports_only_watched_folders():
    seen = []
    roots = {os.path.normcase("C:\\x\\资料"): "r1"}
    router = ParentRouter(SimpleNamespace(check_root_by_id=seen.append), lambda: roots,
                          lambda key, action, delay=None: action())
    router.dispatch(SimpleNamespace(event_type="moved", src_path="C:\\x\\资料", dest_path="C:\\x\\资料2", is_directory=True))
    router.dispatch(SimpleNamespace(event_type="modified", src_path="C:\\x\\资料", dest_path="", is_directory=True))
    router.dispatch(SimpleNamespace(event_type="deleted", src_path="C:\\x\\别的", dest_path="", is_directory=True))
    assert seen == ["r1"]


def test_save_by_temp_rename_keeps_the_record_on_the_documents_path(lib, desk):
    root = lib.add_root(str(desk / "资料"), background=False)
    lib.run_until_idle()
    record = _by_name(lib)["制度.md"]
    router = EventRouter(lib, lib.registry.get_root(root["root_id"]), lambda key, action, delay=None: None)
    calls = []
    router.submit = lambda key, action, delay=None: calls.append(key)
    event = SimpleNamespace(event_type="moved", src_path=str(desk / "资料" / "制度.md"),
                            dest_path=str(desk / "资料" / "~WRL0001.tmp"), is_directory=False)
    router.dispatch(event)
    assert [key[0] for key in calls] == ["sync", "sync"]  # both paths re-checked after the save, no move
    assert lib.registry.get_file(record["file_id"])["path"] == str(desk / "资料" / "制度.md")


@pytest.mark.skipif(os.name != "nt", reason="ReadDirectoryChangesW parent watch")
def test_real_watcher_follows_a_renamed_folder_quickly(lib, desk):
    lib.settings.watch_debounce_seconds = 0.2
    lib.settings.library_monitor_seconds = 3600
    root = lib.add_root(str(desk / "资料"), background=False)
    lib.run_until_idle()
    lib.start(watch=True)
    time.sleep(0.3)
    started = time.perf_counter()
    os.rename(desk / "资料", desk / "资料（新）")
    deadline = started + 10
    while time.perf_counter() < deadline:
        if lib.registry.get_root(root["root_id"])["path"] == str(desk / "资料（新）"):
            break
        time.sleep(0.02)
    elapsed = time.perf_counter() - started
    assert lib.registry.get_root(root["root_id"])["path"] == str(desk / "资料（新）")
    assert elapsed < 3, elapsed
    # Watched at its new place: a new file there is picked up.
    _write(desk / "资料（新）" / "新增.md", "# 新\n内容")
    deadline = time.time() + 10
    while time.time() < deadline and "新增.md" not in _by_name(lib):
        time.sleep(0.05)
    assert "新增.md" in _by_name(lib)


# ---------------------------------------------------------------- queue, changes, registry


def test_queue_waits_while_a_question_is_answered(lib, desk):
    lib.import_path(str(desk / "桌面" / "笔记.txt"))
    started = []
    with lib.answering():
        waiter = threading.Thread(target=lambda: (lib.pause_for_questions(), started.append(time.perf_counter())))
        waiter.start()
        time.sleep(0.3)
        assert started == [] and lib.status()["paused"] is True
        released = time.perf_counter()
    waiter.join(5)
    assert started and started[0] >= released
    lib.settings.index_yield_max_seconds = 0.2  # never starves the queue
    with lib.answering():
        began = time.perf_counter()
        lib.pause_for_questions()
        assert 0.15 <= time.perf_counter() - began < 2


def test_network_failures_are_retried_after_a_delay(lib, desk):
    record = lib.import_path(str(desk / "桌面" / "白板.jpg"))
    real = lib.indexer

    def offline(path, file_id, source, image_mode=None, pause=None):
        raise ConnectionError("proxy dropped the connection")

    lib.indexer = offline
    assert lib.run_until_idle() == ["failed"]
    attempts, due = lib._retries[record["file_id"]]
    assert attempts == 1 and due > time.monotonic()
    assert lib._requeue_due() == 0  # not yet
    lib._retries[record["file_id"]] = (attempts, time.monotonic() - 1)
    lib.indexer = real
    assert lib._requeue_due() == 1 and lib.run_until_idle() == ["indexed"]
    assert record["file_id"] not in lib._retries

    # A broken file (not the network) is not retried on a timer.
    lib.indexer = lambda path, file_id, source, image_mode=None, pause=None: (_ for _ in ()).throw(ValueError("坏文件"))
    other = lib.import_path(str(desk / "桌面" / "笔记.txt"))
    assert lib.run_until_idle() == ["failed"] and other["file_id"] not in lib._retries


def test_registry_adds_the_new_columns_to_an_old_file(tmp_path):
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE watch_roots (root_id TEXT PRIMARY KEY, path TEXT NOT NULL, path_key TEXT NOT NULL, volume_id INTEGER,"
        " status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT,"
        " device_id TEXT NOT NULL);"
        "CREATE TABLE files (file_id TEXT PRIMARY KEY, root_id TEXT, origin TEXT NOT NULL DEFAULT 'watch', path TEXT NOT NULL,"
        " path_key TEXT NOT NULL, name TEXT NOT NULL, volume_id INTEGER, inode INTEGER, size INTEGER, mtime REAL, sha256 TEXT,"
        " indexed_sha256 TEXT, status TEXT NOT NULL DEFAULT 'pending', error TEXT, chunks INTEGER NOT NULL DEFAULT 0,"
        " characters INTEGER, legacy_doc_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT,"
        " device_id TEXT NOT NULL);"
    )
    conn.close()
    registry = FileRegistry(path)
    record = registry.insert_file(str(tmp_path / "a.png"), root_id=None, image_mode="name", notice=None)
    root = registry.add_root(str(tmp_path), volume_id=1, inode=42, image_mode="local")
    assert registry.get_file(record["file_id"])["image_mode"] == "name"
    assert registry.get_root(root["root_id"])["inode"] == 42
    registry.update_file(record["file_id"], status="recycled")
    registry.close()


def test_move_root_rewrites_every_child_path(tmp_path):
    registry = FileRegistry(tmp_path / "r.sqlite3")
    root = registry.add_root(str(tmp_path / "A"), volume_id=1)
    for name in ("x.md", "sub\\y.md" if os.name == "nt" else "sub/y.md"):
        registry.insert_file(str(tmp_path / "A" / name), root_id=root["root_id"])
    assert registry.move_root(root["root_id"], str(tmp_path / "B")) == 2
    assert {r["path"] for r in registry.list_files()} == {str(tmp_path / "B" / "x.md"), str(tmp_path / "B" / "sub" / "y.md")}
    assert registry.root_for_path(str(tmp_path / "B" / "x.md"))["root_id"] == root["root_id"]
    registry.close()


def test_hidden_documents_are_left_out_of_retrieval():
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.types import SearchHit

    def hit(doc_id):
        return SearchHit(id=doc_id, score=1.0, text="t", doc_id=doc_id, parent_id="p", source="s", heading="",
                         locator="", retrieval_method="hybrid")

    store = SimpleNamespace(search_hybrid=lambda query, vector, limit, doc_ids=None: [hit("a" * 32), hit("b" * 32)])
    settings = SimpleNamespace(candidate_auto_scale=False, fusion_top_k=10, rerank_top_k=5, rerank_min_score=0,
                               rerank_relative_score=0, rerank_min_keep=1, rerank_per_document=0, dense_top_k=10,
                               sparse_top_k=10)
    retriever = HybridRetriever(store, SimpleNamespace(embed_query=lambda q: [0.0]), None, settings)
    retriever.hidden_documents = lambda: {"b" * 32}
    assert [h.doc_id for h in retriever.retrieve("差旅", mode="hybrid")] == ["a" * 32]
