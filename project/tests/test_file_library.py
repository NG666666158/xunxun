"""Local file model: registry, folder scan, background queue, watcher events and reconciliation."""

import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import pytest

from db.parent_store_manager import ParentStoreManager
from ingestion.models import ParentChunk
from ingestion.parsers import stable_doc_id
from library import FileRegistry, LibraryError, LibraryService
from library import service as service_module
from library.watcher import Debouncer, EventRouter


class FakeStore:
    """Chunk store stand-in: tracks which doc ids have chunks and their source names."""

    def __init__(self, documents=None):
        self.documents = dict(documents or {})

    def has_document(self, doc_id):
        return doc_id in self.documents

    def delete_document(self, doc_id):
        self.documents.pop(doc_id, None)

    def rename_source(self, doc_id, source):
        if doc_id in self.documents:
            self.documents[doc_id] = source

    def reassign_document(self, old, new):
        self.documents[new] = self.documents.pop(old)

    def list_documents(self):
        return [{"doc_id": doc_id, "source": source} for doc_id, source in self.documents.items()]

    def chunk_counts(self):
        return {doc_id: 2 for doc_id in self.documents}


class Indexer:
    def __init__(self, store):
        self.store = store
        self.calls = []

    def __call__(self, path, file_id, source):
        self.calls.append((Path(path).name, file_id))
        self.store.documents[file_id] = source
        return 2, len(Path(path).read_text(encoding="utf-8"))


def _settings(tmp_path):
    settings = SimpleNamespace(
        allowed_extensions=(".pdf", ".docx", ".xlsx", ".md", ".txt"),
        max_upload_mb=1, max_local_file_mb=1, max_local_text_mb=1,
        upload_dir=tmp_path / "data" / "uploads",
        parsed_dir=tmp_path / "data" / "parsed",
        parent_store_dir=tmp_path / "data" / "parents",
        watch_max_files=1000,
    )
    for path in (settings.upload_dir, settings.parsed_dir, settings.parent_store_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings


@pytest.fixture
def lib(tmp_path):
    settings = _settings(tmp_path)
    store = FakeStore()
    indexer = Indexer(store)
    library = LibraryService(
        settings, FileRegistry(tmp_path / "data" / "registry.sqlite3"), store, ParentStoreManager(settings), indexer
    )
    library.indexer_calls = indexer.calls
    yield library
    library.registry.close()


def _write(path, text="内容"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "资料"
    _write(root / "制度" / "报销制度.md", "# 报销\n住宿费标准")
    _write(root / "说明.txt", "设备说明")
    _write(root / "图片.png", "not really a png")
    _write(root / ".git" / "config.md", "ignored")
    _write(root / "node_modules" / "pkg" / "readme.md", "ignored")
    _write(root / ".hidden" / "secret.md", "ignored")
    _write(root / "~$报销制度.docx", "office lock file")
    (root / "大文件.txt").write_bytes(b"x" * (1024 * 1024 + 1))
    return root


def _by_name(lib):
    return {record["name"]: record for record in lib.registry.list_files()}


# ---------------------------------------------------------------- registry


def test_registry_rows_carry_sync_fields_and_path_is_unique_while_live(tmp_path):
    registry = FileRegistry(tmp_path / "r.sqlite3")
    record = registry.insert_file(str(tmp_path / "A.md"), root_id=None, size=3)
    assert len(record["file_id"]) == 32 and record["device_id"] == registry.device_id
    assert record["created_at"] and record["updated_at"] and record["deleted_at"] is None

    if os.name == "nt":
        assert registry.get_by_path(str(tmp_path / "a.MD"))["file_id"] == record["file_id"]
    with pytest.raises(Exception):
        registry.insert_file(str(tmp_path / "A.md"), root_id=None)

    registry.soft_delete_file(record["file_id"])
    assert registry.get_file(record["file_id"]) is None
    tombstone = registry.get_file(record["file_id"], include_deleted=True)
    assert tombstone["deleted_at"]
    assert any(row["file_id"] == record["file_id"] for row in registry.changes_since(""))
    # A tombstone frees the path for a new record.
    registry.insert_file(str(tmp_path / "A.md"), root_id=None)
    registry.close()

    reopened = FileRegistry(tmp_path / "r.sqlite3")
    assert reopened.device_id == registry.device_id
    with pytest.raises(ValueError):
        reopened.update_file(record["file_id"], status="weird")
    reopened.close()


def test_files_under_matches_only_that_folder(tmp_path):
    registry = FileRegistry(tmp_path / "r.sqlite3")
    registry.insert_file(str(tmp_path / "docs" / "a.md"), root_id=None)
    registry.insert_file(str(tmp_path / "docs" / "sub" / "b.md"), root_id=None)
    registry.insert_file(str(tmp_path / "docs_old" / "c.md"), root_id=None)
    assert {r["name"] for r in registry.files_under(str(tmp_path / "docs"))} == {"a.md", "b.md"}
    registry.close()


# ---------------------------------------------------------------- scan & queue


def test_preview_applies_exclusion_rules_and_estimates(lib, folder):
    preview = lib.preview_root(str(folder))
    assert preview["files"] == 4  # 2 indexable + png + oversized txt
    assert preview["indexable"] == 2
    assert preview["unsupported"] == 2 and preview["too_large"] == 1
    texts = [(folder / "制度" / "报销制度.md").read_bytes().decode("utf-8"), "设备说明"]
    assert preview["estimated_characters"] == sum(len(text) for text in texts)
    assert lib.registry.list_roots() == []  # preview registers nothing


def test_add_root_validates_paths(lib, folder, tmp_path):
    with pytest.raises(LibraryError):
        lib.preview_root(str(tmp_path / "nope"))
    with pytest.raises(LibraryError):
        lib.preview_root("relative\\path")
    lib.add_root(str(folder), background=False)
    for overlapping in (folder, folder / "制度", tmp_path):
        with pytest.raises(LibraryError):
            lib.add_root(str(overlapping), background=False)


def test_add_root_registers_paths_and_indexes_in_background_queue(lib, folder):
    root = lib.add_root(str(folder), background=False)
    files = _by_name(lib)
    assert set(files) == {"报销制度.md", "说明.txt", "图片.png", "大文件.txt"}
    assert files["图片.png"]["status"] == "unsupported"
    assert "大小上限" in files["大文件.txt"]["error"]
    assert files["报销制度.md"]["status"] == "pending"
    assert files["报销制度.md"]["path"] == str(folder / "制度" / "报销制度.md")
    assert files["报销制度.md"]["inode"] == os.stat(folder / "制度" / "报销制度.md").st_ino
    assert lib.status()["queued"] == 2

    assert lib.run_until_idle() == ["indexed", "indexed"]
    files = _by_name(lib)
    assert files["报销制度.md"]["status"] == "indexed"
    assert files["报销制度.md"]["sha256"] == files["报销制度.md"]["indexed_sha256"]
    # The file id is the chunk store's doc id; nothing was copied into the data dir.
    assert files["说明.txt"]["file_id"] in lib.store.documents
    assert not any(lib.settings.upload_dir.iterdir())
    assert lib.list_roots()[0]["counts"]["indexed"] == 2
    assert root["path"] == str(folder)


def test_unchanged_content_is_not_re_embedded(lib, folder):
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    target = folder / "说明.txt"
    os.utime(target, (time.time() + 5, time.time() + 5))
    lib.handle_path(str(target))
    assert lib.run_until_idle() == ["skipped"]

    _write(target, "设备说明（修订）")
    lib.handle_path(str(target))
    assert lib.run_until_idle() == ["indexed"]
    assert [name for name, _ in lib.indexer_calls].count("说明.txt") == 2


def test_reindex_parses_an_unchanged_file_again(lib, folder):
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    record = _by_name(lib)["说明.txt"]

    queued = lib.reindex(record["file_id"])
    assert queued["status"] == "pending"
    assert lib.run_until_idle() == ["indexed"]
    assert [name for name, _ in lib.indexer_calls].count("说明.txt") == 2
    assert _by_name(lib)["说明.txt"]["status"] == "indexed"
    # Only that once: the next unchanged touch is skipped again.
    os.utime(folder / "说明.txt", (time.time() + 5, time.time() + 5))
    lib.handle_path(str(folder / "说明.txt"))
    assert lib.run_until_idle() == ["skipped"]

    (folder / "说明.txt").unlink()
    lib.handle_path(str(folder / "说明.txt"))
    assert lib.reindex(record["file_id"])["status"] == "missing"
    assert lib.run_until_idle() == []
    with pytest.raises(KeyError):
        lib.reindex("0" * 32)


def test_failed_index_keeps_error_and_status(lib, folder):
    lib.add_root(str(folder), background=False)

    def broken(path, file_id, source):
        raise ValueError("没有从文件提取到可索引文本")

    lib.indexer = broken
    assert lib.run_until_idle() == ["failed", "failed"]
    record = _by_name(lib)["说明.txt"]
    assert record["status"] == "failed" and "可索引文本" in record["error"]


def test_remove_file_tombstones_and_drops_index(lib, folder):
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    record = _by_name(lib)["说明.txt"]
    lib.remove_file(record["file_id"])
    assert record["file_id"] not in lib.store.documents
    assert lib.registry.get_file(record["file_id"], include_deleted=True)["deleted_at"]
    assert (folder / "说明.txt").exists()  # the original is never touched


def test_remove_root_removes_its_files(lib, folder):
    root = lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    assert lib.remove_root(root["root_id"]) == 4
    assert lib.registry.list_files() == [] and lib.store.documents == {}
    assert lib.list_roots() == []


# ---------------------------------------------------------------- watcher events


def _event(kind, src, dest=None, is_directory=False):
    return SimpleNamespace(event_type=kind, src_path=str(src), dest_path=str(dest or ""), is_directory=is_directory)


@pytest.fixture
def watched(lib, folder):
    root = lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    lib.indexer_calls.clear()
    router = EventRouter(lib, lib.registry.get_root(root["root_id"]), lambda key, action, delay=None: action())
    return lib, folder, router


def test_watcher_created_file_is_indexed(watched):
    lib, folder, router = watched
    new = _write(folder / "制度" / "新制度.md", "# 新\n内容")
    router.dispatch(_event("created", new))
    assert _by_name(lib)["新制度.md"]["status"] == "pending"
    assert lib.run_until_idle() == ["indexed"]


def test_watcher_rename_only_updates_path(watched):
    lib, folder, router = watched
    before = _by_name(lib)["说明.txt"]
    target = folder / "制度" / "设备说明.txt"
    (folder / "说明.txt").rename(target)
    router.dispatch(_event("moved", folder / "说明.txt", target))

    after = lib.registry.get_file(before["file_id"])
    assert after["path"] == str(target) and after["status"] == "indexed"
    assert lib.store.documents[before["file_id"]] == "设备说明.txt"  # chunk source renamed in place
    assert lib.run_until_idle() == [] and lib.indexer_calls == []


def test_watcher_folder_rename_moves_all_children(watched):
    lib, folder, router = watched
    before = _by_name(lib)["报销制度.md"]
    (folder / "制度").rename(folder / "规章")
    router.dispatch(_event("moved", folder / "制度", folder / "规章", is_directory=True))
    assert lib.registry.get_file(before["file_id"])["path"] == str(folder / "规章" / "报销制度.md")
    assert lib.run_until_idle() == []


def test_watcher_cross_folder_move_as_delete_plus_create_is_matched_by_file_id(watched):
    lib, folder, router = watched
    before = _by_name(lib)["说明.txt"]
    target = folder / "制度" / "说明.txt"
    os.replace(folder / "说明.txt", target)
    router.dispatch(_event("created", target))  # Windows reports these moves as delete + create
    router.dispatch(_event("deleted", folder / "说明.txt"))
    after = lib.registry.get_file(before["file_id"])
    assert after["path"] == str(target) and after["status"] == "indexed"
    assert lib.run_until_idle() == []


def test_watcher_delete_marks_missing_and_keeps_index(watched):
    lib, folder, router = watched
    record = _by_name(lib)["说明.txt"]
    (folder / "说明.txt").unlink()
    router.dispatch(_event("deleted", folder / "说明.txt"))
    assert lib.registry.get_file(record["file_id"])["status"] == "missing"
    assert record["file_id"] in lib.store.documents


def test_watcher_editor_save_via_temp_rename_reindexes_same_record(watched):
    lib, folder, router = watched
    record = _by_name(lib)["说明.txt"]
    temp = _write(folder / "说明.txt.tmp123", "设备说明 v2")
    os.replace(temp, folder / "说明.txt")
    router.dispatch(_event("moved", temp, folder / "说明.txt"))
    assert lib.run_until_idle() == ["indexed"]
    assert lib.registry.get_file(record["file_id"])["status"] == "indexed"
    assert len(lib.registry.list_files()) == 4


def test_watcher_ignores_excluded_paths(watched):
    lib, folder, router = watched
    router.dispatch(_event("created", _write(folder / ".git" / "HEAD.md", "x")))
    router.dispatch(_event("created", _write(folder / "~$说明.docx", "x")))
    router.dispatch(_event("modified", folder / "制度", is_directory=True))
    assert len(lib.registry.list_files()) == 4


def test_debouncer_collapses_bursts_per_key():
    calls = []
    debouncer = Debouncer(0.15)
    for _ in range(5):
        debouncer.submit("a", lambda: calls.append("a"))
    debouncer.submit("b", lambda: calls.append("b"), 0)
    deadline = time.time() + 3
    while len(calls) < 2 and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)
    debouncer.stop()
    assert calls == ["b", "a"]


def test_real_watchdog_reports_new_file(lib, folder):
    lib.settings.watch_debounce_seconds = 0.2
    lib.settings.library_monitor_seconds = 3600
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    lib.start(watch=True)
    try:
        _write(folder / "实时.md", "# 实时\n新增")
        deadline = time.time() + 10
        while time.time() < deadline:
            record = _by_name(lib).get("实时.md")
            if record and record["status"] == "indexed":
                break
            time.sleep(0.1)
        assert _by_name(lib)["实时.md"]["status"] == "indexed"
    finally:
        lib.stop()


# ---------------------------------------------------------------- startup reconciliation


def test_reconcile_finds_moved_file_by_volume_and_file_id(lib, folder):
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    record = _by_name(lib)["说明.txt"]
    (folder / "说明.txt").rename(folder / "制度" / "改名后.txt")  # while the app is closed
    lib.reconcile()
    after = lib.registry.get_file(record["file_id"])
    assert after["path"] == str(folder / "制度" / "改名后.txt")
    assert after["status"] == "indexed"
    assert lib.run_until_idle() == []


def test_reconcile_falls_back_to_size_and_sha256(lib, folder):
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    record = _by_name(lib)["说明.txt"]
    # Copy + delete gives the file a new file ID; only the content still matches.
    shutil.copy2(folder / "说明.txt", folder / "制度" / "副本.txt")
    (folder / "说明.txt").unlink()
    lib.reconcile()
    after = lib.registry.get_file(record["file_id"])
    assert after["path"] == str(folder / "制度" / "副本.txt") and after["status"] == "indexed"
    assert lib.run_until_idle() == []


def test_reconcile_marks_deleted_file_missing(lib, folder):
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    record = _by_name(lib)["说明.txt"]
    (folder / "说明.txt").unlink()
    (folder / "图片.png").unlink()
    lib.reconcile()
    assert lib.registry.get_file(record["file_id"])["status"] == "missing"
    assert record["file_id"] in lib.store.documents  # still searchable
    assert "图片.png" not in _by_name(lib)  # nothing was indexed for it, so it is just forgotten


def test_reconcile_marks_unmounted_volume_offline(lib, folder, monkeypatch):
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    moved_away = folder.parent / "elsewhere"
    shutil.move(str(folder), moved_away)  # the "drive" is gone
    monkeypatch.setattr(service_module, "volume_available", lambda path, volume_id=None: False)
    lib.reconcile()
    statuses = {r["name"]: r["status"] for r in lib.registry.list_files()}
    assert statuses["说明.txt"] == "offline" and statuses["报销制度.md"] == "offline"
    assert lib.list_roots()[0]["status"] == "offline"

    # The drive comes back: files return to indexed without re-embedding.
    shutil.move(str(moved_away), folder)
    monkeypatch.setattr(service_module, "volume_available", lambda path, volume_id=None: True)
    lib.check_volumes()
    assert lib.run_until_idle() == ["skipped", "skipped"]
    assert {r["status"] for r in lib.registry.list_files() if r["name"].endswith((".md", ".txt")) and r["size"] < 1000} == {"indexed"}


def test_relocate_missing_file(lib, folder, tmp_path):
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    record = _by_name(lib)["说明.txt"]
    outside = tmp_path / "备份" / "说明.txt"
    outside.parent.mkdir()
    # Copied out and deleted (like a move to another drive): a new file ID, so it is lost.
    shutil.copy2(folder / "说明.txt", outside)
    (folder / "说明.txt").unlink()
    lib.handle_path(str(folder / "说明.txt"))
    assert lib.registry.get_file(record["file_id"])["status"] == "missing"

    with pytest.raises(LibraryError):
        lib.relocate(record["file_id"], str(tmp_path / "nope.txt"))
    after = lib.relocate(record["file_id"], str(outside))
    assert after["path"] == str(outside) and after["status"] == "indexed" and after["root_id"] is None
    assert lib.run_until_idle() == []


# ---------------------------------------------------------------- legacy migration


def test_migrate_legacy_rekeys_index_and_locates_original(lib, tmp_path):
    original = _write(tmp_path / "sample" / "员工手册.md", "# 手册\n内容")
    old_id = stable_doc_id(original)
    lib.store.documents[old_id] = "员工手册.md"
    lib.parent_store.save_document(
        old_id, [ParentChunk(f"{old_id}-p0", old_id, "员工手册.md", "手册", "", "内容")]
    )
    (lib.settings.parsed_dir / f"{old_id}.json").write_text(
        json.dumps({"doc_id": old_id, "source": "员工手册.md", "file_type": ".md", "units": []}), encoding="utf-8"
    )
    lost_id = "c" * 24
    lib.store.documents[lost_id] = "已删除.pdf"

    migrated = lib.migrate_legacy([tmp_path / "sample"])
    by_name = {record["name"]: record for record in migrated}
    found = by_name["员工手册.md"]
    assert found["status"] == "indexed" and found["path"] == str(original)
    assert found["legacy_doc_id"] == old_id and len(found["file_id"]) == 32
    assert old_id not in lib.store.documents and found["file_id"] in lib.store.documents
    assert lib.parent_store.get(f"{found['file_id']}-p0")["doc_id"] == found["file_id"]
    assert json.loads((lib.settings.parsed_dir / f"{found['file_id']}.json").read_text())["doc_id"] == found["file_id"]
    assert by_name["已删除.pdf"]["status"] == "missing"

    assert lib.migrate_legacy() == []  # idempotent


def test_migrate_after_folder_was_scanned_and_after_interrupted_rekey(lib, tmp_path):
    sample = tmp_path / "sample"
    original = _write(sample / "员工手册.md", "# 手册\n内容")
    other = _write(sample / "价格表.md", "# 价格\n表")
    old_id = stable_doc_id(original)
    lib.store.documents[old_id] = "员工手册.md"
    orphan_id = "d" * 32  # chunks already re-keyed, registry row never written
    lib.store.documents[orphan_id] = "价格表.md"
    lib.add_root(str(sample), background=False)  # scan registers both paths as pending first

    migrated = {record["name"]: record for record in lib.migrate_legacy([sample])}
    assert migrated["员工手册.md"]["status"] == "indexed"
    assert migrated["价格表.md"]["file_id"] == orphan_id
    assert migrated["价格表.md"]["path"] == str(other)
    assert {r["name"]: r["status"] for r in lib.registry.list_files()} == {"员工手册.md": "indexed", "价格表.md": "indexed"}
    lib.run_until_idle()
    assert lib.indexer_calls == []  # nothing re-embedded


# ---------------------------------------------------------------- parser upgrades


def test_parser_upgrade_requeues_unchanged_files_once(lib, folder, monkeypatch):
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    current = service_module.parser_version("a.md")
    assert {r["name"]: r["parser_version"] for r in lib.registry.list_files(statuses=("indexed",))} == {
        "报销制度.md": current,
        "说明.txt": current,
    }
    lib.reconcile()
    assert lib.status()["queued"] == 0

    # A new Markdown parser: only .md files are parsed again, although nothing changed on disk.
    monkeypatch.setattr(service_module, "parser_version", lambda path, settings=None: current + (1 if str(path).endswith(".md") else 0))
    calls = len(lib.indexer_calls)
    stats = lib.reconcile()
    assert stats["reparse"] == 1 and lib.status()["queued"] == 1
    assert lib.run_until_idle() == ["indexed"]  # not "skipped", although the sha256 matches
    assert lib.indexer_calls[calls:] == [("报销制度.md", _by_name(lib)["报销制度.md"]["file_id"])]
    assert _by_name(lib)["报销制度.md"]["parser_version"] == current + 1

    assert lib.reconcile()["reparse"] == 0


def test_registry_without_parser_version_column_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    registry = FileRegistry(path)
    record = registry.insert_file(str(tmp_path / "a.pdf"), root_id=None)
    registry.close()
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE files DROP COLUMN parser_version")
    conn.commit()
    conn.close()

    reopened = FileRegistry(path)
    assert reopened.get_file(record["file_id"])["parser_version"] is None
    reopened.update_file(record["file_id"], parser_version=2)
    assert reopened.get_file(record["file_id"])["parser_version"] == 2
    reopened.close()


def test_failed_reparse_of_indexed_file_is_retried_at_next_start(lib, folder, monkeypatch):
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    current = service_module.parser_version("a.md")
    monkeypatch.setattr(service_module, "parser_version", lambda path, settings=None: current + (1 if str(path).endswith(".md") else 0))
    good_indexer = lib.indexer

    def timeout(path, file_id, source):
        raise TimeoutError("Request timed out.")

    lib.indexer = timeout
    lib.reconcile()
    assert lib.run_until_idle() == ["failed"]
    record = _by_name(lib)["报销制度.md"]
    assert record["status"] == "failed" and record["parser_version"] == current
    assert record["file_id"] in lib.store.documents  # the old index is still searchable

    lib.indexer = good_indexer
    assert lib.reconcile()["reparse"] == 1
    assert lib.run_until_idle() == ["indexed"]
    assert _by_name(lib)["报销制度.md"]["parser_version"] == current + 1


def test_switching_image_mode_requeues_only_pdfs(lib, folder, monkeypatch):
    _write(folder / "手册.pdf", "pdf placeholder")
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    assert lib.requeue_outdated() == 0

    real = service_module.parser_version
    lib.settings.image_understanding = "local"
    monkeypatch.setattr(service_module, "parser_version", lambda path, settings=None: real(path, settings))
    assert lib.requeue_outdated() == 1  # only the PDF's index version depends on the image mode
    assert lib.run_until_idle() == ["indexed"]
    assert lib.requeue_outdated() == 0


def test_a_refused_account_waits_for_the_next_start_instead_of_retrying_every_few_minutes(lib, folder):
    import httpx
    import openai

    def no_balance(path, file_id, source):
        request = httpx.Request("POST", "https://api.siliconflow.cn/v1/embeddings")
        response = httpx.Response(402, request=request, json={"code": 30001})
        raise openai.APIStatusError("Sorry, your account balance is insufficient", response=response, body=None)

    good = lib.indexer
    lib.indexer = no_balance
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    failed = lib.registry.list_files(statuses=("failed",))
    assert len(failed) == 2 and lib._retries == {}  # nothing re-read within minutes
    assert all(r["parser_version"] is None for r in failed)

    lib.indexer = good
    assert lib.reconcile()["queued"] == 2  # the next start tries again
    assert lib.run_until_idle() == ["indexed", "indexed"]


def test_transient_failure_of_a_new_file_is_retried_at_next_start(lib, folder):
    def timeout(path, file_id, source):
        raise TimeoutError("Request timed out.")

    def broken(path, file_id, source):
        raise ValueError("没有从文件中提取到可索引文本")

    good = lib.indexer
    lib.indexer = timeout
    lib.add_root(str(folder), background=False)
    lib.run_until_idle()
    assert {r["status"] for r in lib.registry.list_files(statuses=("failed",))} == {"failed"}
    assert all(r["parser_version"] is None for r in lib.registry.list_files(statuses=("failed",)))

    lib.indexer = good
    assert lib.reconcile()["queued"] == 2  # retried although the files did not change
    assert lib.run_until_idle() == ["indexed", "indexed"]

    lib.indexer = broken
    target = folder / "说明.txt"
    target.write_text("改动后的内容", encoding="utf-8")
    lib.handle_path(str(target))
    lib.run_until_idle()
    lib.indexer = good
    assert lib.reconcile()["queued"] == 0  # a file that cannot be parsed is not retried every start
