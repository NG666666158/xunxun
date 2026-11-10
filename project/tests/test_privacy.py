"""Name-only files and exclusions (library/privacy.py, design/PRIVACY.md): what the library
never reads. The fake indexer records every file it is asked to read; a file kept by name only
or excluded must never be in that list."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from db.parent_store_manager import ParentStoreManager
from library import FileRegistry, LibraryError, LibraryService
from library import privacy
from library import service as service_module
from library import winfs


class FakeStore:
    def __init__(self):
        self.documents = {}
        self.purged = []

    def has_document(self, doc_id):
        return doc_id in self.documents

    def delete_document(self, doc_id):
        self.documents.pop(doc_id, None)

    def purge_document(self, doc_id):
        self.purged.append(doc_id)
        self.delete_document(doc_id)

    def rename_source(self, doc_id, source):
        if doc_id in self.documents:
            self.documents[doc_id] = source

    def list_documents(self):
        return [{"doc_id": d, "source": s} for d, s in self.documents.items()]


class Indexer:
    """Records the path of every file it reads (and its text), like the app's indexer would
    send it to the model service. ``during`` runs once inside a read, before its first pause."""

    def __init__(self, store, settings):
        self.store = store
        self.settings = settings
        self.read = []
        self.during = None

    def __call__(self, path, file_id, source, image_mode=None, pause=None):
        text = Path(path).read_text(encoding="utf-8")
        self.read.append((Path(path).name, text))
        self.store.documents[file_id] = source
        (Path(self.settings.parsed_dir) / f"{file_id}.json").write_text("{}", encoding="utf-8")
        if self.during:
            action, self.during = self.during, None
            action(file_id)
        if pause:
            pause()
        return 1, len(text), {}

    def texts(self):
        return "\n".join(text for _, text in self.read)


def _settings(tmp_path):
    settings = SimpleNamespace(
        allowed_extensions=(".md", ".txt", ".pdf", ".png"), max_upload_mb=1, max_local_file_mb=1, max_local_text_mb=1,
        data_dir=tmp_path / "data", upload_dir=tmp_path / "data" / "uploads", parsed_dir=tmp_path / "data" / "parsed",
        parent_store_dir=tmp_path / "data" / "parents", watch_max_files=1000, image_import_mode="local",
        image_understanding="off", vision_concurrency=6, index_yield_max_seconds=5,
    )
    for path in (settings.upload_dir, settings.parsed_dir, settings.parent_store_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings


@pytest.fixture
def lib(tmp_path):
    settings = _settings(tmp_path)
    store = FakeStore()
    indexer = Indexer(store, settings)
    library = LibraryService(settings, FileRegistry(tmp_path / "data" / "r.sqlite3"), store,
                             ParentStoreManager(settings), indexer)
    library.purged = []
    library.on_purged = library.purged.extend
    library.indexer_calls = indexer
    yield library
    library.stop()
    library.registry.close()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def desk(tmp_path):
    """资料/ with a 证件 sub-folder and a 合同 sub-folder, each holding a secret."""
    root = tmp_path / "资料"
    _write(root / "公开" / "制度.md", "# 差旅制度\n报销在十五个工作日内提交。")
    _write(root / "证件" / "护照.txt", "护照号码 SECRET-PASSPORT-7731")
    _write(root / "合同" / "租房合同.txt", "月租金 SECRET-RENT-4410")
    _write(root / "合同" / "附件" / "身份证.txt", "身份证号 SECRET-ID-9902")
    _write(root / "server.key.txt", "SECRET-KEY-0001")
    return root


def _record(lib, name):
    return next(row for row in lib.registry.list_files() if row["name"] == name)


def _secrets_read(lib):
    return [secret for secret in ("PASSPORT", "RENT", "ID-9902", "KEY-0001", "NOTES") if f"SECRET-{secret}" in lib.indexer_calls.texts()]


# ---------------------------------------------------------------- rules


def test_decide_most_specific_rule_and_exclusion_wins():
    rule = lambda kind, value, action: {"rule_id": value + action, "kind": kind, "value": value, "action": action}  # noqa: E731
    path_rules = [rule("path", "", "name"), rule("path", os.path.join("合同", "公开"), "full")]
    assert privacy.decide(["a.txt"], path_rules, [], ["D:", "资料"]).mode == "name"
    assert privacy.decide(["合同", "公开", "a.txt"], path_rules, [], ["资料"]).mode == "full"
    # A pattern at the same depth as a path rule loses; a deeper pattern wins.
    patterns = [rule("pattern", "公开", "name")]
    assert privacy.decide(["合同", "公开", "a.txt"], path_rules, patterns, ["资料"]).mode == "full"
    assert privacy.decide(["合同", "公开", "证件", "a.txt"], path_rules, [rule("pattern", "证件", "name")], ["资料"]).mode == "name"
    # Exclusion wins over everything, patterns match folders and files, case-insensitively.
    assert privacy.decide(["x", "ID.KEY"], path_rules, [rule("pattern", "*.key", "exclude")], []).excluded
    assert privacy.decide(["证件", "a.txt"], [rule("path", "证件", "exclude")], [], []).excluded
    assert not privacy.decide(["证件2", "a.txt"], [rule("path", "证件", "exclude")], [], []).excluded
    # Single files (no watched folder): only patterns, on every folder name of the path.
    assert privacy.decide(None, path_rules, [rule("pattern", "证件", "exclude")], ["Users", "证件", "a.txt"]).excluded


def test_patterns_are_one_name_not_a_path_or_everything():
    assert privacy.clean_pattern(" *.key ") == "*.key"
    for bad in ("", "a\\b", "C:\\x", "*", "*.*", "  "):
        with pytest.raises(ValueError):
            privacy.clean_pattern(bad)


# ---------------------------------------------------------------- name only


def test_folder_kept_by_name_only_is_never_read_and_new_files_follow(lib, desk):
    lib.add_root(str(desk), background=False)
    lib.set_folder_mode(str(desk / "证件"), "name")
    lib.run_until_idle()
    passport = _record(lib, "护照.txt")
    assert passport["status"] == "named" and passport["content_mode"] == "name"
    assert passport["chunks"] == 0 and not lib.store.has_document(passport["file_id"])
    assert "SECRET-PASSPORT" not in lib.indexer_calls.texts()
    # Findable: listed with its name, folder and dates, but nothing to cite.
    view = lib.file_view(passport)
    assert view["content_mode"] == "name" and view["searchable"] is False and view["size"] > 0

    # A file added to the folder later follows the folder; so does a re-index request.
    _write(desk / "证件" / "驾照.txt", "驾照号 SECRET-LICENSE-55")
    lib.handle_path(str(desk / "证件" / "驾照.txt"))
    lib.reindex(passport["file_id"])
    lib.run_until_idle()
    assert _record(lib, "驾照.txt")["status"] == "named"
    assert "SECRET-LICENSE" not in lib.indexer_calls.texts() and _secrets_read(lib) == ["RENT", "ID-9902", "KEY-0001"]
    # Restart: nothing is read either.
    lib.reconcile()
    lib.run_until_idle()
    assert "SECRET-PASSPORT" not in lib.indexer_calls.texts()


def test_switching_an_indexed_file_drops_its_text_and_back_reads_it(lib, desk):
    lib.add_root(str(desk), background=False)
    lib.run_until_idle()
    rent = _record(lib, "租房合同.txt")
    assert lib.store.has_document(rent["file_id"])
    parsed = Path(lib.settings.parsed_dir) / f"{rent['file_id']}.json"
    assert parsed.is_file()

    preview = lib.set_content_mode(rent["file_id"], "name", dry_run=True)
    assert preview["changed"] == 1 and preview["indexed"] == 1
    assert lib.store.has_document(rent["file_id"])  # a dry run changes nothing

    result = lib.set_content_mode(rent["file_id"], "name")
    assert result["file"]["status"] == "named" and result["file"]["content_mode"] == "name"
    assert not lib.store.has_document(rent["file_id"]) and not parsed.exists()
    assert rent["file_id"] in lib.store.purged and rent["file_id"] in lib.purged  # backups get scrubbed
    reads = len(lib.indexer_calls.read)
    lib.reindex(rent["file_id"])
    lib.run_until_idle()
    assert len(lib.indexer_calls.read) == reads

    lib.set_content_mode(rent["file_id"], "full")
    lib.run_until_idle()
    assert _record(lib, "租房合同.txt")["status"] == "indexed" and lib.store.has_document(rent["file_id"])
    assert len(lib.indexer_calls.read) == reads + 1


def test_switching_while_the_file_is_being_read_stops_and_keeps_nothing(lib, desk):
    lib.add_root(str(desk), background=False)
    lib.indexer_calls.during = lambda file_id: lib.set_content_mode(file_id, "name")
    lib.run_until_idle()
    switched = [row for row in lib.registry.list_files() if row["content_mode"] == "name"]
    assert len(switched) == 1
    row = switched[0]
    assert row["status"] == "named" and not lib.store.has_document(row["file_id"])
    assert not (Path(lib.settings.parsed_dir) / f"{row['file_id']}.json").exists()


def test_file_moved_into_a_name_only_folder_is_kept_by_name_from_then_on(lib, desk):
    lib.add_root(str(desk), background=False)
    lib.set_folder_mode(str(desk / "证件"), "name")
    lib.run_until_idle()
    rule = lib.file_view(_record(lib, "制度.md"))
    assert rule["status"] == "indexed"
    os.replace(desk / "公开" / "制度.md", desk / "证件" / "制度.md")
    lib.handle_move(str(desk / "公开" / "制度.md"), str(desk / "证件" / "制度.md"))
    moved = _record(lib, "制度.md")
    assert moved["content_mode"] == "name" and moved["status"] == "named"
    assert not lib.store.has_document(moved["file_id"])
    # Moving it out again does not read it without being asked.
    os.replace(desk / "证件" / "制度.md", desk / "公开" / "制度.md")
    lib.handle_move(str(desk / "证件" / "制度.md"), str(desk / "公开" / "制度.md"))
    lib.run_until_idle()
    assert _record(lib, "制度.md")["status"] == "named"


def test_whole_folder_by_name_then_one_sub_folder_read(lib, desk):
    lib.add_root(str(desk), background=False)
    lib.set_folder_mode(str(desk), "name")
    lib.set_folder_mode(str(desk / "公开"), "full")
    lib.run_until_idle()
    assert _secrets_read(lib) == [] and "十五个工作日" in lib.indexer_calls.texts()
    root = lib.list_roots()[0]
    assert root["content_mode"] == "name"
    assert sorted(rule["action"] for rule in root["rules"]) == ["full", "name"]
    # Back to full for the whole folder: the deeper choice is replaced, all is read.
    impact = lib.set_folder_mode(str(desk), "full", dry_run=True)
    assert impact["changed"] == 4
    lib.set_folder_mode(str(desk), "full")
    lib.run_until_idle()
    assert [rule["action"] for rule in lib.list_roots()[0]["rules"]] == []
    assert _secrets_read(lib) == ["PASSPORT", "RENT", "ID-9902", "KEY-0001"]


# ---------------------------------------------------------------- exclusions


def test_excluded_sub_folder_and_pattern_are_never_registered(lib, desk):
    root = lib.add_root(str(desk), background=False)
    lib.run_until_idle()
    before = len(lib.indexer_calls.read)
    preview = lib.add_rule(action="exclude", path=str(desk / "合同"), dry_run=True)
    assert preview == {"files": 2, "indexed": 2}
    lib.add_rule(action="exclude", path=str(desk / "合同"))
    lib.add_rule(action="exclude", pattern="*.key.txt", root_id=root["root_id"])
    names = {row["name"] for row in lib.registry.list_files()}
    assert names == {"制度.md", "护照.txt"}
    assert len(lib.purged) == 3 and len(lib.store.documents) == 2

    # New files there, rescans, watcher events and explicit imports all keep out.
    _write(desk / "合同" / "新合同.txt", "SECRET-NEW-1")
    _write(desk / "b.key.txt", "SECRET-NEW-2")
    lib.handle_path(str(desk / "合同" / "新合同.txt"))
    lib.handle_path(str(desk / "合同"))
    lib.handle_path(str(desk / "b.key.txt"))
    lib.scan_root(root["root_id"])
    lib.run_until_idle()
    with pytest.raises(LibraryError):
        lib.import_path(str(desk / "合同" / "新合同.txt"))
    assert {row["name"] for row in lib.registry.list_files()} == {"制度.md", "护照.txt"}
    assert len(lib.indexer_calls.read) == before


def test_global_pattern_covers_new_folders_and_single_files(lib, desk, tmp_path):
    lib.add_rule(action="exclude", pattern="合同")
    lib.add_rule(action="name", pattern="证件")
    lib.add_root(str(desk), background=False)
    lib.run_until_idle()
    assert _secrets_read(lib) == ["KEY-0001"]
    assert _record(lib, "护照.txt")["status"] == "named"
    single = _write(tmp_path / "别处" / "证件" / "签证.txt", "SECRET-VISA-3")
    lib.import_path(str(single))
    lib.run_until_idle()
    assert _record(lib, "签证.txt")["status"] == "named" and "SECRET-VISA" not in lib.indexer_calls.texts()
    with pytest.raises(LibraryError):
        lib.add_root(str(_write(tmp_path / "合同" / "x.txt", "x").parent))
    fresh = tmp_path / "新资料"
    _write(fresh / "合同" / "a.txt", "a")
    _write(fresh / "b.txt", "b")
    inspected = lib.inspect([str(fresh)])
    assert inspected["totals"]["indexable"] == 1  # the 合同 folder is left out of the estimate


def test_renamed_excluded_folder_stays_excluded(lib, desk, monkeypatch):
    root = lib.add_root(str(desk), background=False)
    lib.add_rule(action="exclude", path=str(desk / "合同"))
    # Renamed while the app watches: the rule moves with the folder.
    os.replace(desk / "合同", desk / "合同2024")
    lib.handle_move(str(desk / "合同"), str(desk / "合同2024"))
    lib.run_until_idle()
    rules = lib.list_roots()[0]["rules"]
    assert [rule["value"] for rule in rules] == ["合同2024"]
    # Renamed while the app was closed: found again by its file ID at the next scan.
    inode = rules[0]["rule_id"] and lib.registry.list_rules()[0]["inode"]
    os.replace(desk / "合同2024", desk / "旧合同")
    monkeypatch.setattr(service_module, "resolve_file_id",
                        lambda path, wanted: str(desk / "旧合同") if wanted == inode else winfs.NOT_FOUND)
    if not inode:  # file systems without file IDs: nothing to follow by, the test does not apply
        pytest.skip("no file IDs on this file system")
    lib.scan_root(root["root_id"])
    lib.run_until_idle()
    assert "SECRET-RENT" not in lib.indexer_calls.texts() and "SECRET-ID" not in lib.indexer_calls.texts()
    assert [rule["value"] for rule in lib.list_roots()[0]["rules"]] == ["旧合同"]


def test_file_moved_into_an_excluded_folder_leaves_the_library(lib, desk):
    lib.add_root(str(desk), background=False)
    lib.add_rule(action="exclude", path=str(desk / "合同"))
    lib.run_until_idle()
    record = _record(lib, "制度.md")
    os.replace(desk / "公开" / "制度.md", desk / "合同" / "制度.md")
    lib.handle_move(str(desk / "公开" / "制度.md"), str(desk / "合同" / "制度.md"))
    assert lib.registry.get_file(record["file_id"]) is None
    assert not lib.store.has_document(record["file_id"]) and record["file_id"] in lib.purged


def test_taking_an_exclusion_back_reads_the_files_again(lib, desk):
    root = lib.add_root(str(desk), background=False)
    rule = lib.add_rule(action="exclude", path=str(desk / "合同"))["rule"]
    lib.run_until_idle()
    assert lib.remove_rule(rule["rule_id"], dry_run=True)["files"] == 2
    assert len(lib.registry.list_rules()) == 1
    lib.remove_rule(rule["rule_id"])
    lib._scanning.discard(root["root_id"])
    lib.scan_root(root["root_id"])
    lib.run_until_idle()
    assert "SECRET-RENT" in lib.indexer_calls.texts()
    assert lib.registry.list_rules() == []


def test_removing_a_watched_folder_drops_its_rules(lib, desk):
    root = lib.add_root(str(desk), background=False)
    lib.add_rule(action="exclude", path=str(desk / "合同"))
    lib.set_folder_mode(str(desk / "证件"), "name")
    lib.remove_root(root["root_id"])
    assert lib.registry.list_rules() == []
    summary = lib.privacy_summary()
    assert summary == {"rules": [], "named_files": 0}


def test_an_images_cached_reading_is_forgotten(tmp_path):
    """The text OCR or the vision model read from an image is cached by image hash; keeping the
    image by name only deletes that cached text too (ingestion/images.forget_readings)."""
    import pymupdf

    from ingestion.figures import Figure, LocalReader
    from ingestion.images import forget_readings, load_pages

    image = tmp_path / "id.png"
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    pixmap.clear_with(200)
    pixmap.save(str(image))
    settings = SimpleNamespace(data_dir=tmp_path / "data", vision_model="Qwen/Qwen3-VL-32B-Instruct", image_max_side=2000)
    cache = Path(settings.data_dir) / "cache" / "figures"
    (page, mime), = load_pages(image)
    figure = Figure(page=1, bbox=(0, 0, 0, 0), kind="image", context=image.name, image=page, mime=mime)
    readings = []
    for mode, model in (("local", LocalReader.model), ("cloud", settings.vision_model)):
        key = figure.key(mode, model)
        readings.append(_write(cache / key[:2] / f"{key}.txt", "身份证号 SECRET-OCR"))
    other = _write(cache / "ab" / "abcdef.txt", "another image")
    assert forget_readings(image, settings) == 2
    assert not any(path.exists() for path in readings) and other.exists()
