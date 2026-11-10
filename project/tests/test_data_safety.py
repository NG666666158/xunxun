"""The start-up check, backups and the two ways back (core/data_safety.py, design/DATA_SAFETY.md)."""

import json
import os
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from api import main
from api.errors import describe_error, is_damage
from core import data_safety
from core.conversations import ConversationCheckpointer, ConversationStore
from core.data_safety import DataSafety, DatabaseDamaged, EXTRA_DATABASES, RecoveryError, check_file, read_registry
from core.rag_system import RAGSystem
from core.user_settings import load_preferences
from config import Settings
from db.parent_store_manager import ParentStoreManager
from library import FileRegistry, LibraryService


def _settings(tmp_path, **extra):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    values = dict(
        data_dir=data, embedded_db_path=data / "knowledge.sqlite3", registry_path=data / "knowledge.sqlite3",
        backup_keep=3, backup_interval_hours=24.0, backup_first_delay_seconds=600.0, app_version="1.1.0",
    )
    return SimpleNamespace(**{**values, **extra})


def _library_db(settings, files=("a.md", "b.md")):
    """A registry with one watched folder and a few files, written like the app does (WAL)."""
    registry = FileRegistry(settings.embedded_db_path)
    root = registry.add_root(str(settings.data_dir / "docs"), volume_id=1)
    rows = [
        registry.insert_file(str(settings.data_dir / "docs" / name), root_id=root["root_id"], status="indexed",
                             indexed_sha256="abc", chunks=7, characters=70)
        for name in files
    ]
    registry.close()
    return root, rows


def _damage(path: Path):
    """Overwrite the middle of the file, as a failing disk would; page 1 is left alone."""
    size = path.stat().st_size
    with path.open("r+b") as handle:
        handle.seek(4096)
        handle.write(os.urandom(max(4096, size - 8192)))


def _safety(settings):
    safety = DataSafety(settings)
    # "last run" is this version unless a test says otherwise: no upgrade backup in the way.
    (Path(settings.data_dir) / "preferences.json").write_text(json.dumps({"last_run_version": settings.app_version}), encoding="utf-8")
    return safety


def test_a_good_database_passes_and_a_missing_one_is_fine(tmp_path):
    settings = _settings(tmp_path)
    assert check_file(settings.embedded_db_path) == ""
    _library_db(settings)
    assert check_file(settings.embedded_db_path) == ""
    safety = _safety(settings)
    safety.prepare()
    assert safety.status()["state"] == "ok"
    safety.open_for_use()


def test_a_damaged_database_is_found_and_not_opened(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    _damage(settings.embedded_db_path)
    safety = _safety(settings)
    safety.prepare()
    status = safety.status()
    assert status["state"] == "damaged" and "索引库已损坏" in status["message"]
    with pytest.raises(DatabaseDamaged):
        safety.open_for_use()


def test_something_that_is_not_a_database_is_damage(tmp_path):
    path = tmp_path / "x.sqlite3"
    path.write_bytes(b"this is not sqlite" * 100)
    assert "not a database" in check_file(path)


def test_a_locked_database_is_not_called_damaged(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    _library_db(settings)
    real_connect = sqlite3.connect

    class Locked:
        def execute(self, _sql):
            raise sqlite3.OperationalError("database is locked")

        def close(self):
            pass

    monkeypatch.setattr(data_safety.sqlite3, "connect", lambda *a, **k: Locked())
    assert check_file(settings.embedded_db_path) == ""
    monkeypatch.setattr(data_safety.sqlite3, "connect", real_connect)


def test_backup_is_verified_and_holds_rows_still_in_the_wal(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    writer = sqlite3.connect(settings.embedded_db_path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE IF NOT EXISTS kept(v)")
    writer.execute("INSERT INTO kept VALUES ('only in the wal')")
    writer.commit()
    safety = _safety(settings)
    backup = safety.backup(reason="manual")
    writer.close()
    assert backup["reason"] == "manual" and backup["size"] > 0
    copy = settings.data_dir / "backups" / backup["name"] / "knowledge.sqlite3"
    assert check_file(copy, immutable=True) == ""
    conn = sqlite3.connect(copy)
    assert conn.execute("SELECT v FROM kept").fetchone()[0] == "only in the wal"
    assert conn.execute("SELECT count(*) FROM files").fetchone()[0] == 2
    conn.close()
    assert not list((settings.data_dir / "backups").glob(".tmp-*"))


def test_only_the_newest_backups_are_kept(tmp_path):
    settings = _settings(tmp_path, backup_keep=2)
    _library_db(settings)
    safety = _safety(settings)
    made = []
    for second in range(3):
        made.append(safety.backup()["name"])
        time.sleep(1.05)
    names = [item["name"] for item in safety.list_backups()]
    assert names == [made[2], made[1]]
    assert not (settings.data_dir / "backups" / made[0]).exists()


def test_backups_cover_the_conversations_database_when_it_exists(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    sqlite3.connect(settings.data_dir / "conversations.sqlite3").executescript("CREATE TABLE c(v); INSERT INTO c VALUES (1);")
    backup = _safety(settings).backup()
    assert sorted(backup["files"]) == ["conversations.sqlite3", "knowledge.sqlite3"]


def test_not_enough_disk_space_skips_the_backup_and_says_so(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    _library_db(settings)
    safety = _safety(settings)
    monkeypatch.setattr(data_safety.shutil, "disk_usage", lambda _p: SimpleNamespace(free=10))
    assert safety.backup() is None
    assert "磁盘空间不足" in safety.status()["backup_note"]
    assert safety.list_backups() == []


def test_a_backup_of_a_damaged_database_is_refused_and_raises_the_alarm(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    safety = _safety(settings)
    safety.prepare()
    _damage(settings.embedded_db_path)
    assert safety.backup() is None
    assert safety.list_backups() == []
    assert safety.status()["state"] == "damaged"


def test_an_unfinished_backup_never_counts_and_is_cleared_later(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    safety = _safety(settings)
    leftover = settings.data_dir / "backups" / ".tmp-20260101-000000"
    leftover.mkdir(parents=True)
    no_manifest = settings.data_dir / "backups" / "20260101-000000"
    no_manifest.mkdir()
    assert safety.list_backups() == []
    safety.backup()
    assert not leftover.exists()
    assert [b["name"] for b in safety.list_backups()] != ["20260101-000000"]


def test_first_start_after_an_upgrade_keeps_a_copy_first(tmp_path):
    settings = _settings(tmp_path, app_version="1.1.0")
    _library_db(settings)
    safety = DataSafety(settings)
    safety.prepare()
    backups = safety.list_backups()
    assert [(b["reason"], b["app_version"]) for b in backups] == [("upgrade", "1.0.0")]
    assert load_preferences(settings)["last_run_version"] == "1.1.0"
    # The next start of the same version makes none.
    again = DataSafety(settings)
    again.prepare()
    assert len(again.list_backups()) == 1


def test_a_new_install_has_nothing_to_back_up(tmp_path):
    settings = _settings(tmp_path)
    safety = DataSafety(settings)
    safety.prepare()
    assert safety.list_backups() == []
    assert load_preferences(settings)["last_run_version"] == "1.1.0"


def test_restore_puts_the_backup_back_and_sets_the_damaged_file_aside(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    safety = _safety(settings)
    name = safety.backup()["name"]
    _damage(settings.embedded_db_path)
    (settings.data_dir / "knowledge.sqlite3-wal").write_bytes(b"old wal")

    result = safety.restore(name)
    assert result == {"restart_required": False, "action": "restore", "name": name}
    assert safety.status()["state"] == "ok"
    assert not (settings.data_dir / "knowledge.sqlite3-wal").exists()
    replaced = settings.data_dir / "backups" / "replaced"
    assert (replaced / "knowledge.sqlite3").exists() and (replaced / "knowledge.sqlite3-wal").exists()
    registry = FileRegistry(settings.embedded_db_path)
    assert len(registry.list_files()) == 2
    registry.close()


def test_restore_leaves_a_healthy_conversations_database_alone(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    conversations = settings.data_dir / "conversations.sqlite3"
    conn = sqlite3.connect(conversations)
    conn.executescript("CREATE TABLE c(v); INSERT INTO c VALUES ('old');")
    conn.close()
    safety = _safety(settings)
    name = safety.backup()["name"]
    conn = sqlite3.connect(conversations)
    conn.execute("INSERT INTO c VALUES ('newer than the backup')")
    conn.commit()
    conn.close()

    safety.restore(name)
    conn = sqlite3.connect(conversations)
    assert conn.execute("SELECT count(*) FROM c").fetchone()[0] == 2
    conn.close()


def test_restore_refuses_an_unknown_or_damaged_backup(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    safety = _safety(settings)
    with pytest.raises(RecoveryError, match="找不到"):
        safety.restore("20260101-000000")
    name = safety.backup()["name"]
    (settings.data_dir / "backups" / name / "knowledge.sqlite3").write_bytes(b"garbage" * 1000)
    with pytest.raises(RecoveryError, match="也损坏"):
        safety.restore(name)
    # Nothing was touched.
    assert check_file(settings.embedded_db_path) == ""


def test_while_in_use_a_recovery_waits_for_the_next_start(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    safety = _safety(settings)
    name = safety.backup()["name"]
    safety.open_for_use()
    result = safety.restore(name)
    assert result["restart_required"] is True
    assert safety.status()["pending"] == "restore"
    # Nothing moved yet.
    assert not (settings.data_dir / "backups" / "replaced").exists()

    _damage(settings.embedded_db_path)
    next_start = DataSafety(settings)
    next_start.prepare()
    assert next_start.status()["state"] == "ok"
    assert next_start.status()["pending"] is None
    assert (settings.data_dir / "backups" / "replaced" / "knowledge.sqlite3").exists()


def test_a_recovery_that_fails_does_not_keep_the_app_from_starting(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    (settings.data_dir / data_safety.PENDING_FILE).write_text(json.dumps({"action": "restore", "name": "nope"}), encoding="utf-8")
    safety = _safety(settings)
    safety.prepare()
    assert safety.status()["state"] == "ok"
    assert safety.status()["pending"] is None


def test_rebuild_keeps_the_file_list_and_marks_everything_for_reading(tmp_path):
    settings = _settings(tmp_path)
    root, rows = _library_db(settings)
    registry = FileRegistry(settings.embedded_db_path)
    registry.update_file(rows[0]["file_id"], status="unsupported")
    gone = registry.insert_file(str(settings.data_dir / "docs" / "gone.md"), root_id=root["root_id"], status="indexed")
    registry.soft_delete_file(gone["file_id"])
    registry.close()

    safety = _safety(settings)
    result = safety.rebuild()
    assert result["action"] == "rebuild" and result["folders"] == 1 and result["files"] == 3
    assert safety.status()["state"] == "ok"
    registry = FileRegistry(settings.embedded_db_path)
    assert [r["root_id"] for r in registry.list_roots()] == [root["root_id"]]
    live = {r["file_id"]: r for r in registry.list_files()}
    assert live[rows[0]["file_id"]]["status"] == "unsupported"
    second = live[rows[1]["file_id"]]
    assert second["status"] == "pending" and second["indexed_sha256"] is None and second["chunks"] == 0
    # Deleted rows stay as tombstones.
    assert gone["file_id"] not in live and registry.get_file(gone["file_id"], include_deleted=True)
    registry.close()


def test_rebuild_of_an_unreadable_file_falls_back_to_the_newest_backup(tmp_path):
    settings = _settings(tmp_path)
    _root, rows = _library_db(settings)
    safety = _safety(settings)
    safety.backup()
    settings.embedded_db_path.write_bytes(b"garbage" * 4096)
    assert read_registry(settings.embedded_db_path) == ([], [])

    result = safety.rebuild()
    assert result["files"] == len(rows)
    assert (settings.data_dir / "backups" / "replaced" / "knowledge.sqlite3").read_bytes().startswith(b"garbage")


def test_rebuild_salvages_rows_from_a_damaged_file(tmp_path):
    settings = _settings(tmp_path)
    many = [f"f{i}.md" for i in range(400)]
    _library_db(settings, files=many)
    registry = FileRegistry(settings.embedded_db_path)
    registry._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    registry.close()
    size = settings.embedded_db_path.stat().st_size
    assert size > 40960
    # One page in the middle of the file.
    with settings.embedded_db_path.open("r+b") as handle:
        handle.seek(size // 2 // 4096 * 4096)
        handle.write(b"\xff" * 4096)
    _roots, files = read_registry(settings.embedded_db_path)
    assert 300 < len(files) <= 400


def test_after_a_rebuild_the_library_reads_every_file_again(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha", encoding="utf-8")
    (docs / "b.md").write_text("beta", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    settings = _settings(
        tmp_path, allowed_extensions=(".md", ".txt"), max_upload_mb=1, max_local_file_mb=1024, max_local_text_mb=50, upload_dir=data / "uploads", parsed_dir=data / "parsed",
        parent_store_dir=data / "parents", watch_max_files=1000, image_import_mode="ask", image_understanding="off",
        vision_concurrency=6, app_token="",
    )
    for path in (settings.upload_dir, settings.parsed_dir, settings.parent_store_dir):
        path.mkdir()

    class Store:
        def __init__(self):
            self.documents = {}

        def has_document(self, doc_id):
            return doc_id in self.documents

        def delete_document(self, doc_id):
            self.documents.pop(doc_id, None)

        def rename_source(self, doc_id, source):
            self.documents[doc_id] = source

        def list_documents(self):
            return []

    def start():
        store = Store()

        def indexer(path, file_id, source, image_mode=None, pause=None):
            store.documents[file_id] = source
            return 1, 5, {}

        library = LibraryService(settings, FileRegistry(settings.embedded_db_path), store, ParentStoreManager(settings), indexer)
        library.start(watch=False)
        return library, store

    def wait_until_read(library, store):
        deadline = time.time() + 15
        while time.time() < deadline:
            counts = library.registry.status_counts()
            if counts.get("indexed") == 2 and len(store.documents) == 2:
                return
            time.sleep(0.05)
        raise AssertionError(library.registry.status_counts())

    library, store = start()
    library.add_root(str(docs))
    wait_until_read(library, store)
    library.stop()
    library.registry.close()

    safety = _safety(settings)
    safety.rebuild()

    library, store = start()
    try:
        wait_until_read(library, store)
        assert sorted(store.documents.values()) == ["a.md", "b.md"]
    finally:
        library.stop()
        library.registry.close()


# ---------------- the conversations, with the real stores ----------------


def _conversation(conversation_id, text="问题"):
    return {"id": conversation_id, "title": text, "updated": 1.0, "messages": [{"role": "user", "text": text}]}


def _state(checkpointer, thread_id):
    checkpoint = {"v": 1, "id": f"1ef-{thread_id}", "ts": "2026-11-05T00:00:00+00:00", "channel_values": {"question": thread_id},
                  "channel_versions": {}, "versions_seen": {}}
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpointer.put(config, checkpoint, {}, {})


def _has_state(path, thread_id):
    checkpointer = ConversationCheckpointer(path)
    try:
        return checkpointer.get_tuple({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}) is not None
    finally:
        checkpointer.close()


def _ids(path):
    store = ConversationStore(path)
    try:
        return {item["id"] for item in store.list()}
    finally:
        store.close()


def test_the_app_keeps_conversations_where_the_backups_look():
    settings = Settings(data_dir=Path("somewhere"), _env_file=None)
    assert Path(settings.conversations_db_path) == settings.data_dir / EXTRA_DATABASES[0]


def test_a_backup_holds_conversations_and_state_still_in_the_wal(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    path = settings.data_dir / "conversations.sqlite3"
    store, checkpointer = ConversationStore(path), ConversationCheckpointer(path)
    store.save(_conversation("kept"))
    _state(checkpointer, "kept")
    assert Path(f"{path}-wal").stat().st_size > 0

    backup = _safety(settings).backup()
    store.close()
    checkpointer.close()

    copy = settings.data_dir / "backups" / backup["name"] / path.name
    assert "conversations.sqlite3" in backup["files"]
    assert not Path(f"{copy}-wal").exists()
    assert _ids(copy) == {"kept"}
    assert _has_state(copy, "kept")


@pytest.mark.parametrize("while_running", [False, True])
def test_restore_does_not_take_back_newer_conversations(tmp_path, while_running):
    settings = _settings(tmp_path)
    _library_db(settings)
    path = settings.data_dir / "conversations.sqlite3"
    store, checkpointer = ConversationStore(path), ConversationCheckpointer(path)
    store.save(_conversation("before"))
    _state(checkpointer, "before")
    safety = _safety(settings)
    name = safety.backup()["name"]
    store.save(_conversation("after the backup"))
    _state(checkpointer, "after the backup")
    store.delete("before")
    store.close()
    checkpointer.close()

    if while_running:
        safety.open_for_use()
        assert safety.restore(name)["restart_required"] is True
        safety = _safety(settings)
        safety.prepare()
    else:
        assert safety.restore(name)["restart_required"] is False

    assert safety.status()["state"] == "ok" and safety.status()["notice"] == ""
    assert _ids(path) == {"after the backup"}
    assert _has_state(path, "after the backup")


def test_rebuild_leaves_the_conversations_alone(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    path = settings.data_dir / "conversations.sqlite3"
    store = ConversationStore(path)
    store.save(_conversation("kept"))
    store.close()
    safety = _safety(settings)
    _damage(settings.embedded_db_path)
    safety.prepare()
    assert safety.status()["state"] == "damaged"
    safety.rebuild()
    assert safety.status()["state"] == "ok"
    assert _ids(path) == {"kept"}


def _damaged_conversations(settings, *, backed_up=True):
    """Conversations written, maybe backed up, written again, then damaged while the app was closed."""
    path = settings.data_dir / "conversations.sqlite3"
    store = ConversationStore(path)
    # Enough rows that the file has pages past the first for _damage to hit.
    for number in range(200):
        store.save(_conversation(f"old-{number}", "很长的问题" * 200))
    safety = _safety(settings)
    if backed_up:
        safety.backup()
    store.save(_conversation("lost with the file"))
    store.close()
    _damage(path)
    assert check_file(path)
    return path


def test_damaged_conversations_go_back_to_their_newest_good_copy_and_the_library_opens(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    path = _damaged_conversations(settings)

    safety = _safety(settings)
    safety.open_for_use()  # would raise DatabaseDamaged if this counted as a damaged index

    status = safety.status()
    assert status["state"] == "ok"
    assert "对话记录文件损坏了" in status["notice"] and "备份" in status["notice"]
    assert (settings.data_dir / "backups" / "replaced" / path.name).exists()
    ids = _ids(path)
    assert "old-0" in ids and "lost with the file" not in ids
    assert not list(settings.data_dir.glob("*.restoring"))


def test_damaged_conversations_without_a_backup_start_empty_and_say_so(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    path = _damaged_conversations(settings, backed_up=False)

    safety = _safety(settings)
    safety.open_for_use()

    assert "没有可用的备份" in safety.status()["notice"]
    assert (settings.data_dir / "backups" / "replaced" / path.name).exists()
    assert _ids(path) == set()


def test_a_damaged_backup_of_the_conversations_is_skipped(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    path = _damaged_conversations(settings)
    good = _safety(settings).list_backups()[0]["name"]
    newer = settings.data_dir / "backups" / "20991231-000000"
    newer.mkdir()
    (newer / path.name).write_bytes((settings.data_dir / "backups" / good / path.name).read_bytes())
    _damage(newer / path.name)
    (newer / "manifest.json").write_text(json.dumps({"created_at": "2099-12-31T00:00:00", "files": [path.name]}), encoding="utf-8")

    safety = _safety(settings)
    safety.prepare()
    assert "old-0" in _ids(path)
    assert "12月31日" not in safety.status()["notice"]


def test_a_backup_while_the_conversations_are_damaged_still_covers_the_index(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    safety = _safety(settings)
    safety.prepare()
    _damaged_conversations(settings, backed_up=False)

    backup = safety.backup()

    assert backup["files"] == ["knowledge.sqlite3"]
    assert safety.status()["state"] == "ok"
    assert safety.status()["notice"]


def test_a_query_error_is_put_on_the_file_that_is_actually_damaged(tmp_path):
    settings = _settings(tmp_path)
    _library_db(settings)
    safety = _safety(settings)
    safety.prepare()

    path = _damaged_conversations(settings, backed_up=False)
    safety.note_damage(sqlite3.DatabaseError("database disk image is malformed"))
    assert safety.status()["state"] == "ok" and safety.status()["notice"]

    path.unlink()
    _damage(settings.embedded_db_path)
    safety.note_damage(sqlite3.DatabaseError("database disk image is malformed"))
    assert safety.status()["state"] == "damaged"


def test_sqlite_words_for_damage(tmp_path):
    assert is_damage(sqlite3.DatabaseError("database disk image is malformed"))
    assert is_damage(sqlite3.DatabaseError("file is not a database"))
    assert not is_damage(sqlite3.OperationalError("database is locked"))
    assert not is_damage(ValueError("malformed"))
    assert describe_error(sqlite3.DatabaseError("database disk image is malformed"))[0] == 503
    assert describe_error(DatabaseDamaged("索引库已损坏"))[0] == 503


# ---------------- over HTTP ----------------


@pytest.fixture
def api(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", siliconflow_api_key="", _env_file=None)
    settings.ensure_directories()
    system = RAGSystem(settings)
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "system", system)
    monkeypatch.setattr(main, "_show_in_explorer", lambda path: None)
    monkeypatch.setattr(main, "_start_library", lambda: None)
    return TestClient(main.app), settings, system


def test_settings_still_answer_on_a_damaged_index(api):
    client, settings, _system = api
    (settings.data_dir / "preferences.json").write_text(json.dumps({"last_run_version": settings.app_version}), encoding="utf-8")
    _library_db(settings)
    _damage(settings.embedded_db_path)
    payload = client.get("/api/v1/settings").json()
    assert payload["data_safety"]["state"] == "damaged"
    assert payload["suggestions"]["blocked_files"] == []
    assert client.get("/api/v1/data-safety").json()["state"] == "damaged"


def test_history_still_answers_on_a_damaged_index(api):
    client, settings, _system = api
    (settings.data_dir / "preferences.json").write_text(json.dumps({"last_run_version": settings.app_version}), encoding="utf-8")
    _library_db(settings)
    store = ConversationStore(settings.conversations_db_path)
    store.save(_conversation("kept"))
    store.close()
    _damage(settings.embedded_db_path)

    assert [item["id"] for item in client.get("/api/v1/conversations").json()["conversations"]] == ["kept"]
    assert client.get("/api/v1/data-safety").json()["state"] == "damaged"


def test_a_recovery_waiting_for_this_start_runs_before_the_conversations_open(api):
    client, settings, _system = api
    (settings.data_dir / "preferences.json").write_text(json.dumps({"last_run_version": settings.app_version}), encoding="utf-8")
    _library_db(settings)
    (settings.data_dir / "recovery-pending.json").write_text(json.dumps({"action": "rebuild"}), encoding="utf-8")

    assert client.get("/api/v1/conversations").json()["conversations"] == []
    assert not (settings.data_dir / "recovery-pending.json").exists()


def test_restore_over_http_then_the_index_is_usable(api):
    client, settings, system = api
    (settings.data_dir / "preferences.json").write_text(json.dumps({"last_run_version": settings.app_version}), encoding="utf-8")
    _library_db(settings)
    made = client.post("/api/v1/data-safety/backup").json()
    name = made["backups"][0]["name"]
    # The backup made that connection-free run; damage the file before anything opens it.
    _damage(settings.embedded_db_path)
    system.safety._check()
    assert client.post("/api/v1/data-safety/backup").status_code == 409

    done = client.post("/api/v1/data-safety/restore", json={"name": name}).json()
    assert done["restart_required"] is False and done["data_safety"]["state"] == "ok"
    assert client.post("/api/v1/data-safety/restore", json={"name": "../evil"}).status_code == 422
    assert client.post("/api/v1/data-safety/restore", json={"name": "20200101-000000"}).status_code == 409


def test_rebuild_over_http(api):
    client, settings, system = api
    (settings.data_dir / "preferences.json").write_text(json.dumps({"last_run_version": settings.app_version}), encoding="utf-8")
    _library_db(settings)
    _damage(settings.embedded_db_path)
    system.safety.prepare()
    done = client.post("/api/v1/data-safety/rebuild").json()
    assert done["action"] == "rebuild" and done["data_safety"]["state"] == "ok"


def test_guided_start_flags_are_remembered(api):
    client, _settings_, _system = api
    first = client.get("/api/v1/settings").json()
    assert first["onboarding_done"] is False and first["data_notice_seen"] is False
    saved = client.put("/api/v1/settings", json={"onboarding_done": True, "data_notice_seen": True}).json()
    assert saved["onboarding_done"] is True and saved["data_notice_seen"] is True
    assert client.get("/api/v1/settings").json()["onboarding_done"] is True


