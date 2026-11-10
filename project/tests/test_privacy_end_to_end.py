"""Nothing of a file kept by name only or excluded leaves the machine (design/PRIVACY.md).

The real app (RAGSystem: parser, chunker, embeddings, retrieval, answering, file search,
suggestions) runs against the fake model service of the smoke tests, wrapped so that it
records every request body it receives: embeddings, reranking and chat. The assertions are on
those recorded inputs. Then a file that was read is switched to name-only: its text must leave
the index file and the backups, and later questions must not send it either.
"""

from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).parent / "smoke"))
from fake_models import create_app  # noqa: E402

from config import Settings  # noqa: E402
from core.rag_system import RAGSystem  # noqa: E402

uvicorn = pytest.importorskip("uvicorn")

# Markers inside the files' text; none of them may ever be in a request.
PASSPORT = "SECRET-PASSPORT-QZXA"   # 证件/ : kept by name only
RENT = "SECRET-RENT-QZXB"           # 合同/ : excluded folder
KEY = "SECRET-KEY-QZXC"             # *.key.txt : excluded pattern
NOTES = "SECRET-NOTES-QZXD"         # read first, then switched to name only
NAME_ONLY_FILE = "ZJQQ护照扫描.txt"   # its name must not be sent either


class Recorder:
    """ASGI wrapper keeping every request's path and JSON body (decoded, so escaped Chinese
    and plain text compare the same)."""

    def __init__(self, app):
        self.app = app
        self.requests: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks: list[bytes] = []

        async def recording_receive():
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
            return message

        await self.app(scope, recording_receive, send)
        body = b"".join(chunks).decode("utf-8", "replace")
        try:
            body = json.dumps(json.loads(body), ensure_ascii=False)
        except ValueError:
            pass
        with self.lock:
            self.requests.append((scope["path"], body))

    def sent(self) -> str:
        with self.lock:
            return "\n".join(body for _, body in self.requests)

    def clear(self) -> None:
        with self.lock:
            self.requests.clear()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def model_service():
    recorder = Recorder(create_app())
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(recorder, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="fake-models", daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "fake model service did not start"
    yield recorder, f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=10)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def app(tmp_path, model_service, monkeypatch):
    recorder, base_url = model_service
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        siliconflow_api_key="sk-test-privacy",
        siliconflow_base_url=base_url,
        watch_enabled=False,
        request_max_retries=0,
    )
    settings.ensure_directories()
    rag = RAGSystem(settings).initialize()
    recorder.clear()
    yield rag, recorder
    rag.library.stop()


def _desk(tmp_path: Path) -> Path:
    desk = tmp_path / "资料"
    _write(desk / "公开" / "制度.md", "# 差旅制度\n\n差旅报销要在十五个工作日内提交，逾期不予受理。\n")
    _write(desk / "证件" / NAME_ONLY_FILE, f"# 护照\n\n护照号码 {PASSPORT}，签发地北京，有效期十年。\n")
    _write(desk / "合同" / "租房合同.txt", f"租房合同\n\n月租金 {RENT} 元，押一付三。\n")
    _write(desk / "服务器.key.txt", f"私钥 {KEY}\n")
    _write(desk / "笔记.md", f"# 项目笔记\n\n会议记录：新项目的代号是 {NOTES}，下周启动。\n")
    return desk


def _ask(rag, question: str) -> dict:
    return rag.chat(question, rag.new_thread_id())


def _file_bytes(path: Path) -> bytes:
    data = path.read_bytes() if path.exists() else b""
    wal = path.with_name(path.name + "-wal")
    return data + (wal.read_bytes() if wal.exists() else b"")


def test_name_only_and_excluded_files_never_reach_the_model_service(app, tmp_path):
    rag, recorder = app
    library = rag.library
    desk = _desk(tmp_path)

    # Added, then the user marks 证件 name-only and excludes 合同 and *.key.txt before it is read.
    root = library.add_root(str(desk), background=False)
    library.set_folder_mode(str(desk / "证件"), "name")
    library.add_rule(action="exclude", path=str(desk / "合同"))
    library.add_rule(action="exclude", pattern="*.key.txt", root_id=root["root_id"])
    library.run_until_idle()
    rag.backfill_file_entries()
    rag.suggestions.sync()
    rag.suggestions.run_pending()

    files = {row["name"]: row for row in library.registry.list_files()}
    assert set(files) == {"制度.md", NAME_ONLY_FILE, "笔记.md"}
    assert files[NAME_ONLY_FILE]["status"] == "named" and files["制度.md"]["status"] == "indexed"

    # Questions that would retrieve the secrets if they were indexed.
    answers = [
        _ask(rag, "护照号码是多少？"),
        _ask(rag, "租房合同的月租金是多少？"),
        _ask(rag, "私钥是什么？"),
        _ask(rag, "差旅报销要在多少个工作日内提交？"),
    ]
    assert "十五个工作日" in answers[-1]["answer"] and answers[-1]["citations"]
    # The name-only file is still found by its name, on this machine.
    found = rag.search_files("护照扫描")
    assert [item["name"] for item in found["files"]][:1] == [NAME_ONLY_FILE]
    assert found["files"][0]["status"] == "named"

    sent = recorder.sent()
    paths = {path for path, _ in recorder.requests}
    assert {"/v1/embeddings", "/v1/chat/completions"} <= paths  # the app did talk to the service
    assert "十五个工作日" in sent and NOTES in sent                  # ... about the files it may read
    for secret in (PASSPORT, RENT, KEY, "ZJQQ"):
        assert secret not in sent, f"{secret} was sent to the model service"
    for answer in answers:
        assert all(secret not in answer["answer"] for secret in (PASSPORT, RENT, KEY))


def test_switching_a_read_file_to_name_only_purges_its_text_everywhere(app, tmp_path):
    rag, recorder = app
    library = rag.library
    desk = _desk(tmp_path)
    library.add_root(str(desk), background=False)
    library.run_until_idle()
    notes = next(row for row in library.registry.list_files() if row["name"] == "笔记.md")
    assert NOTES in recorder.sent()  # it was read: expected, before the switch

    backup = rag.safety.backup()
    backup_db = rag.settings.data_dir / "backups" / backup["name"] / "knowledge.sqlite3"
    live_db = Path(rag.settings.embedded_db_path)
    assert NOTES.encode() in _file_bytes(live_db) and NOTES.encode() in _file_bytes(backup_db)

    impact = library.set_content_mode(notes["file_id"], "name")
    assert impact["indexed"] == 1 and impact["file"]["status"] == "named"
    assert rag.scrubber.status()["pending"] == 1
    assert rag.scrubber.run_now()
    assert rag.scrubber.status()["pending"] == 0

    # Gone from the live index file (and its log), the backup, the parsed text and the parents.
    for path in (live_db, backup_db):
        data = _file_bytes(path)
        assert NOTES.encode() not in data and b"qzxd" not in data.lower(), f"text left in {path.name}"
    assert not (rag.settings.parsed_dir / f"{notes['file_id']}.json").exists()
    assert not rag.parent_store.has_document(notes["file_id"])
    assert not rag.store.has_document(notes["file_id"])
    # Restoring that backup would not bring the old choice back: its registry says name-only.
    import sqlite3

    conn = sqlite3.connect(backup_db)
    try:
        row = conn.execute("SELECT status, content_mode, chunks FROM files WHERE file_id = ?", (notes["file_id"],)).fetchone()
    finally:
        conn.close()
    assert row == ("named", "name", 0)

    # From now on questions about it send nothing of it, and nothing of it comes back.
    recorder.clear()
    library.run_until_idle()
    answer = _ask(rag, "新项目的代号是什么？")
    assert NOTES not in recorder.sent() and NOTES not in answer["answer"]

    # Read in full again after the user confirms: then it is sent, as the confirmation says.
    library.set_content_mode(notes["file_id"], "full")
    library.run_until_idle()
    assert NOTES in recorder.sent()
    assert library.registry.get_file(notes["file_id"])["status"] == "indexed"


def test_removed_files_leave_the_backups_too(app, tmp_path):
    rag, _recorder = app
    library = rag.library
    desk = _desk(tmp_path)
    root = library.add_root(str(desk), background=False)
    library.run_until_idle()
    backup = rag.safety.backup()
    backup_db = rag.settings.data_dir / "backups" / backup["name"] / "knowledge.sqlite3"
    assert RENT.encode() in _file_bytes(backup_db)
    # Excluding a folder that was read, and removing a whole watched folder, both purge.
    library.add_rule(action="exclude", path=str(desk / "合同"))
    rag.scrubber.run_now()
    assert RENT.encode() not in _file_bytes(backup_db) and RENT.encode() not in _file_bytes(Path(rag.settings.embedded_db_path))
    assert NOTES.encode() in _file_bytes(backup_db)
    library.remove_root(root["root_id"])
    rag.scrubber.run_now()
    for secret in (NOTES, PASSPORT, "十五个工作日"):
        assert secret.encode() not in _file_bytes(backup_db)
