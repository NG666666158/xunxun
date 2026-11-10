"""HTTP contract of adding files/folders, the image choice and library sync (design/LIBRARY_SYNC_AND_IMAGES.md)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from api import main
from db.parent_store_manager import ParentStoreManager
from library import FileRegistry, LibraryService

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


class FakeStore:
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


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        allowed_extensions=(".md", ".txt", ".pdf", *IMAGE_EXTENSIONS), max_upload_mb=1, max_local_file_mb=1, max_local_text_mb=1, data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads", parsed_dir=tmp_path / "data" / "parsed",
        parent_store_dir=tmp_path / "data" / "parents", watch_max_files=1000, image_import_mode="ask",
        image_understanding="off", vision_concurrency=6, app_token="",
    )
    for path in (settings.upload_dir, settings.parsed_dir, settings.parent_store_dir):
        path.mkdir(parents=True)
    store = FakeStore()

    def indexer(path, file_id, source, image_mode=None, pause=None):
        store.documents[file_id] = source
        return 1, 5, {}

    library = LibraryService(settings, FileRegistry(tmp_path / "data" / "r.sqlite3"), store,
                             ParentStoreManager(settings), indexer)
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "_library", lambda: library)
    yield TestClient(main.app), library, tmp_path
    library.stop()
    library.registry.close()


def _files(tmp_path):
    (tmp_path / "图").mkdir()
    (tmp_path / "图" / "a.png").write_bytes(b"png")
    (tmp_path / "图" / "说明.md").write_text("# 说明", encoding="utf-8")
    (tmp_path / "单个.jpg").write_bytes(b"jpg")
    return [str(tmp_path / "图"), str(tmp_path / "单个.jpg")]


def test_inspect_then_import_with_remembered_choice(client):
    http, library, tmp_path = client
    paths = _files(tmp_path)
    inspected = http.post("/api/v1/library/inspect", json={"paths": paths}).json()
    assert inspected["ask_image_mode"] is True and inspected["totals"]["images"] == 2
    assert set(inspected["estimate"]["images"]) == {"local", "cloud", "name"}

    imported = http.post("/api/v1/library/import", json={"paths": paths, "image_mode": "name", "remember": True})
    assert imported.status_code == 200 and imported.json()["added"] == 2
    prefs = json.loads((tmp_path / "data" / "preferences.json").read_text(encoding="utf-8"))
    assert prefs["image_import_mode"] == "name" and main.settings.image_import_mode == "name"
    assert http.post("/api/v1/library/inspect", json={"paths": [paths[1]]}).json()["ask_image_mode"] is False

    bad = http.post("/api/v1/library/import", json={"paths": paths, "image_mode": "ocr"})
    assert bad.status_code == 422


def test_status_since_lost_list_check_and_image_mode(client):
    http, library, tmp_path = client
    record = library.import_path(str(_write(tmp_path / "笔记.txt")))
    library.run_until_idle()
    seq = http.get("/api/v1/library/status", params={"since": 0}).json()["changes"]["seq"]
    (tmp_path / "笔记.txt").unlink()

    sync = http.post("/api/v1/library/sync").json()
    assert sync["changes"]["counts"] == {"missing": 1}
    status = http.get("/api/v1/library/status", params={"since": seq}).json()
    assert status["changes"]["counts"] == {"missing": 1} and status["lost"]["missing"] == 1
    assert [f["file_id"] for f in http.get("/api/v1/library/lost").json()["files"]] == [record["file_id"]]
    assert http.get("/api/v1/files", params={"status": "missing,recycled"}).json()["total"] == 1
    assert http.post(f"/api/v1/files/{record['file_id']}/check").json()["status"] == "missing"
    assert http.post(f"/api/v1/files/{'0' * 32}/check").status_code == 404

    image = library.import_path(str(_png(tmp_path)))
    changed = http.post(f"/api/v1/files/{image['file_id']}/image-mode", json={"mode": "cloud"}).json()
    assert changed["image_mode"] == "cloud" and changed["image"] is True
    refused = http.post(f"/api/v1/files/{record['file_id']}/image-mode", json={"mode": "cloud"})
    assert refused.status_code == 400


def test_settings_carry_the_image_preference(client, monkeypatch):
    http, library, tmp_path = client
    monkeypatch.setattr(main, "_settings_payload", lambda: {"image_import_mode": main.settings.image_import_mode})
    response = http.put("/api/v1/settings", json={"image_import_mode": "local"})
    assert response.status_code == 200 and response.json()["image_import_mode"] == "local"
    assert http.put("/api/v1/settings", json={"image_import_mode": "maybe"}).status_code == 422


def _write(path: Path, text="内容"):
    path.write_text(text, encoding="utf-8")
    return path


def _png(tmp_path):
    path = tmp_path / "图片.png"
    path.write_bytes(b"png")
    return path


def test_files_carry_their_card_and_filter_by_type(client, monkeypatch):
    """The library page shows each file's type and card (/files), lists the types with counts
    (/library/types) and filters by one; the file's own name wins over its card, as in search."""
    http, library, tmp_path = client
    cards = {}
    library.store.doc_cards = lambda doc_ids=None, types=None: {
        doc_id: card for doc_id, card in cards.items() if doc_ids is None or doc_id in set(doc_ids)}
    library.store.doc_card_hashes = lambda: {doc_id: "h" for doc_id in cards}
    monkeypatch.setattr(main.settings, "doc_cards", True, raising=False)
    resume = library.import_path(str(_write(tmp_path / "张三-agent开发.txt", "教育经历 工作经历")))
    plan = library.import_path(str(_write(tmp_path / "附件（商业计划书）.txt", "项目背景")))
    notes = library.import_path(str(_write(tmp_path / "随手记.txt", "备忘")))
    library.run_until_idle()
    cards[resume["file_id"]] = {"doc_type": "简历", "label": "简历", "subject": "张三", "period": "", "summary": "张三的简历"}
    cards[plan["file_id"]] = {"doc_type": "合同", "label": "合同", "subject": "", "period": "2025年", "summary": "一份通知"}

    listed = {f["name"]: f["card"] for f in http.get("/api/v1/files").json()["files"]}
    assert listed["张三-agent开发.txt"]["type"] == "简历" and listed["张三-agent开发.txt"]["subject"] == "张三"
    assert listed["附件（商业计划书）.txt"]["type"] == "方案"  # the name says 计划书
    assert listed["附件（商业计划书）.txt"]["by_name"] is True and listed["附件（商业计划书）.txt"]["card_type"] == "合同"
    assert listed["随手记.txt"] is None  # still waiting for its card

    only = http.get("/api/v1/files", params={"doc_type": "简历"}).json()
    assert only["total"] == 1 and [f["name"] for f in only["files"]] == ["张三-agent开发.txt"]
    assert http.get("/api/v1/files", params={"doc_type": "合同"}).json()["total"] == 0

    types = http.get("/api/v1/library/types").json()
    assert types["types"] == [{"type": "方案", "count": 1}, {"type": "简历", "count": 1}, {"type": "其他", "count": 1}]
    assert types["pending"] == 1 and types["enabled"] is True
    assert notes["file_id"] not in cards
