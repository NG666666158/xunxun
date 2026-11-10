"""关于 › 导出诊断信息: logs and a summary, never keys or document text (core/diagnostics.py)."""

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from fastapi.testclient import TestClient

from api import main
from config import Settings
from core import diagnostics
from core.diagnostics import scrub_log, scrub_text
from core.rag_system import RAGSystem

KEY = "sk-abcdefghijklmnopqrstuvwxyz012345"
DOCUMENT_TEXT = "第三条员工年假申请须提前五个工作日提交直属负责人审批逾期视为自动放弃当年额度不予补发"
HEADLINE = "2026-11-05 10:00:00,123 WARNING rag_agent.nodes: "


def test_keys_and_tokens_are_removed():
    for line in (
        f"{HEADLINE}calling with {KEY}",
        f"{HEADLINE}Authorization: Bearer abc.def-ghi",
        f"{HEADLINE}api_key=hunter2hunter2",
        f'{HEADLINE}{{"token": "per-launch-secret"}}',
        f"{HEADLINE}X-App-Token: per-launch-secret",
    ):
        out = "\n".join(scrub_log([line]))
        assert "[已隐去]" in out
        for secret in (KEY, "abc.def-ghi", "hunter2", "per-launch-secret"):
            assert secret not in out


def test_the_stored_key_is_removed_whatever_it_looks_like():
    assert "custom-key-format" not in scrub_text("key custom-key-format here", ["custom-key-format"])


def test_the_home_folder_is_hidden_in_any_case():
    out = scrub_text(r"Indexing C:\USERS\me\Documents\a.pdf failed", home=r"C:\Users\me")
    assert out == r"Indexing %USERPROFILE%\Documents\a.pdf failed"


def test_lines_outside_the_allow_list_are_dropped_and_counted():
    log = [
        f"{HEADLINE}Answer generation failed\n",
        "Traceback (most recent call last):\n",
        '  File "D:\\x\\nodes.py", line 12, in answer\n',
        "    result = chain.invoke(prompt)\n",
        f"langchain_core.exceptions.OutputParserException: Failed to parse: {DOCUMENT_TEXT}\n",
        f"{DOCUMENT_TEXT}\n",
        "more of the model's reply on another line\n",
    ]
    out = scrub_log(log)
    joined = "\n".join(out)
    assert DOCUMENT_TEXT not in joined and "model's reply" not in joined
    assert "Traceback (most recent call last):" in joined
    assert 'File "D:\\x\\nodes.py", line 12, in answer' in joined
    # The exception keeps its type, not its message.
    assert "langchain_core.exceptions.OutputParserException" in out
    assert "  [省略 2 行]" in out


def test_document_text_in_a_headline_is_removed():
    for line in (
        f"{HEADLINE}Fact check failed closed: {DOCUMENT_TEXT}",
        f"{HEADLINE}validation error input_value='{DOCUMENT_TEXT}', type=str",
        f"{HEADLINE}Failed to parse completion: {DOCUMENT_TEXT}",
    ):
        assert DOCUMENT_TEXT not in "\n".join(scrub_log([line]))


def test_a_headline_is_capped():
    out = scrub_log([HEADLINE + "x" * 2000])[0]
    assert len(out) <= diagnostics.HEADLINE_LIMIT + 1 and out.endswith("…")


def test_ordinary_lines_survive():
    line = "2026-11-05 10:00:00,123 INFO library.service: Reconciled file registry: {'lost': 0}"
    assert scrub_log([line]) == [line]
    named = f"{HEADLINE}Indexing D:\\资料\\季度报告.pdf failed: file is encrypted"
    assert scrub_log([named]) == [named]


def _export(tmp_path):
    settings = SimpleNamespace(
        data_dir=tmp_path, log_dir=tmp_path / "logs", embedded_db_path=tmp_path / "knowledge.sqlite3", app_version="1.1.0",
        llm_model="m", embedding_model="e", rerank_model="r", answer_mode="strict", image_understanding="off",
        siliconflow_api_key=KEY, app_token="per-launch-secret",
    )
    settings.log_dir.mkdir()
    (settings.log_dir / "backend.log").write_text(f"{HEADLINE}using {KEY}\n{DOCUMENT_TEXT}\n", encoding="utf-8")
    (settings.log_dir / "backend.log.1").write_text(f"{HEADLINE}token=per-launch-secret\n", encoding="utf-8")
    (settings.log_dir / "other.txt").write_text("not a log", encoding="utf-8")
    (tmp_path / "knowledge.sqlite3").write_bytes(b"sqlite bytes " + DOCUMENT_TEXT.encode())
    (tmp_path / "preferences.json").write_text("{}", encoding="utf-8")
    safety = {"state": "ok", "checked_at": "t", "check_ms": 3, "backups": [{"name": "n", "created_at": "c", "size": 9, "reason": "auto"}]}
    return settings, diagnostics.export(settings, safety, {"indexed": 4})


def test_the_zip_holds_logs_and_a_summary_only(tmp_path):
    _settings, path = _export(tmp_path)
    assert path.parent == tmp_path / "diagnostics" and path.name.startswith("寻寻诊断-")
    with zipfile.ZipFile(path) as archive:
        names = sorted(archive.namelist())
        assert names == ["about.json", "logs/backend.log", "logs/backend.log.1"]
        everything = "".join(archive.read(name).decode("utf-8") for name in names)
        about = json.loads(archive.read("about.json"))
    assert KEY not in everything and "per-launch-secret" not in everything and DOCUMENT_TEXT not in everything
    assert about["app_version"] == "1.1.0" and about["files_by_status"] == {"indexed": 4}
    assert about["data_safety"]["backups"][0]["size"] == 9
    assert "name" not in about["data_safety"]["backups"][0]


def test_only_a_few_exports_are_kept(tmp_path):
    settings, _first = _export(tmp_path)
    folder = tmp_path / "diagnostics"
    for index in range(5):
        (folder / f"old-{index}.zip").write_bytes(b"x")
    diagnostics.export(settings, {}, {})
    assert len(list(folder.glob("*.zip"))) == diagnostics.KEEP_EXPORTS


def test_export_over_http_shows_the_file(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", siliconflow_api_key=KEY, _env_file=None)
    settings.ensure_directories()
    (Path(settings.log_dir) / "backend.log").write_text(f"{HEADLINE}hello\n", encoding="utf-8")
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "system", RAGSystem(settings))
    shown = []
    monkeypatch.setattr(main, "_show_in_explorer", shown.append)
    payload = TestClient(main.app).post("/api/v1/diagnostics/export").json()
    assert shown == [Path(payload["path"])] and payload["size"] > 0
