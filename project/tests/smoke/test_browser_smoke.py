"""The main flows in a real browser, against the real backend and a fake model service.

A new install walks through the guided start, adds a folder, asks a question, opens the cited
passage, finds the conversation again after a reload, adds a single file, a deck and a workbook,
asks about them as a whole, and finds the data-safety rows. Needs Playwright:

    pip install playwright && playwright install chromium
    python -m pytest project/tests/smoke -q

SMOKE_BROWSER_CHANNEL=msedge uses the installed Edge instead of downloading Chromium;
SMOKE_PYTHON runs the backend with another interpreter. Without Playwright this is skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

sys.path.insert(0, str(Path(__file__).parent))
from office_files import write_deck, write_workbook  # noqa: E402

SERVE = Path(__file__).with_name("serve.py")
QUESTION = "差旅报销要在多少个工作日内提交？"
ANSWER_FRAGMENT = "十五个工作日"
PAGE_TIMEOUT = 60_000


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_live(base: str, process: subprocess.Popen, seconds: float = 90) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"backend exited with {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base}/api/v1/health/live", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)
    raise TimeoutError("backend did not start")


def _get(base: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base}/api/v1{path}", timeout=10) as response:
        return json.load(response)


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    port = _free_port()
    log = (data / "serve.log").open("wb")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    process = subprocess.Popen(
        [os.environ.get("SMOKE_PYTHON") or sys.executable, str(SERVE), "--port", str(port), "--data-dir", str(data)],
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_live(base, process)
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


@pytest.fixture(scope="module")
def documents(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("documents")
    (folder / "差旅报销办法.md").write_text(
        "# 差旅报销办法\n\n第三条 员工出差报销须在返回后十五个工作日内提交，超期需部门负责人说明原因。\n", encoding="utf-8"
    )
    (folder / "年假规定.md").write_text("# 年假规定\n\n工作满一年可享受五天带薪年假，须提前三个工作日申请。\n", encoding="utf-8")
    return folder


@pytest.fixture(scope="module")
def page(backend):
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=os.environ.get("SMOKE_BROWSER_CHANNEL") or None)
        context = browser.new_context(viewport={"width": 1280, "height": 820}, locale="zh-CN")
        tab = context.new_page()
        tab.set_default_timeout(PAGE_TIMEOUT)
        errors: list[str] = []
        tab.on("pageerror", lambda error: errors.append(str(error)))
        tab.goto(f"{backend}/ui/")
        yield tab
        browser.close()
        assert errors == [], errors


def test_guided_start_from_nothing(page, backend, documents):
    title = page.locator("#guideTitle")

    # Where data goes, plainly, before anything else.
    expect(title).to_have_text("先说清楚数据去向")
    expect(page.locator(".guide-facts")).to_contain_text("文档文字、文件名和你的问题")
    page.get_by_role("button", name="知道了").click()

    # A key: refused, then an empty account, then a good one. Keys go to the fake service only.
    expect(title).to_have_text("连接硅基流动")
    go = page.locator(".guide [data-go]")
    expect(go).to_be_disabled()
    key = page.locator(".guide input[type=password]")
    result = page.locator(".guide-result")
    key.fill("sk-bad-000000000")
    page.get_by_role("button", name="测试").click()
    expect(result).to_contain_text("Key 无效")
    expect(go).to_be_disabled()
    key.fill("sk-empty-00000000")
    page.get_by_role("button", name="测试").click()
    expect(result).to_contain_text("余额不足")
    key.fill("sk-good-000000000")
    page.get_by_role("button", name="测试").click()
    expect(result).to_contain_text("可以用")
    expect(go).to_be_enabled()
    go.click()

    # The recommended model is preselected.
    expect(title).to_have_text("选一个问答模型")
    expect(page.locator(".guide input:checked")).to_have_value("Qwen/Qwen3-30B-A3B-Instruct-2507")
    page.get_by_role("button", name="继续").click()

    # A first folder, by path (a plain browser has no native picker).
    expect(title).to_have_text("给寻寻第一批资料")
    page.locator(".guide-form input").fill(str(documents))
    page.locator(".guide-form button").click()
    expect(result).to_contain_text("开始读了")
    page.get_by_role("button", name="开始用").click()
    expect(page.locator(".guide-scrim")).to_have_count(0)

    settings = _get(backend, "/settings")
    assert settings["onboarding_done"] and settings["data_notice_seen"]
    assert settings["llm_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert settings["api_key"]["configured"]


def test_the_guide_does_not_come_back(page, backend):
    page.goto(f"{backend}/ui/")
    expect(page.locator("#composer")).to_be_visible()
    page.wait_for_timeout(1000)
    expect(page.locator(".guide-scrim")).to_have_count(0)


def _wait_indexed(backend: str, count: int) -> None:
    deadline = time.time() + 60
    while time.time() < deadline:
        status = _get(backend, "/library/status")
        if status["counts"].get("indexed", 0) >= count and not status["busy"]:
            return
        time.sleep(0.3)
    raise TimeoutError(_get(backend, "/library/status")["counts"])


def test_ask_and_get_a_cited_answer(page, backend):
    _wait_indexed(backend, 2)
    page.locator("#input").fill(QUESTION)
    page.locator("#send").click()
    expect(page.locator("#thread")).to_contain_text(ANSWER_FRAGMENT)
    expect(page.locator(".source").first).to_be_visible()
    expect(page.locator(".source").first).to_contain_text("差旅报销办法.md")


def test_open_the_cited_file_in_the_preview(page):
    page.locator(".source").first.click()
    viewer = page.locator("#viewer")
    expect(viewer).to_have_attribute("aria-hidden", "false")
    expect(page.locator("#viewerTitle")).to_have_text("差旅报销办法.md")
    expect(page.locator("#viewerOrig")).to_contain_text("十五个工作日")
    page.locator("#viewerClose").click()
    expect(viewer).to_have_attribute("aria-hidden", "true")


def test_the_conversation_comes_back_after_a_reload(page, backend):
    """It is read from the data folder, not from the page."""
    deadline = time.time() + 15
    while time.time() < deadline and not _get(backend, "/conversations")["conversations"]:
        time.sleep(0.2)
    saved = _get(backend, "/conversations")["conversations"]
    assert len(saved) == 1 and saved[0]["title"] == QUESTION

    page.goto(f"{backend}/ui/")
    expect(page.locator("#thread")).not_to_contain_text(ANSWER_FRAGMENT)
    page.locator("#historyList").get_by_role("button", name=QUESTION).click()
    expect(page.locator("#thread")).to_contain_text(ANSWER_FRAGMENT)
    expect(page.locator(".source").first).to_contain_text("差旅报销办法.md")


def test_add_a_single_file_from_the_library(page, backend, tmp_path):
    single = tmp_path / "会议纪要.md"
    single.write_text("# 会议纪要\n\n下季度预算增加百分之十。\n", encoding="utf-8")
    page.locator('.nav-item[data-view="library"]').click()
    page.locator("#rootPath").fill(str(single))
    page.locator("#rootScan").click()
    expect(page.locator("#docList")).to_contain_text("会议纪要.md")
    _wait_indexed(backend, 3)

    page.locator("#fileSearch").fill("年假")
    expect(page.locator("#docList .tree-file")).to_have_count(1)
    expect(page.locator("#docList")).to_contain_text("年假规定.md")
    page.locator("#fileSearch").fill("")
    expect(page.locator("#docList .tree-file")).to_have_count(3)


def test_a_deck_and_a_workbook_are_read_as_a_whole(page, backend, tmp_path):
    deck = write_deck(tmp_path / "路演材料.pptx", ["市场规模约三百亿", "团队有十二人"])
    book = write_workbook(tmp_path / "报名名单.xlsx", [["姓名", "部门"], ["甲", "人事"], ["乙", "财务"], ["丙", "人事"]])
    for path in (deck, book):
        page.locator("#rootPath").fill(str(path))
        page.locator("#rootScan").click()
        expect(page.locator("#docList")).to_contain_text(path.name)
    _wait_indexed(backend, 5)

    page.locator('.nav-item[data-view="chat"]').click()
    page.locator("#input").fill("总结一下《路演材料》")
    page.locator("#send").click()
    expect(page.locator("#thread")).to_contain_text("已通读全部 2 张幻灯片")
    expect(page.locator(".source").last).to_contain_text("路演材料.pptx")

    page.locator("#input").fill("《报名名单》一共多少条")
    page.locator("#send").click()
    expect(page.locator("#thread")).to_contain_text("共 3 条")


def test_settings_say_where_data_goes_and_offer_a_way_back(page, backend):
    page.locator("#openSettings").click()
    page.locator('label:has(input[value="key"])').click()
    expect(page.locator("#dataNotice")).to_contain_text("发到硅基流动")
    page.locator('label:has(input[value="about"])').click()
    expect(page.locator("#safetyCheck")).to_contain_text("通过")
    page.locator("#backupNow").click()
    expect(page.locator("#backupState")).to_contain_text("1 份")
    backups = _get(backend, "/data-safety")["backups"]
    assert len(backups) == 1 and backups[0]["reason"] == "manual"
    assert sorted(backups[0]["files"]) == ["conversations.sqlite3", "knowledge.sqlite3"]
    expect(page.locator("#exportDiagnostics")).to_contain_text("不含 Key 和文档内容")
    page.locator("#settingsClose").click()
