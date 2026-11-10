"""Home-page suggestions: local templates, verification, privacy rules, sampling and retention."""

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from db.parent_store_manager import ParentStoreManager
from library import FileRegistry, LibraryService
from suggestions import SuggestionService, SuggestionStore, privacy
from suggestions.sampler import KIND_CONTINUE, KIND_FIND, KIND_INTEREST, KIND_NEW, Sampler, tokens_for_tests
from suggestions.store import HISTORY_LIMIT
from suggestions.templates import ASK, FIND, candidates, clean_heading, clean_title, title_key

NOW = datetime(2026, 10, 25, 9, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- templates


def test_section_headings_become_questions_without_a_model():
    parsed = {"units": [
        {"heading": "第 1 章 AI Agent 入门 > 1.1 现代 Agent = LLM + 上下文 + 工具 > 1.1.5 ReAct 循环", "text": "…"},
        {"heading": "第 1 章 AI Agent 入门 > 1.2 Harness 工程", "text": "…"},
    ]}
    questions = [c.question for c in candidates("AI-Agents.pdf", parsed)]
    assert "ReAct 循环是如何工作的？" in questions
    assert "什么是Harness 工程？" in questions


def test_file_names_choose_whole_file_questions():
    found = candidates("XX 检测报告.pdf", {"units": [{"heading": "检 验 报 告 > 重 要 声 明", "text": "x"}]})
    questions = [c.question for c in found]
    assert questions[0] == "《XX 检测报告》的检测结论是什么？"
    assert sum(c.qtype == FIND for c in found) == 1
    # Letter-spaced boilerplate headings are not questions.
    assert not any("声明" in q or "声 明" in q for q in questions)


def test_titles_and_headings_lose_numbers_copies_and_noise():
    assert clean_title("03 答辩参会回执.pdf") == "答辩参会回执"
    assert clean_title("纳米硅-传感器领域应用20260101 V1.1(2).pdf") == "纳米硅-传感器领域应用"
    assert clean_title("【配套资料】大模型面试题200问.pdf") == "大模型面试题200问"
    assert title_key("拜访函.docx") == title_key("拜访函.pdf") == title_key("拜访函 (1).docx")
    assert clean_heading("表格 1") == ""
    assert clean_heading("第 3 页") == ""
    assert clean_heading("工作表：Sheet1") == ""
    assert clean_heading("二、行业分析：范式革命、国家战略、万亿蓝海") == "行业分析"


# ---------------------------------------------------------------- service fixtures


class FakeRetriever:
    """First hit = the file whose name shares the most characters with the question."""

    embeddings = None

    def __init__(self, registry):
        self.registry = registry
        self.queries = []

    def retrieve(self, query, top_k=3, query_vector=None, **kwargs):
        self.queries.append(query)
        rows = self.registry.list_files(statuses=("indexed",))
        best = max(rows, key=lambda row: len(set(Path(row["name"]).stem) & set(query)), default=None)
        return [SimpleNamespace(doc_id=best["file_id"])] if best else []


class FakeFileSearch(FakeRetriever):
    def search(self, query, filters=None, limit=8, scope=None):
        return {"files": [{"file_id": hit.doc_id} for hit in self.retrieve(query)]}


class FakeChat:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return SimpleNamespace(content=self.reply)


@pytest.fixture
def env(tmp_path):
    settings = SimpleNamespace(data_dir=tmp_path, parsed_dir=tmp_path / "parsed")
    settings.parsed_dir.mkdir()
    registry = FileRegistry(tmp_path / "knowledge.sqlite3")
    store = SuggestionStore(tmp_path / "knowledge.sqlite3")
    chat = FakeChat("钠基电池循环寿命是多少？\n1. 钠基电池低温表现如何？")
    service = SuggestionService(
        settings, registry, store,
        retriever=FakeRetriever(registry), file_search=FakeFileSearch(registry),
        chat_model=lambda: chat,
    )
    service.chat = chat
    yield SimpleNamespace(settings=settings, registry=registry, store=store, service=service, root=tmp_path)
    store.close()
    registry.close()


def add_file(env, rel_path, headings=("产品概述", "技术参数"), created=None):
    path = env.root / "资料" / rel_path
    record = env.registry.insert_file(str(path), root_id=None, status="indexed", indexed_sha256="v1")
    if created:
        env.registry.update_file(record["file_id"], created_at=created)
    units = [{"heading": heading, "text": f"{Path(rel_path).stem} {heading}"} for heading in headings]
    (env.settings.parsed_dir / f"{record['file_id']}.json").write_text(
        json.dumps({"source": record["name"], "units": units}, ensure_ascii=False), encoding="utf-8"
    )
    return record["file_id"]


def build(env):
    env.service.sync()
    env.service.run_pending()


# ---------------------------------------------------------------- pool


def test_pool_keeps_only_questions_whose_first_hit_is_the_source(env):
    first = add_file(env, "钠基安全电池检测报告.pdf")
    add_file(env, "防蓝光眼镜宣传册.docx")
    build(env)
    rows = env.store.questions(first)
    assert rows and all(row["file_id"] == first for row in rows)
    assert any(row["verified"] for row in rows)
    for row in rows:
        assert row["method"] == "template"
        assert row["qtype"] in (ASK, FIND)
    # A generic heading question ("什么是技术参数？") is answered by either file: rejected.
    generic = [row for row in rows if row["question"] == "《钠基安全电池检测报告》提到的技术参数有哪些？"]
    assert all(row["verified"] for row in generic)
    assert env.store.stats()["files"] == 2


def test_copies_in_another_format_count_as_the_source(env):
    docx = add_file(env, "深圳星河科技拜访函.docx")
    add_file(env, "深圳星河科技拜访函.pdf")
    assert env.service._sources(env.registry.get_file(docx)) == {
        row["file_id"] for row in env.registry.list_files()
    }


def test_ai_questions_are_off_by_default_and_one_call_per_file_when_on(env):
    add_file(env, "钠基电池.pdf")
    build(env)
    assert env.service.chat.calls == []
    privacy.save_prefs(env.settings, ai_questions=True)
    build(env)
    assert len(env.service.chat.calls) == 1
    prompt = env.service.chat.calls[0]
    assert "钠基电池" in prompt and "章节：" in prompt
    ai = [row for row in env.store.questions() if row["method"] == "ai"]
    assert {row["question"] for row in ai} == {"钠基电池循环寿命是多少？", "钠基电池低温表现如何？"}
    build(env)  # nothing changed: no second call
    assert len(env.service.chat.calls) == 1


def test_changed_and_removed_files_update_the_pool(env):
    file_id = add_file(env, "钠基电池.pdf")
    build(env)
    assert env.store.questions(file_id)
    env.registry.update_file(file_id, path=str(env.root / "资料" / "钠基安全电池手册.pdf"))
    build(env)
    assert all("钠基安全电池手册" in row["question"] or row["qtype"] == ASK for row in env.store.questions(file_id))
    assert all(row["file_name"] == "钠基安全电池手册.pdf" for row in env.store.questions(file_id))
    env.registry.soft_delete_file(file_id)
    env.service.sync()
    assert env.store.questions(file_id) == []
    assert env.store.file_states() == {}


def test_library_removal_drops_the_questions_at_once(env, tmp_path):
    settings = SimpleNamespace(
        allowed_extensions=(".pdf",), max_upload_mb=1, max_local_file_mb=1, max_local_text_mb=1, upload_dir=tmp_path / "up", parsed_dir=env.settings.parsed_dir,
        parent_store_dir=tmp_path / "parents", watch_max_files=10,
    )
    settings.parent_store_dir.mkdir()
    store = SimpleNamespace(delete_document=lambda doc_id: None, has_document=lambda doc_id: True)
    library = LibraryService(settings, env.registry, store, ParentStoreManager(settings), indexer=None)
    library.on_dropped = env.service.file_removed
    file_id = add_file(env, "钠基电池.pdf")
    build(env)
    assert env.store.questions(file_id)
    library.remove_file(file_id)
    assert env.store.questions(file_id) == []


# ---------------------------------------------------------------- privacy


def test_sensitive_files_get_no_questions_but_stay_in_the_library(env):
    medical = add_file(env, "个人/2026体检报告.pdf")
    personal_test = add_file(env, "个人检测报告-张三.pdf")
    company = add_file(env, "公司/钠基电池检测报告.pdf")
    build(env)
    assert env.store.questions(medical) == []
    assert env.store.questions(personal_test) == []
    assert env.store.questions(company)
    assert env.registry.get_file(medical)["status"] == "indexed"
    assert env.store.file_states()[medical]["excluded"] == "sensitive:体检"
    assert {card["doc_id"] for card in env.service.suggest(n=4)["cards"]} == {company}


def test_sensitive_words_and_excluded_folders_are_editable(env):
    salary = add_file(env, "人事/2026年度奖金方案.pdf")
    other = add_file(env, "产品/防蓝光眼镜.docx")
    build(env)
    assert env.store.questions(salary)
    privacy.save_prefs(env.settings, sensitive_words=["奖金"])
    # Applies at once to serving, then the sync removes the questions.
    assert all(card["doc_id"] != salary for _ in range(20) for card in env.service.suggest()["cards"])
    env.service.sync()
    assert env.store.questions(salary) == []
    privacy.save_prefs(env.settings, sensitive_words=[], excluded_folders=[str(env.root / "资料" / "产品")])
    build(env)
    assert env.store.questions(other) == []
    assert env.store.questions(salary)


def test_default_word_list_covers_the_agreed_terms():
    for word in ("体检", "病历", "诊断", "身份证", "护照", "工资", "薪资", "合同", "简历", "银行", "密码", "离婚", "律师函"):
        assert word in privacy.DEFAULT_SENSITIVE_WORDS
    assert privacy.sensitive_word("D:/资料/员工合同.docx", "员工合同.docx", privacy.DEFAULT_SENSITIVE_WORDS) == "合同"
    assert privacy.sensitive_word("D:/公司/碳粉检测报告.pdf", "碳粉检测报告.pdf", privacy.DEFAULT_SENSITIVE_WORDS) == ""


def test_questions_are_only_kept_when_history_is_used(env):
    env.service.question_asked("钠基电池的循环寿命？")
    assert env.store.history_count() == 1
    privacy.save_prefs(env.settings, personalize="library")
    env.service.question_asked("防蓝光眼镜的原理？")
    assert env.store.history_count() == 1
    privacy.save_prefs(env.settings, personalize="history", enabled=False)
    env.service.question_asked("防蓝光眼镜的原理？")
    assert env.store.history_count() == 1


def test_history_keeps_the_newest_200_of_the_last_90_days(env):
    env.store.log_question("很久以前的问题", now=NOW - timedelta(days=120))
    for index in range(HISTORY_LIMIT + 30):
        env.store.log_question(f"问题 {index}", now=NOW - timedelta(minutes=HISTORY_LIMIT + 30 - index))
    env.store.prune_history(now=NOW)
    recent = env.store.recent_questions(limit=1000)
    assert len(recent) == HISTORY_LIMIT
    assert recent[0]["question"] == f"问题 {HISTORY_LIMIT + 29}"
    assert "很久以前的问题" not in {row["question"] for row in recent}
    env.store.log_question("90 天前的问题", now=NOW - timedelta(days=91))
    env.store.prune_history(now=NOW)
    assert "90 天前的问题" not in {row["question"] for row in env.store.recent_questions(limit=1000)}
    assert env.store.clear_history() == HISTORY_LIMIT
    assert env.store.history_count() == 0


def test_not_interested_and_blocked_files_stop_suggestions(env):
    first = add_file(env, "钠基电池检测报告.pdf")
    second = add_file(env, "防蓝光眼镜.docx")
    build(env)
    dismissed = env.store.questions(first, verified_only=True)[0]
    env.service.record("dismiss", [dismissed["id"]])
    for _ in range(30):
        assert dismissed["id"] not in {card["id"] for card in env.service.suggest()["cards"]}
    env.service.block_file(second)
    for _ in range(30):
        assert second not in {card["doc_id"] for card in env.service.suggest()["cards"]}
    assert [row["file_id"] for row in env.store.blocked_files()] == [second]
    env.store.unblock_file(second)
    assert any(second in {c["doc_id"] for c in env.service.suggest()["cards"]} for _ in range(30))


def test_empty_library_and_disabled_home_page(env):
    payload = env.service.suggest()
    assert payload["library_empty"] and payload["cards"] == []
    add_file(env, "钠基电池.pdf")
    build(env)
    privacy.save_prefs(env.settings, enabled=False)
    payload = env.service.suggest()
    assert payload["enabled"] is False and payload["cards"] == []


# ---------------------------------------------------------------- sampling


TOPICS = ("钠基电池", "防蓝光眼镜", "前海政策", "铜粉计划", "生产例会", "碳纤维", "光刻工艺", "重点项目")
ASPECTS = ("", "原理", "参数", "应用")


def pool(files=8, per_file=4, days_old=lambda i: i * 3, folders=4):
    rows = []
    for i in range(files):
        topic = TOPICS[i]
        for j in range(per_file):
            rows.append({
                "id": f"q{i}-{j}", "file_id": f"f{i}", "title_key": f"t{i}", "file_name": f"{topic}.pdf",
                "question": f"{topic}{ASPECTS[j]}是什么？" if j else f"找一下{topic}",
                "qtype": FIND if j == 0 else ASK, "method": "template",
                "shown_count": 0, "click_count": 0, "dismissed": 0, "last_shown_at": None,
                "added_at": (NOW - timedelta(days=days_old(i))).isoformat(),
                "path": f"D:/资料/{i % folders}/资料{i}.pdf",
            })
    return rows


def sampler(seed=1):
    return Sampler(tokenizer=tokens_for_tests, rng=random.Random(seed), clock=lambda: NOW)


def test_cold_start_shows_two_new_one_library_one_find():
    cards = sampler().pick(pool(), [], n=4, cold_start=True)
    assert [card.kind for card in cards] == [KIND_NEW, KIND_NEW, KIND_INTEREST, KIND_FIND]
    assert [card.label for card in cards][:2] == ["新加入", "新加入"]
    assert cards[2].label.startswith("来自《") and cards[3].label == "找文件"
    assert cards[3].qtype == FIND


def test_cards_never_share_a_source_file_and_mix_kinds():
    picker = sampler(7)
    rows = pool(files=6) + [  # a copy of file 0 in another format: same title key
        {**row, "id": "copy-" + row["id"], "file_id": "copy0"} for row in pool(files=1)
    ]
    for _ in range(200):
        cards = picker.pick(rows, [], n=4, cold_start=False)
        keys = [next(r["title_key"] for r in rows if r["id"] == card.id) for card in cards]
        assert len(cards) == 4 and len(set(keys)) == 4
        assert sum(card.qtype == FIND for card in cards) == 1


def test_at_most_one_history_card_about_half_the_time():
    picker = sampler(3)
    recent = [{"question": "铜粉计划的产能有多大", "asked_at": (NOW - timedelta(hours=1)).isoformat()}]
    kinds = Counter()
    for _ in range(400):
        cards = picker.pick(pool(), recent, n=4, cold_start=False)
        history_cards = [card for card in cards if card.kind == KIND_CONTINUE]
        assert len(history_cards) <= 1
        kinds[bool(history_cards)] += 1
        for card in history_cards:
            assert card.label == "接着上次的话题" and "铜粉" in card.question
    assert 0.35 < kinds[True] / 400 < 0.65


def test_questions_already_asked_are_not_suggested_again():
    rows = pool(files=2)
    asked = rows[1]["question"]
    recent = [{"question": asked, "asked_at": NOW.isoformat()}]
    for seed in range(50):
        assert asked not in {card.question for card in sampler(seed).pick(rows, recent, n=4)}


def test_feedback_and_novelty_shift_the_draw():
    rows = pool(files=6, days_old=lambda i: 30)
    for row in rows:
        if row["file_id"] == "f0":
            row.update(click_count=8, shown_count=10)
        elif row["file_id"] == "f1":
            row.update(shown_count=40, last_shown_at=NOW.isoformat())
    picker = sampler(11)
    seen = Counter(card.file_id for _ in range(300) for card in picker.pick(rows, [], n=4, cold_start=False))
    assert seen["f0"] > seen["f1"] * 2


def test_exclude_gives_a_different_batch():
    rows = pool()
    picker = sampler(5)
    first = picker.pick(rows, [], n=4, cold_start=False)
    second = picker.pick(rows, [], n=4, exclude=[card.id for card in first], cold_start=False)
    assert not {card.id for card in first} & {card.id for card in second}


def test_labels_are_neutral():
    cards = [card for seed in range(30) for card in sampler(seed).pick(pool(), [{"question": "防蓝光眼镜怎么选", "asked_at": NOW.isoformat()}])]
    for card in cards:
        assert "注意到" not in card.label and "浏览" not in card.label and "你" not in card.label


# ---------------------------------------------------------------- API


def test_settings_and_history_work_before_the_system_is_initialized(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from api import main
    from config import Settings
    from core.rag_system import RAGSystem

    settings = Settings(_env_file=None, data_dir=tmp_path, siliconflow_api_key="")
    system = RAGSystem(settings)
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "system", system)
    client = TestClient(main.app)

    sg = client.get("/api/v1/settings").json()["suggestions"]
    assert sg["enabled"] is True and sg["personalize"] == "history" and sg["ai_questions"] is False
    assert "体检" in sg["sensitive_words"] and sg["history_limit"] == {"count": 200, "days": 90}
    assert client.get("/api/v1/suggestions").json() == {
        "enabled": True, "ready": False, "file_count": 0, "library_empty": False, "cards": [],
    }
    body = client.put("/api/v1/settings", json={
        "suggestions_personalize": "library", "suggestions_sensitive_words": ["工资", " 工资 ", "合同&甲方"],
        "suggestions_excluded_folders": ["D:\资料\个人"],
    }).json()["suggestions"]
    assert body["personalize"] == "library" and body["sensitive_words"] == ["工资", "合同&甲方"]
    assert client.put("/api/v1/settings", json={"suggestions_personalize": "everything"}).status_code == 422
    system.suggestion_store.log_question("钠基电池的循环寿命？")
    cleared = client.delete("/api/v1/suggestions/history").json()
    assert cleared["removed"] == 1 and cleared["history_count"] == 0
    assert client.put("/api/v1/settings", json={"suggestions_enabled": False}).status_code == 200
    assert client.get("/api/v1/suggestions").json()["enabled"] is False


def test_suggestion_endpoints_serve_cards_and_take_feedback(env, monkeypatch):
    from fastapi.testclient import TestClient

    from api import main

    class System:
        initialized = True
        suggestions = env.service

        def initialize(self):
            return self

        def start_library(self):
            return None

    first = add_file(env, "钠基电池检测报告.pdf")
    add_file(env, "防蓝光眼镜.docx")
    build(env)
    env.settings.app_token = ""
    monkeypatch.setattr(main, "settings", env.settings)
    monkeypatch.setattr(main, "system", System())
    client = TestClient(main.app)

    data = client.get("/api/v1/suggestions?n=4").json()
    assert data["ready"] and data["file_count"] == 2 and len(data["cards"]) == 2
    card = data["cards"][0]
    assert set(card) >= {"id", "question", "kind", "label", "doc_id", "file_name"}
    assert client.post("/api/v1/suggestions/events", json={"action": "shown", "ids": [card["id"]]}).json() == {"updated": 1}
    assert client.post("/api/v1/suggestions/events", json={"action": "dismiss", "ids": [card["id"]]}).json() == {"updated": 1}
    assert env.store.get(card["id"])["dismissed"] == 1
    assert client.post("/api/v1/suggestions/events", json={"action": "opened", "ids": [card["id"]]}).status_code == 422
    assert client.post("/api/v1/suggestions/blocked", json={"doc_id": first}).json()["doc_id"] == first
    assert all(c["doc_id"] != first for c in client.get("/api/v1/suggestions").json()["cards"])
    restored = client.delete(f"/api/v1/suggestions/blocked/{first}").json()
    assert restored["blocked_files"] == []
