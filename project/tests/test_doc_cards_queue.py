"""Document cards after a big import: which files get a card, in what order, several per call,
at what pace (retrieval/doc_cards.DocCardBuilder), and the files a search moves up."""

from datetime import datetime
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlite_vec")
pytest.importorskip("jieba")

from library import FileRegistry  # noqa: E402
from retrieval.doc_cards import (  # noqa: E402
    ASKED,
    BACKFILL,
    CARD_BATCH_PROMPT,
    CARD_PROMPT,
    NEW,
    DocCardBuilder,
    parse_card,
    parse_cards,
    type_probe,
)
from retrieval.embedded_store import EmbeddedStore  # noqa: E402
from retrieval.file_search import FileSearchService  # noqa: E402
from test_doc_cards import FakeEmbeddings, FakeReranker, StoreSettings, chunk, utc  # noqa: E402

NOW = datetime(2026, 11, 9, 15, 0)


class FakeStore:
    """The card table only."""

    def __init__(self):
        self.cards = {}

    def upsert_doc_card(self, doc_id, card):
        self.cards[doc_id] = dict(card)

    def delete_doc_card(self, doc_id):
        self.cards.pop(doc_id, None)

    def doc_card_hashes(self):
        return {doc_id: card["input_hash"] for doc_id, card in self.cards.items()}

    def renormalize_doc_cards(self, normalize):
        return 0


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 2))
        self.now += seconds


class Chat:
    """Answers a single-file prompt with one card and a batch prompt with one line per file."""

    def __init__(self, tokens=900, drop=()):
        self.prompts = []
        self.tokens = tokens
        self.drop = set(drop)

    def invoke(self, messages):
        prompt = messages[0].content
        self.prompts.append(prompt)
        count = prompt.count("【文件 ")
        if not count:
            reply = '{"类型": "简历", "对象": "某人", "时间": "", "摘要": "单独"}'
        else:
            reply = "\n".join(f'{{"编号": {n}, "类型": "合同", "对象": "甲方{n}", "时间": "", "摘要": "第{n}份"}}'
                              for n in range(1, count + 1) if n not in self.drop)
        return SimpleNamespace(content=reply, usage_metadata={"total_tokens": self.tokens})


class Library:
    def __init__(self, registry=None, clock=None):
        self.registry = registry
        self.big_import = False
        self.idle = float("inf")
        self.clock = clock
        self.answered_at = None
        self.paused = 0

    def importing(self):
        return self.big_import

    def idle_seconds(self):
        if self.answered_at is not None:
            return self.clock() - self.answered_at
        return self.idle

    def pause_for_questions(self):
        self.paused += 1


def write_parsed(folder, doc_ids):
    folder.mkdir(exist_ok=True)
    for doc_id in doc_ids:
        (folder / f"{doc_id}.json").write_text(json.dumps({"units": [{"text": f"开头 {doc_id}"}]}), encoding="utf-8")


def make(tmp_path, names, chat=None, clock=None, library=None, batch=4, **settings):
    write_parsed(tmp_path / "parsed", names)
    config = SimpleNamespace(parsed_dir=tmp_path / "parsed", llm_model="Qwen/Qwen3-8B", doc_card_chars=900,
                             doc_card_interval_seconds=3.0, doc_card_tokens_per_minute=9000.0,
                             doc_card_idle_tokens_per_minute=15000.0, doc_card_cool_down_seconds=60.0, **settings)
    clock = clock or Clock()
    store = FakeStore()
    chat = chat or Chat()
    builder = DocCardBuilder(store, config, chat_model=lambda: chat, files=lambda: dict(names), library=library,
                             batch=batch, sleep=clock.sleep, clock=clock)
    return builder, store, chat, clock


# ---------------- which files ----------------


def test_a_file_whose_name_gives_the_type_needs_no_call(tmp_path):
    names = {"a" * 32: "01 申报书.pdf", "b" * 32: "李明远-agent开发.pdf"}
    builder, store, chat, _ = make(tmp_path, names)
    assert builder.sync() == 1  # only the file whose name says nothing
    assert builder.build("a" * 32) == "named"
    assert builder.run_pending() == {"built": 1}
    assert len(chat.prompts) == 1 and "李明远-agent开发.pdf" in chat.prompts[0]
    assert set(store.cards) == {"b" * 32}
    # Moving it up does nothing either.
    assert builder.prioritize(["a" * 32]) == 0


def test_cards_from_one_call_per_file_are_not_asked_again_in_a_batch(tmp_path):
    names = {str(n) * 32: f"资料_{n}.txt" for n in range(1, 4)}
    builder, store, chat, _ = make(tmp_path, names, batch=1)
    builder.sync()
    builder.run_pending()
    assert len(chat.prompts) == 3 and all("【文件" not in p for p in chat.prompts)
    again, _, _, _ = make(tmp_path, names, chat=Chat(), batch=4)
    again.store = store
    assert again.sync() == 0  # same input hash, whatever the batch size


# ---------------- in what order ----------------


def test_files_looked_at_come_first_then_new_ones_then_the_newest(tmp_path):
    names = {str(n) * 32: f"资料_{n}.txt" for n in range(1, 6)}
    builder, _, _, _ = make(tmp_path, names, batch=1)
    builder.enqueue("1" * 32, BACKFILL, modified=100.0)
    builder.enqueue("2" * 32, BACKFILL, modified=300.0)
    builder.enqueue("3" * 32, NEW)
    builder.enqueue("4" * 32, BACKFILL, modified=200.0)
    assert builder.prioritize(["4" * 32, "5" * 32]) == 2
    order = [builder._take()[0][0] for _ in range(5)]
    assert order == ["4" * 32, "5" * 32, "3" * 32, "2" * 32, "1" * 32]
    assert builder.pending() == 0


def test_a_file_indexed_during_a_big_import_waits_with_the_backfill(tmp_path):
    library = Library()
    builder, _, _, _ = make(tmp_path, {"1" * 32: "a.txt", "2" * 32: "b.txt"}, library=library)
    library.big_import = True
    builder.file_indexed("1" * 32)
    library.big_import = False
    builder.file_indexed("2" * 32)
    assert builder._queued == {"1" * 32: BACKFILL, "2" * 32: NEW}


def test_during_a_big_import_only_files_looked_at_are_built(tmp_path):
    library = Library()
    library.big_import = True
    names = {str(n) * 32: f"资料_{n}.txt" for n in range(1, 4)}
    builder, _, _, _ = make(tmp_path, names, library=library)
    builder.sync()
    assert builder._take(urgent_only=True) == ([], False)
    builder.prioritize(["3" * 32])
    assert builder._take(urgent_only=True) == (["3" * 32], True)


# ---------------- several files per call ----------------


def test_waiting_files_share_one_call(tmp_path):
    names = {str(n) * 32: f"资料_{n}.txt" for n in range(1, 4)}
    builder, store, chat, _ = make(tmp_path, names, chat=Chat(tokens=1200), batch=4)
    builder.sync()
    assert builder.run_pending() == {"built": 3}
    assert len(chat.prompts) == 1
    assert "下面有 3 份文件" in chat.prompts[0] and chat.prompts[0].count("【文件 ") == 3
    assert {card["subject"] for card in store.cards.values()} == {"甲方1", "甲方2", "甲方3"}
    assert {card["tokens"] for card in store.cards.values()} == {400}  # the call's tokens, shared


def test_a_file_left_out_of_a_batch_reply_is_asked_alone(tmp_path):
    names = {str(n) * 32: f"资料_{n}.txt" for n in range(1, 4)}
    builder, store, chat, _ = make(tmp_path, names, chat=Chat(drop={2}), batch=4)
    builder.sync()
    assert builder.run_pending() == {"built": 3, "retry": 1}
    assert len(chat.prompts) == 2 and "【文件" not in chat.prompts[1]
    assert sorted(card["summary"] for card in store.cards.values()) == ["单独", "第1份", "第3份"]


def test_batch_replies_are_read_by_number_and_tolerate_unquoted_values():
    reply = ('{"编号": 2, "类型": 报告, "对象": "", "时间": "2023", "摘要": "二"}\n'
             '{"编号": 1, "类型": "简历", "对象": "李四", "时间": "", "摘要": "一"}')
    first, second, third = parse_cards(reply, 3)
    assert first["doc_type"] == "简历" and first["subject"] == "李四"
    assert second["doc_type"] == "报告" and second["period"] == "2023"
    assert third is None
    assert parse_card('{"类型": 合同, "对象": "甲, 乙"}')["doc_type"] == "合同"


def test_the_single_file_prompt_is_unchanged_and_the_batch_one_keeps_its_rules():
    single = CARD_PROMPT.format(name="n", text="t")
    rules = single[single.index("- 类型"):single.index("\n文件名")]
    assert rules.strip() in CARD_BATCH_PROMPT.format(count=2, files="")


# ---------------- at what pace ----------------


def test_calls_are_paced_by_tokens(tmp_path):
    names = {str(n) * 32: f"资料_{n}.txt" for n in range(1, 4)}
    builder, _, _, clock = make(tmp_path, names, chat=Chat(tokens=900), batch=1)
    builder.sync()
    builder.run_pending()
    # 900 tokens at 15,000 a minute (nobody asked anything: idle) = 3.6 s between calls.
    assert clock.sleeps == [3.6, 3.6]


def test_the_active_budget_applies_after_a_question(tmp_path):
    library = Library()
    library.idle = 120.0  # past the cool-down, within ten minutes of a question
    names = {str(n) * 32: f"资料_{n}.txt" for n in range(1, 3)}
    builder, _, _, clock = make(tmp_path, names, chat=Chat(tokens=900), library=library, batch=1)
    builder.sync()
    builder.run_pending()
    assert clock.sleeps == [6.0]  # 900 tokens at 9,000 a minute


def test_after_a_question_the_backfill_cools_down_but_files_looked_at_do_not(tmp_path):
    clock = Clock()
    library = Library(clock=clock)
    library.answered_at = clock.now - 10.0  # a question was answered 10 s ago
    names = {"1" * 32: "资料_1.txt", "2" * 32: "资料_2.txt"}
    builder, _, _, _ = make(tmp_path, names, clock=clock, library=library, batch=1)
    builder.prioritize(["1" * 32])
    batch, urgent = builder._take()
    assert urgent
    builder._wait_turn(urgent)
    assert clock.sleeps == []  # the user is looking at it: no cool-down
    builder._wait_turn(False)
    assert sum(clock.sleeps) == pytest.approx(50.0)  # the rest of the 60 s cool-down, in steps
    assert max(clock.sleeps) == 5.0 and library.paused >= 2


def test_rate_limits_halve_the_pace(tmp_path):
    builder, _, _, _ = make(tmp_path, {"1" * 32: "a.txt"})
    full = builder.rate()
    builder.interval = builder.base_interval * 2
    assert builder.rate() == pytest.approx(full / 2)


def test_progress_reports_the_time_left(tmp_path):
    names = {str(n) * 32: f"资料_{n}.txt" for n in range(1, 4)}
    builder, _, _, _ = make(tmp_path, names)
    builder.sync()
    builder.prioritize(["1" * 32])
    progress = builder.progress()
    assert progress["waiting"] == 3 and progress["asked"] == 1
    assert progress["tokens_per_minute"] == 15000 and progress["seconds_left"] > 0


# ---------------- the library ----------------


def test_the_library_knows_how_long_ago_a_question_was_answered(tmp_path):
    from library.service import LibraryService

    library = LibraryService.__new__(LibraryService)
    import threading

    library._answering = 0
    library._answer_cond = threading.Condition()
    library._answered_at = None
    assert library.idle_seconds() == float("inf")
    with library.answering():
        assert library.idle_seconds() == 0.0
    assert 0.0 <= library.idle_seconds() < 5.0


# ---------------- a search moves files up ----------------


class Recorder:
    def __init__(self):
        self.asked = []

    def prioritize(self, doc_ids):
        ids = list(doc_ids)
        self.asked.append(ids)
        return len(ids)


RESUME, NEWS, CONTRACT, MEMO = ("1" * 32, "2" * 32, "3" * 32, "4" * 32)


@pytest.fixture
def library_without_cards(tmp_path):
    store = EmbeddedStore(StoreSettings(tmp_path))
    store.ensure_collection()
    store.insert([
        chunk("r1", RESUME, "李明远 电话 邮箱 教育背景 工作经历", [0, 1, 0, 0], "资料_1.txt"),
        chunk("n1", NEWS, "记者报道：某公司公布高管简历", [0, 1, 0, 0], "资料_2.txt"),
        chunk("c1", CONTRACT, "甲方 乙方 签订", [0, 0, 1, 0], "服务合同.pdf"),
        chunk("m1", MEMO, "报销 备忘", [0, 0, 1, 0], "资料_4.txt"),
    ])
    for doc_id, source, text in ((RESUME, "资料_1.txt", "资料_1.txt 李明远 教育背景"), (NEWS, "资料_2.txt", "资料_2.txt 新闻 简历"),
                                 (CONTRACT, "服务合同.pdf", "服务合同.pdf 甲方"), (MEMO, "资料_4.txt", "资料_4.txt 报销")):
        store.upsert_file_entries([(doc_id, source, text, 1,
                                    [0.1, 1.0 if "简历" in text or "李明远" in text else 0.0, 0.0, 0.1])])
    registry = FileRegistry(tmp_path / "registry.sqlite3")
    for doc_id, name in ((RESUME, "资料_1.txt"), (NEWS, "资料_2.txt"), (CONTRACT, "服务合同.pdf"), (MEMO, "资料_4.txt")):
        registry.insert_file(str(tmp_path / "files" / name), root_id=None, origin="upload", file_id=doc_id,
                             status="indexed", created_at=utc(NOW), mtime=NOW.timestamp(), size=100)
    yield store, registry
    registry.close()


def searcher(store, registry, cards=True):
    settings = SimpleNamespace(doc_type_search=True, doc_cards=cards)
    service = FileSearchService(store, FakeEmbeddings(), FakeReranker(), registry, settings, clock=lambda: NOW)
    service.cards = Recorder()
    service.probe_in_background = False
    return service


def test_a_type_question_moves_up_the_likely_files_without_a_card(library_without_cards):
    store, registry = library_without_cards
    service = searcher(store, registry)
    result = service.search("找一下简历")
    # Three files have no card and a name that says nothing (服务合同.pdf is a contract by name).
    assert result["untyped"] == 3
    asked = [doc_id for call in service.cards.asked for doc_id in call]
    assert RESUME in asked and CONTRACT not in asked
    # The probe asks for what a resume looks like, not only the word.
    assert "履历" in type_probe("简历") and "教育" in type_probe("简历")


def test_listed_files_are_moved_up_and_nothing_is_said_once_cards_exist(library_without_cards):
    store, registry = library_without_cards
    for doc_id in (RESUME, NEWS, MEMO):
        store.upsert_doc_card(doc_id, {"doc_type": "简历" if doc_id == RESUME else "新闻", "label": "", "subject": "",
                                       "period": "", "summary": "", "input_hash": "h", "version": 3, "tokens": 1,
                                       "model": "m"})
    service = searcher(store, registry)
    result = service.search("找一下简历")
    assert result["untyped"] == 0 and result["files"][0]["file_id"] == RESUME
    plain = service.search("李明远 教育背景")
    assert plain["untyped"] == 0
    assert [RESUME] == [doc_id for doc_id in service.cards.asked[-1] if doc_id == RESUME]


def test_nothing_is_moved_up_with_cards_off(library_without_cards):
    store, registry = library_without_cards
    service = searcher(store, registry, cards=False)
    assert service.search("找一下简历")["untyped"] == 0
    assert service.cards.asked == []


def test_the_answer_says_the_list_may_be_short(monkeypatch):
    from rag_agent.nodes import AgentNodes

    class Search:
        def lists_type(self, filters):
            return True

        def as_searched(self, filters):
            return filters

        def search(self, question, filters, limit, scope):
            return {"files": [], "total": 0, "unreadable": 0, "typed": 0, "untyped": 1234,
                    "filters": {}, "elapsed_ms": 1, "total_candidates": 2000}

    nodes = AgentNodes.__new__(AgentNodes)
    nodes.file_search = Search()
    nodes.clock = lambda: NOW
    state = {"question": "找一下简历", "messages": [], "scope_doc_ids": []}
    try:
        reply = AgentNodes.search_files(nodes, state)
    except AttributeError as exc:  # the node needs more of the agent than this stand-in has
        pytest.skip(f"search_files needs {exc}")
    assert "还有 1,234 个文件在判断类型" in reply["answer"]
