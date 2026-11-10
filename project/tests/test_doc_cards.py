"""Document cards: type vocabulary, card parsing, the store table, the builder, type ranking."""

from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlite_vec")
pytest.importorskip("jieba")

import openai  # noqa: E402
import httpx  # noqa: E402

from library import FileRegistry  # noqa: E402
from rag_agent.nodes import AgentNodes  # noqa: E402
from retrieval.doc_cards import (  # noqa: E402
    CARD_PROMPT,
    DocCardBuilder,
    detect_doc_type,
    file_card,
    normalize_type,
    parse_card,
    subject_matches,
    type_matches,
)
from retrieval.embedded_store import EmbeddedStore  # noqa: E402
from retrieval.file_filters import merge_planner_filter, parse_query  # noqa: E402
from retrieval.file_search import FileSearchService  # noqa: E402
from retrieval.types import ChunkRecord  # noqa: E402

NOW = datetime(2026, 11, 9, 15, 0)
RESUME, PROPOSAL, RECEIPT, OTHER_RESUME, CONTRACT = ("1" * 32, "2" * 32, "3" * 32, "4" * 32, "5" * 32)


# ---------------- vocabulary ----------------


@pytest.mark.parametrize(
    "label, expected",
    [
        ("简历", "简历"),
        ("个人简历", "简历"),
        ("求职履历", "简历"),
        ("CV", "简历"),
        ("项目申报书", "申报书"),
        ("申报公告", "通知公告"),  # the head of the phrase decides
        ("会议纪要", "会议纪要"),
        ("检测报告", "检测报告"),
        ("年度工作报告", "工作报告"),
        ("报告", "报告"),
        ("合同审查制度", "制度规范"),
        ("证件照片", "照片"),
        ("学术论文", "论文"),
        ("其他：海报", ""),
        ("其它", ""),
        ("", ""),
        ("数据表", ""),
    ],
)
def test_labels_normalise_onto_the_vocabulary(label, expected):
    assert normalize_type(label) == expected


def test_a_parent_type_finds_its_children_but_not_the_reverse():
    assert type_matches("报告", "检测报告")
    assert type_matches("报告", "报告")
    assert not type_matches("检测报告", "工作报告")
    assert not type_matches("检测报告", "报告")
    assert not type_matches("", "简历")


@pytest.mark.parametrize(
    "question, doc_type, rest, mine",
    [
        ("找一下我的简历", "简历", "", True),
        ("找一下简历", "简历", "", False),
        ("帮我找一下我的 CV", "简历", "", True),
        ("张三的简历在哪", "简历", "张三", False),
        ("找一下申报书", "申报书", "", False),  # 书 is not dropped as filler here
        ("2024年的检测报告", "检测报告", "", False),
        ("找一下关于报销的通知", "通知公告", "报销", False),
        ("txt 版本的合同审查制度在哪", "制度规范", "合同审查", False),
        ("我的毕业证书在哪", "证书证件", "毕业", True),
        # Not a type: the type word is not the head of the phrase, or there is none.
        ("简历模板在哪", "", "简历模板", False),
        ("讲多 Agent 协作的书", "", "Agent 协作", False),
        ("上周加的报价单", "", "报价单", False),
        ("一共有多少个 PDF", "", "", False),
    ],
)
def test_questions_name_a_document_type(question, doc_type, rest, mine):
    filters = parse_query(question, NOW)
    assert filters.doc_type == doc_type
    assert filters.type_rest == rest
    assert filters.mine is mine


def test_planner_topic_can_name_the_type_the_rules_missed():
    rules = parse_query("帮我看看那个东西", NOW)
    merged = merge_planner_filter(rules, {"topic": "个人简历"}, NOW)
    assert merged.doc_type == "简历" and merged.type_rest == ""
    assert detect_doc_type(["简历", "模板"]) == ("", "")


def test_card_replies_are_parsed_tolerantly():
    card = parse_card('```json\n{"类型": "个人简历", "对象": "李明远", "时间": "2025", "摘要": "求职 Agent 开发"}\n```')
    assert card == {"label": "个人简历", "doc_type": "简历", "subject": "李明远", "period": "2025",
                    "summary": "求职 Agent 开发"}
    assert parse_card('好的：{"类型": "其它: 海报"}')["label"] == "其他：海报"
    with pytest.raises(ValueError):
        parse_card("这是一份简历")


def test_prompt_lists_the_vocabulary_and_formats():
    text = CARD_PROMPT.format(name="a.pdf", text="开头")
    assert "简历（" in text and "申报书" in text and "文件名：a.pdf" in text


def test_owner_is_compared_without_spaces():
    assert subject_matches("李 明远", "李明远")
    assert not subject_matches("李四", "李明远")
    assert not subject_matches("李明远", "")


# ---------------- store ----------------


class StoreSettings:
    milvus_collection = "unit"
    embedding_dimension = 4
    dense_top_k = 3
    sparse_top_k = 3
    fusion_top_k = 3
    rerank_top_k = 6
    candidate_auto_scale = True
    candidate_pool_max = 80
    rerank_min_keep = 3
    rerank_min_score = 0.05
    rerank_relative_score = 0.1
    rerank_per_document = 0
    rrf_k = 60

    def __init__(self, data_dir):
        self.data_dir = data_dir


def chunk(identifier, doc_id, text, dense, source):
    return ChunkRecord(identifier, doc_id, f"{doc_id}-p0", source, "", "page:1", text, dense)


def card(doc_type, label="", subject="", summary=""):
    return {"doc_type": doc_type, "label": label or doc_type, "subject": subject, "period": "", "summary": summary,
            "input_hash": "h-" + doc_type, "version": 1, "tokens": 600, "model": "m"}


@pytest.fixture
def store(tmp_path):
    store = EmbeddedStore(StoreSettings(tmp_path))
    store.ensure_collection()
    store.insert(
        [
            # The resume never says 简历.
            chunk("r1", RESUME, "李明远 电话 138 邮箱 教育背景 工作经历 Agent 开发", [0, 1, 0, 0], "李明远-agent开发.pdf"),
            # The proposal mentions 简历 (a member's bio): a mention, not a resume.
            chunk("p1", PROPOSAL, "项目组成员简历：李明远，负责 Agent 开发", [0, 1, 0, 0], "01 申报书.pdf"),
            chunk("p2", PROPOSAL, "申报单位 研究内容 经费预算", [0, 0, 1, 0], "01 申报书.pdf"),
            chunk("c1", RECEIPT, "参会回执 姓名 单位 是否住宿", [0, 0, 0, 1], "参会回执.docx"),
            chunk("o1", OTHER_RESUME, "李四 电话 邮箱 教育背景 销售经理", [0, 1, 0, 0], "李四.docx"),
            chunk("k1", CONTRACT, "甲方 乙方 报销 服务费用 违约责任", [0, 0, 1, 0], "服务协议.pdf"),
        ]
    )
    store.upsert_file_entries(
        [
            (RESUME, "李明远-agent开发.pdf", "李明远-agent开发\n个人主页 FreshHouse", 1, [0, 1, 0, 0]),
            (PROPOSAL, "01 申报书.pdf", "01 申报书\n章节：研究内容", 1, [0, 0, 1, 0]),
        ]
    )
    yield store
    store.close()


def test_cards_follow_the_document_lifecycle(store):
    store.upsert_doc_card(RESUME, card("简历", "个人简历", "李明远"))
    store.upsert_doc_card(PROPOSAL, card("申报书"))
    assert set(store.doc_cards(types=["简历"])) == {RESUME}
    assert set(store.doc_cards(doc_ids=[PROPOSAL, RECEIPT])) == {PROPOSAL}
    assert store.doc_cards()[RESUME]["subject"] == "李明远"
    assert store.doc_card_hashes()[RESUME] == "h-简历"

    store.reassign_document(PROPOSAL, "9" * 32)
    assert "9" * 32 in store.doc_cards() and PROPOSAL not in store.doc_cards()

    store.purge_document(RESUME)  # a file switched to name-only loses its card with its text
    assert RESUME not in store.doc_cards()


def test_labels_are_renormalised_without_a_model_call(store):
    store.upsert_doc_card(RESUME, {**card(""), "label": "个人简历"})
    assert store.renormalize_doc_cards(normalize_type) == 1
    assert store.doc_cards()[RESUME]["doc_type"] == "简历"
    assert store.renormalize_doc_cards(normalize_type) == 0


def test_backup_scrub_drops_cards(tmp_path, store):
    from core.data_safety import scrub_database

    store.upsert_doc_card(RESUME, card("简历"))
    path = store.db_path if hasattr(store, "db_path") else tmp_path / "knowledge.sqlite3"
    store.close()
    scrub_database(path, [RESUME], {}, [])
    reopened = EmbeddedStore(StoreSettings(tmp_path))
    try:
        assert RESUME not in reopened.doc_cards()
    finally:
        reopened.close()


# ---------------- builder ----------------


class FakeChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages[0].content)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply, usage_metadata={"total_tokens": 640})


def rate_limit_error():
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"))
    return openai.RateLimitError("TPM limit reached", response=response, body=None)


@pytest.fixture
def parsed(tmp_path):
    folder = tmp_path / "parsed"
    folder.mkdir()
    for doc_id, text in ((RESUME, "李明远 电话 邮箱 教育背景"), (PROPOSAL, "申报单位 研究内容")):
        (folder / f"{doc_id}.json").write_text(json.dumps({"units": [{"text": text}]}, ensure_ascii=False),
                                               encoding="utf-8")
    return folder


def builder(store, parsed, chat, files, sleeps=None, pauses=None):
    settings = SimpleNamespace(parsed_dir=parsed, llm_model="Qwen/Qwen3-8B", doc_card_chars=900)
    return DocCardBuilder(
        store, settings, chat_model=lambda: chat, files=lambda: dict(files),
        pause=(lambda: pauses.append(1)) if pauses is not None else None,
        interval=0.0, sleep=(sleeps.append if sleeps is not None else (lambda s: None)),
    )


def test_builder_writes_each_card_once_and_waits_for_questions(store, parsed):
    chat = FakeChat(['{"类型": "个人简历", "对象": "李明远", "时间": "", "摘要": "求职"}'])
    pauses = []
    cards = builder(store, parsed, chat, {RESUME: "李明远-agent开发.pdf"}, pauses=pauses)
    assert cards.sync() == 1
    assert cards.run_pending() == {"built": 1}
    assert store.doc_cards()[RESUME]["doc_type"] == "简历"
    assert store.doc_cards()[RESUME]["tokens"] == 640
    assert pauses == [1]
    assert "李明远-agent开发.pdf" in chat.calls[0] and "教育背景" in chat.calls[0]
    # Unchanged name and text: nothing is asked again.
    assert cards.sync() == 0
    assert cards.build(RESUME) == "cached" and len(chat.calls) == 1


def test_builder_never_reads_files_kept_by_name_only(store, parsed):
    chat = FakeChat(['{"类型": "申报书"}'])
    store.upsert_doc_card(PROPOSAL, card("申报书"))
    cards = builder(store, parsed, chat, {})  # PROPOSAL switched to name-only: not eligible
    cards.sync()
    assert PROPOSAL not in store.doc_cards()
    assert cards.build(PROPOSAL) == "skipped"
    assert chat.calls == []


def test_builder_slows_down_on_rate_limits(store, parsed):
    chat = FakeChat([rate_limit_error(), '{"类型": "简历"}'])
    sleeps = []
    cards = builder(store, parsed, chat, {RESUME: "r.pdf"}, sleeps=sleeps)
    cards.base_interval = cards.interval = 2.0
    assert cards.build(RESUME) == "built"
    assert cards.stats["rate_limited"] == 1
    assert sleeps and sleeps[-1] >= 25  # waited before asking again
    assert cards.interval > 2.0  # and keeps a longer gap for a while


def test_unreadable_replies_are_kept_as_unknown(store, parsed):
    cards = builder(store, parsed, FakeChat(["我觉得是简历"]), {RESUME: "r.pdf"})
    assert cards.build(RESUME) == "built"
    assert store.doc_cards()[RESUME]["doc_type"] == ""


# ---------------- ranking by type ----------------


class FakeEmbeddings:
    def embed_query(self, text):
        return self.vector(text)

    def embed_documents(self, texts):
        return [self.vector(text) for text in texts]

    @staticmethod
    def vector(text):
        return [0.1, 1.0 if "简历" in text or "李明远" in text else 0.0, 1.0 if "报销" in text else 0.0, 0.1]


class FakeReranker:
    def rerank(self, query, documents, top_n):
        words = [word.lower() for word in query.split()]
        scores = [sum(word in doc.lower() for word in words) / len(words) * 0.9 for doc in documents]
        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [SimpleNamespace(index=i, score=scores[i]) for i in order]


def utc(moment):
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")


@pytest.fixture
def registry(tmp_path, store):
    registry = FileRegistry(tmp_path / "registry.sqlite3")
    rows = [
        (RESUME, "李明远-agent开发.pdf", datetime(2026, 10, 1)),
        (PROPOSAL, "01 申报书.pdf", datetime(2026, 10, 2)),
        (RECEIPT, "参会回执.docx", datetime(2026, 10, 3)),
        (OTHER_RESUME, "李四.docx", datetime(2026, 10, 4)),
        (CONTRACT, "服务协议.pdf", datetime(2026, 10, 5)),
    ]
    for doc_id, name, modified in rows:
        registry.insert_file(str(tmp_path / "files" / name), root_id=None, origin="upload", file_id=doc_id,
                             status="indexed", created_at=utc(modified), mtime=modified.timestamp(), size=100)
    store.upsert_doc_card(RESUME, card("简历", "个人简历", "李明远"))
    store.upsert_doc_card(OTHER_RESUME, card("简历", "简历", "李四"))
    store.upsert_doc_card(PROPOSAL, card("申报书", "项目申报书"))
    store.upsert_doc_card(RECEIPT, card("回执", "会议回执"))
    store.upsert_doc_card(CONTRACT, card("合同", "服务合同"))
    yield registry
    registry.close()


def search(store, registry, question, **switches):
    settings = SimpleNamespace(**{"doc_type_search": True, **switches})
    service = FileSearchService(store, FakeEmbeddings(), FakeReranker(), registry, settings, clock=lambda: NOW)
    return service.search(question)


def test_without_the_switch_nothing_changes(store, registry):
    names = [f["name"] for f in search(store, registry, "找一下我的简历", doc_type_search=False)["files"]]
    assert names[0] == "01 申报书.pdf"  # the old failure: the mention wins, the resume is missed
    assert "李明远-agent开发.pdf" not in names


def test_files_of_the_type_come_first_and_mentions_after(store, registry):
    files = search(store, registry, "找一下我的简历")["files"]
    names = [f["name"] for f in files]
    assert set(names[:2]) == {"李明远-agent开发.pdf", "李四.docx"}
    assert names[2] == "01 申报书.pdf"
    assert all("type" in f["matched_by"] for f in files[:2])
    proposal = files[2]
    assert "mention" in proposal["matched_by"] and proposal["doc_type"] == "申报书"
    assert files[0]["type_label"] in ("个人简历", "简历")
    assert min(f["score"] for f in files[:2]) > proposal["score"]


def test_my_resume_comes_first_with_the_owner_switch(store, registry):
    files = search(store, registry, "找一下我的简历", doc_card_owner=True, owner_name="李明远")["files"]
    assert files[0]["name"] == "李明远-agent开发.pdf" and "owner" in files[0]["matched_by"]
    other = search(store, registry, "找一下我的简历", doc_card_owner=True, owner_name="李四")["files"]
    assert other[0]["name"] == "李四.docx"
    # Without 我的 the owner does not matter.
    plain = search(store, registry, "找一下简历", doc_card_owner=True, owner_name="李四")["files"]
    assert all("owner" not in f["matched_by"] for f in plain)


def test_latest_of_a_type_comes_first_whatever_its_score(store, registry):
    # The older resume (Oct 1) reads better for "简历" than the newer one (Oct 4).
    store.insert([chunk("r2", RESUME, "个人简历 李明远", [0, 1, 0, 0], "李明远-agent开发.pdf")])
    by_score = [f["name"] for f in search(store, registry, "找一下简历")["files"]][:2]
    latest = [f["name"] for f in search(store, registry, "最新的简历")["files"]]
    assert by_score == ["李明远-agent开发.pdf", "李四.docx"]
    assert latest[:2] == ["李四.docx", "李明远-agent开发.pdf"]
    # Both are over a month old (NOW is Nov 9): 最近 alone no longer means the last seven days.
    assert [f["name"] for f in search(store, registry, "最近的简历")["files"]][:2] == latest[:2]
    assert search(store, registry, "最近的简历", doc_type_search=False)["files"] == []
    # With 我的 the user's own still comes first.
    mine = search(store, registry, "我的最新简历", doc_card_owner=True, owner_name="李明远")["files"]
    assert [f["name"] for f in mine][:2] == ["李明远-agent开发.pdf", "李四.docx"]


def test_more_than_the_type_needs_the_rest_of_the_question_too(store, registry):
    files = search(store, registry, "找一下李四的简历")["files"]
    assert files[0]["name"] == "李四.docx" and "type" in files[0]["matched_by"]
    # The contract is about 报销: a 合同 question about 报销 finds it by type and content.
    files = search(store, registry, "关于报销的合同在哪")["files"]
    assert files[0]["name"] == "服务协议.pdf" and "type" in files[0]["matched_by"]


def test_type_matches_respect_metadata_filters(store, registry):
    files = search(store, registry, "找一下 Word 版的简历")["files"]
    assert [f["name"] for f in files][:1] == ["李四.docx"]
    assert "李明远-agent开发.pdf" not in [f["name"] for f in files]


def test_questions_without_a_type_rank_as_before(store, registry):
    on = [f["name"] for f in search(store, registry, "Agent 开发")["files"]]
    off = [f["name"] for f in search(store, registry, "Agent 开发", doc_type_search=False)["files"]]
    assert on == off


def test_chat_answer_says_why_each_file_is_listed(store, registry):
    filters = parse_query("找一下我的简历", NOW)
    files = search(store, registry, "找一下我的简历")["files"]
    message = AgentNodes._describe_files(files, filters, len(files), 0)
    assert "（类型：个人简历）" in message or "（类型：简历）" in message
    assert "提到“简历”" in message


@pytest.mark.parametrize(
    "name, expected",
    [
        ("01 申报书.pdf", "申报书"),
        ("李明远-agent开发.pdf", ""),
        ("简历2024(1).pdf", "简历"),
        ("简历模板.docx", ""),  # a template, not a resume
        ("025_OPS_北方区_信息系统运行维护手册.txt", "说明书"),
        ("CV_v2.pdf", "简历"),
        ("会议纪要（定稿）.docx", "会议纪要"),
        ("某检测报告-最终版.pdf", "检测报告"),
        ("agents.pdf", ""),
    ],
)
def test_a_file_name_can_say_what_the_file_is(name, expected):
    from retrieval.doc_cards import name_doc_type

    assert name_doc_type(name) == expected


def test_a_file_named_after_the_type_counts_without_a_card(store, registry):
    store.delete_doc_card(PROPOSAL)
    files = search(store, registry, "找一下申报书")["files"]
    assert files[0]["name"] == "01 申报书.pdf" and "type" in files[0]["matched_by"]


def test_reindexing_keeps_the_card_removal_drops_it(store):
    store.upsert_doc_card(RESUME, card("简历"))
    store.replace_document(RESUME, [chunk("r9", RESUME, "李明远 新版", [0, 1, 0, 0], "李明远-agent开发.pdf")])
    assert RESUME in store.doc_cards()  # re-read: the builder decides by its input hash
    store.delete_document(RESUME)
    assert RESUME not in store.doc_cards()


def test_a_broken_quote_in_the_summary_keeps_the_type():
    card = parse_card('{"类型": "论文", "对象": "Inconel 718合金", "时间": "", "摘要": "研究γ"相析出"}')
    assert card["doc_type"] == "论文"


def test_a_name_that_says_another_type_wins_over_the_card(tmp_path, store, registry):
    profile = "6" * 32
    store.insert([chunk("f1", profile, "公司成立于2019年，主营纳米硅", [0, 1, 0, 0], "某公司简介2026.pdf")])
    registry.insert_file(str(tmp_path / "files" / "某公司简介2026.pdf"), root_id=None, origin="upload",
                         file_id=profile, status="indexed", size=100)
    store.upsert_doc_card(profile, card("简历", "简历"))  # the model's mistake
    names = [f["name"] for f in search(store, registry, "找一下简历")["files"] if "type" in f["matched_by"]]
    assert "某公司简介2026.pdf" not in names and "李明远-agent开发.pdf" in names


def test_a_mention_beyond_the_shown_snippet_is_still_labelled(store, registry):
    store.replace_document(PROPOSAL, [chunk("p9", PROPOSAL, "申报单位 研究内容 经费预算 " * 20 + "附：项目负责人简历 李明远",
                                            [0, 1, 0, 0], "01 申报书.pdf")])
    files = search(store, registry, "找一下简历")["files"]
    proposal = next(f for f in files if f["name"] == "01 申报书.pdf")
    assert "简历" not in proposal["snippets"][0]["text"]  # cut off in what is shown
    assert "mention" in proposal["matched_by"]


def test_the_library_page_shows_the_type_search_uses():
    # The file's own name wins over the card, as in FileSearchService._typed_files.
    card = {"doc_type": "合同", "label": "合同", "subject": "某公司", "period": "2025年", "summary": "一份通知"}
    shown = file_card("附件（商业计划书）.pdf", card)
    assert shown["type"] == "方案" and shown["by_name"] and shown["card_type"] == "合同" and shown["has_card"]
    assert file_card("张三.pdf", {"doc_type": "简历", "label": "简历"})["by_name"] is False
    # The model's own words are kept when they say more than the type.
    assert file_card("政策.pdf", {"doc_type": "", "label": "政策集合"}) | {} == {
        "type": "其他", "label": "政策集合", "subject": "", "period": "", "summary": "",
        "has_card": True, "by_name": False, "card_type": ""}
    # No card yet: the name alone, or nothing.
    assert file_card("01 申报书.pdf", None)["type"] == "申报书"
    assert file_card("01 申报书.pdf", None)["has_card"] is False
    assert file_card("随手记.txt", None) is None
