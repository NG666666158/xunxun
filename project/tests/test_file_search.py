"""Finding files: filter parsing, scoped store search, file entries, ranking, agent routing."""

from datetime import datetime, timezone
import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlite_vec")
pytest.importorskip("jieba")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from library import FileRegistry  # noqa: E402
from rag_agent.edges import route_after_analysis  # noqa: E402
from rag_agent.nodes import AgentNodes  # noqa: E402
from rag_agent.schemas import QueryPlan  # noqa: E402
from retrieval.doc_cards import DOC_TYPES, type_word_at_end  # noqa: E402
from retrieval.embedded_store import EmbeddedStore  # noqa: E402
from retrieval.file_filters import (  # noqa: E402
    TYPE_WORDS_NOT_A_REQUEST,
    find_file_cue,
    is_inventory,
    merge_planner_filter,
    name_match,
    parse_query,
    resolve_time,
    type_request,
)
from retrieval.file_index import FILE_ENTRY_VERSION, FileIndexer, build_entry_text  # noqa: E402
from retrieval.file_search import FileSearchService  # noqa: E402
from retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from retrieval.types import ChunkRecord  # noqa: E402

NOW = datetime(2026, 10, 24, 15, 0)  # a Saturday


# ---------------- filter parsing ----------------


@pytest.mark.parametrize(
    "phrase, start, end",
    [
        ("今天", "2026-10-24", "2026-10-25"),
        ("昨天", "2026-10-23", "2026-10-24"),
        ("上周", "2026-10-12", "2026-10-19"),
        ("这周", "2026-10-19", "2026-10-26"),
        ("上个月", "2026-09-01", "2026-10-01"),
        ("本月", "2026-10-01", "2026-11-01"),
        ("去年", "2025-01-01", "2026-01-01"),
        ("最近三天", "2026-10-21", "2026-10-25"),
        ("近两周", "2026-10-10", "2026-10-25"),
        ("2024年3月", "2024-03-01", "2024-04-01"),
        ("2024-03-05", "2024-03-05", "2024-03-06"),
        ("12月", "2025-12-01", "2026-01-01"),  # December not reached yet this year
        ("3月5日", "2026-03-05", "2026-03-06"),
        ("两个月前", "2026-08-01", "2026-09-01"),
    ],
)
def test_time_phrases_resolve_to_local_ranges(phrase, start, end):
    resolved = resolve_time(phrase, NOW)
    assert resolved is not None
    assert (resolved[0].date().isoformat(), resolved[1].date().isoformat()) == (start, end)


def test_last_month_rolls_over_the_year():
    start, end = resolve_time("上个月", datetime(2026, 1, 10))
    assert (start, end) == (datetime(2025, 12, 1), datetime(2026, 1, 1))


def test_unknown_time_phrase_is_not_guessed():
    assert resolve_time("春节前", NOW) is None


def test_parse_query_extracts_types_time_field_and_topic():
    filters = parse_query("那份讲 KV Cache 的 PDF 在哪", NOW)
    assert filters.file_types == ["pdf"]
    assert filters.topic == "KV Cache"
    assert filters.time_range is None

    filters = parse_query("上周加的报价单", NOW)
    assert filters.time_field == "added"
    assert filters.time_phrase == "上周"
    assert filters.name_keywords == ["报价单"]

    filters = parse_query("大模型文件夹里上个月放进来的PDF", NOW)
    assert filters.folder == "大模型"
    assert filters.file_types == ["pdf"]
    assert filters.topic == ""

    filters = parse_query("最近三天改过的 Word 文档", NOW)
    assert filters.time_field == "modified"
    assert filters.file_types == ["word"]


@pytest.mark.parametrize(
    "question, cue",
    [
        ("那份讲 KV Cache 的 PDF 在哪", "strong"),
        ("上周加的报价单", "strong"),
        ("有哪些关于 Agent 评估的资料", "strong"),
        ("帮我找一下差旅制度", "strong"),
        ("华东区的报销制度文件", "weak"),
        ("讲多 Agent 协作的书", "strong"),
        ("华东区讲报销的那份 Word 文档", "strong"),
        ("有哪些仓储配送和库存管理的规范", "strong"),
        ("介绍一下 ReAct 的原理", None),
        ("KV Cache 的原理是什么", None),
        ("Qwen3 增加的功能有哪些", None),
        ("RAG 有哪些优点，书里怎么说", None),
        # "在哪" finds a file only as "<a kind of document> 在哪" at the end (suite_v1 fix 1).
        ("会议终端的产品手册在哪", "strong"),
        ("生产会议纪要在哪？", "strong"),
        ("讲 Agent 评估的那本书在哪里呢", "strong"),
        ("报销制度放在哪个文件夹", "strong"),
        ("在哪里能找到差旅制度", "strong"),
        ("福鼎在哪个市", None),
        ("欧盟在哪一天表态支持西非国家经济共同体？", None),
        ("在GeneralPoints和V-IRL的实验中，SFT和RL在哪些方面表现不同？", None),
        ("VLA 控制技术的最新论文发表在哪个会议上？", None),
        ("公司总部在哪", "weak"),  # a place, or a file by its name: the planner decides
        ("面试题200问在哪", "weak"),
        # Content questions the old cues sent to file search.
        ("迪士尼+决定不播放哪个剧集？", None),
        ("办理入职需要提交哪些材料？", None),
        ("办个体营业执照需要什么材料", None),
        ("8岁孩子看什么书", None),
    ],
)
def test_find_file_cues(question, cue):
    assert find_file_cue(question) == cue


# ---------------- documents asked for by their type (doc_cards.DOC_TYPES) ----------------


@pytest.mark.parametrize(
    "question",
    [
        "有哪些合同", "我的简历呢", "把会议纪要列一下", "最近的检测报告",
        "合同有哪些", "简历都有哪些？", "有几份合同", "我们的合同有几份", "所有论文", "列出全部简历", "论文列一下",
        "看看我的简历", "给我看看合同", "打开我的简历", "有没有论文", "有简历吗？", "知识库里有哪些论文？",
        "上个月的发票", "去年的工作报告", "2024年的合同", "最新的检测报告", "上次的会议纪要", "新加的合同",
        "帮我把合同列出来", "我的 CV", "我的申报书",
        # Described, still asked for: "…的合同" or a short compound.
        "有哪些采购合同", "列出所有劳动合同", "我的采购合同", "有哪些和华为签的合同", "有哪些需要续签的合同",
    ],
)
def test_a_document_type_asked_for_goes_to_find_file(question):
    assert find_file_cue(question) == "strong"


@pytest.mark.parametrize(
    "question",
    [
        "合同里有哪些违约条款", "检测报告的结论是什么", "申报书要写哪些内容", "办理入职需要提交哪些材料？",
        "什么是合同", "合同是什么意思", "如何写会议纪要", "需要签合同吗", "公司跟谁签了合同", "简历里写了哪些工作经历",
        "论文的主要贡献是什么", "发票丢了怎么办", "我的合同到期了怎么办", "检测报告有效期多久", "会议纪要里提到了哪些决定",
        "体检报告里的血脂偏高说明什么", "所有员工都要签劳动合同吗", "看不懂合同", "有效合同的条件", "怎么查看我的简历",
        "这些合同有哪些",
        # Type words that are as often what a question is about.
        "有哪些标准", "最近有什么新闻", "双方达成了哪些协议", "劳动合同法有哪些规定",
        # Bare titles stay with the file-name lookup and the planner, as before.
        "合同", "无痛胃镜收费标准", "最新外地车进京规定", "劳资管理员证书", "苹果工程部审核补开的发票",
    ],
)
def test_content_questions_that_name_a_type_stay_knowledge_questions(question):
    assert type_request(question) is None
    assert find_file_cue(question) is None


@pytest.mark.parametrize(
    "question",
    ["简历呢", "合同呢？", "张三的简历呢", "有没有签合同", "员工手册有哪些", "需要提交的检测报告有哪些", "有哪些软件开发合同"],
)
def test_less_plain_type_requests_are_left_to_the_planner(question):
    assert find_file_cue(question) == "weak"


@pytest.mark.parametrize(
    "question, doc_type",
    [("有哪些合同", "合同"), ("我的简历呢", "简历"), ("把会议纪要列一下", "会议纪要"), ("最近的检测报告", "检测报告"),
     ("有几份合同", "合同"), ("所有论文", "论文"), ("看看我的简历", "简历"), ("帮我把合同列出来", "合同"),
     ("最新的检测报告", "检测报告"), ("合同有多少份", "合同"), ("有简历吗？", "简历")],
)
def test_type_requests_parse_to_the_type_alone(question, doc_type):
    filters = parse_query(question, NOW)
    assert (filters.doc_type, filters.type_rest) == (doc_type, "")
    assert filters.topic == filters.doc_type_word


def test_the_type_word_must_end_the_question():
    assert type_word_at_end("有哪些会议纪要") == ("会议纪要", "会议纪要")
    assert type_word_at_end("我的CV") == ("简历", "CV")
    assert type_word_at_end("有哪些简历模板") == ("", "")
    vocabulary = {word for item in DOC_TYPES for word in item.words}
    assert TYPE_WORDS_NOT_A_REQUEST <= vocabulary


def test_planner_fills_what_the_rules_missed_but_rules_win_on_dates():
    rules = parse_query("那份 agent 评估的东西", NOW)
    merged = merge_planner_filter(
        rules, {"time_phrase": "上个月", "file_types": ["docx"], "topic": "Agent 评估", "name_keywords": ["评估"]}, NOW
    )
    assert merged.time_range == (datetime(2026, 9, 1), datetime(2026, 10, 1))
    assert merged.file_types == ["word"]
    assert merged.topic == "Agent 评估"

    rules = parse_query("上周的 PDF", NOW)
    merged = merge_planner_filter(rules, {"time_phrase": "去年"}, NOW)
    assert merged.time_phrase == "上周"


def test_name_match_prefers_the_named_variant():
    keywords = ["华东区", "报销制度"]
    east = name_match(keywords, "007_FIN_华东区_费用报销与预算控制制度.docx")
    head = name_match(keywords, "006_FIN_总部_费用报销与预算控制制度.md")
    assert east > 0.7
    assert head == 0.0
    assert name_match(["KV Cache"], "notes-kv-cache.md") == 1.0


# ---------------- store: scoped search and file entries ----------------


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


DOC_A, DOC_B, DOC_C = "a" * 32, "b" * 32, "c" * 32


def chunk(identifier, doc_id, text, dense, source, locator=""):
    return ChunkRecord(identifier, doc_id, f"{doc_id}-p0", source, "", locator, text, dense)


@pytest.fixture
def store(tmp_path):
    store = EmbeddedStore(StoreSettings(tmp_path))
    store.ensure_collection()
    store.insert(
        [
            chunk("a1", DOC_A, "KV Cache 缓存已经计算的 Key 和 Value", [1, 0, 0, 0], "agents.pdf", "page:57"),
            chunk("a2", DOC_A, "Prompt Cache 与 KV Cache 两个层级的缓存", [0.9, 0.1, 0, 0], "agents.pdf", "page:58"),
            chunk("a3", DOC_A, "Agent 评估要从真实案例生成回归任务", [0, 1, 0, 0], "agents.pdf", "page:221"),
            chunk("b1", DOC_B, "KV Cache 面试题：为什么需要缓存", [0.8, 0.2, 0, 0], "interview.pdf", "page:3"),
            chunk("c1", DOC_C, "差旅报销需要在30天内提交发票", [0, 0, 1, 0], "travel.docx"),
        ]
    )
    yield store
    store.close()


def test_scoped_hybrid_search_only_returns_the_given_documents(store):
    hits = store.search_hybrid("KV Cache", [1, 0, 0, 0], limit=5, doc_ids={DOC_B})
    assert {hit.doc_id for hit in hits} == {DOC_B}
    assert store.search_hybrid("KV Cache", [1, 0, 0, 0], limit=5, doc_ids=set()) == []


def test_scoped_search_falls_back_to_a_filtered_global_knn_for_large_scopes(store):
    store.exact_scan_limit = 0
    hits = store.search_hybrid("KV Cache", [1, 0, 0, 0], limit=5, doc_ids={DOC_A})
    assert hits and {hit.doc_id for hit in hits} == {DOC_A}


def test_unscoped_search_is_unchanged(store):
    hits = store.search_hybrid("KV Cache", [1, 0, 0, 0], limit=3)
    assert hits[0].retrieval_method == "hybrid_rrf"
    assert len({hit.doc_id for hit in hits}) >= 2


def test_file_entries_follow_document_lifecycle(store):
    store.upsert_file_entries(
        [
            (DOC_A, "agents.pdf", "agents\n章节：第7章 Agent 的评估", 1, [0, 1, 0, 0]),
            (DOC_B, "interview.pdf", "interview\n章节：KV Cache 面试题", 1, [1, 0, 0, 0]),
        ]
    )
    entries = store.search_file_entries("Agent 评估", [0, 1, 0, 0], limit=2)
    assert entries[0]["doc_id"] == DOC_A
    assert [e["doc_id"] for e in store.search_file_entries("评估", [0, 1, 0, 0], doc_ids={DOC_B})] == [DOC_B]

    store.rename_source(DOC_A, "agents-v2.pdf")
    assert store.file_entry_versions()[DOC_A] == 0  # rebuilt by the next backfill

    store.reassign_document(DOC_B, "d" * 32)
    assert "d" * 32 in store.file_entry_versions()

    store.delete_document(DOC_A)
    assert DOC_A not in store.file_entry_versions()
    assert all(e["doc_id"] != DOC_A for e in store.search_file_entries("评估", [0, 1, 0, 0]))


# ---------------- file entries from parser snapshots ----------------


def test_entry_text_uses_outline_and_skips_front_matter():
    parsed = {
        "source": "book.pdf",
        "units": [
            {"heading": "版权声明", "text": "Copyright 2024 all rights reserved"},
            {"heading": "目录", "text": "第1章 . . . . . . . . . . . 1"},
            {"heading": "第1章 入门 > 1.1 概念", "text": "Agent 由模型、上下文和工具组成。"},
            {"heading": "第2章 上下文工程", "text": "上下文窗口有限。"},
        ],
    }
    text = build_entry_text("深入理解 Agent.pdf", parsed)
    assert text.splitlines()[0] == "深入理解 Agent"
    assert "第1章 入门 / 第2章 上下文工程" in text
    assert "Copyright" not in text and ". . ." not in text
    assert "Agent 由模型" in text


class FakeEmbeddings:
    """4-d topic vectors: cache / evaluation / reimbursement / other."""

    def embed_query(self, text):
        return self.vector(text)

    def embed_documents(self, texts):
        return [self.vector(text) for text in texts]

    @staticmethod
    def vector(text):
        lowered = text.lower()
        return [
            1.0 if "cache" in lowered or "缓存" in text else 0.0,
            1.0 if "评估" in text else 0.0,
            1.0 if "报销" in text or "报价" in text else 0.0,
            0.1,
        ]


class FakeReranker:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def rerank(self, query, documents, top_n):
        self.calls += 1
        if self.fail:
            raise RuntimeError("reranker down")
        words = [word.lower() for word in query.split()]
        scores = [sum(word in doc.lower() for word in words) / len(words) * 0.9 for doc in documents]
        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [SimpleNamespace(index=i, score=scores[i]) for i in order]


def test_indexer_backfills_missing_and_outdated_entries(tmp_path, store):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    for doc_id, name in ((DOC_A, "agents.pdf"), (DOC_B, "interview.pdf")):
        (parsed_dir / f"{doc_id}.json").write_text(
            json.dumps({"source": name, "units": [{"heading": "第1章 KV Cache", "text": "缓存"}]}), encoding="utf-8"
        )
    indexer = FileIndexer(store, FakeEmbeddings(), SimpleNamespace(parsed_dir=parsed_dir))
    files = {DOC_A: "agents.pdf", DOC_B: "interview.pdf", DOC_C: "travel.docx"}  # no snapshot for C
    assert indexer.backfill(files) == 2
    assert indexer.backfill(files) == 0
    store.rename_source(DOC_A, "agents-v2.pdf")
    assert indexer.backfill({**files, DOC_A: "agents-v2.pdf"}) == 1
    assert store.file_entry_versions() == {DOC_A: FILE_ENTRY_VERSION, DOC_B: FILE_ENTRY_VERSION}


# ---------------- ranking files ----------------


def utc(moment):
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")


@pytest.fixture
def library(tmp_path, store):
    if sys.platform != "win32":
        pytest.skip("the rows below hold Windows paths")
    registry = FileRegistry(tmp_path / "registry.sqlite3")
    rows = [
        (DOC_A, r"C:\书\大模型\agents.pdf", datetime(2026, 9, 10), datetime(2026, 9, 1)),
        (DOC_B, r"C:\书\大模型\interview.pdf", datetime(2026, 10, 14), datetime(2026, 10, 13)),
        (DOC_C, r"C:\工作\travel.docx", datetime(2026, 10, 20), datetime(2025, 3, 1)),
    ]
    for doc_id, path, added, modified in rows:
        registry.insert_file(
            path, root_id=None, origin="upload", file_id=doc_id, status="indexed",
            created_at=utc(added), mtime=modified.timestamp(), size=100,
        )
    store.upsert_file_entries(
        [
            (DOC_A, "agents.pdf", "agents\n章节：第2章 上下文工程 / 第7章 Agent 的评估", 1, [0.3, 1, 0, 0]),
            (DOC_B, "interview.pdf", "interview\n章节：KV Cache 面试题", 1, [1, 0, 0, 0]),
            (DOC_C, "travel.docx", "travel\n差旅报销制度", 1, [0, 0, 1, 0]),
        ]
    )
    yield registry
    registry.close()


def service(store, registry, reranker=None):
    return FileSearchService(store, FakeEmbeddings(), reranker or FakeReranker(), registry,
                             SimpleNamespace(), clock=lambda: NOW)


def test_files_are_ranked_once_with_their_best_passages(store, library):
    result = service(store, library).search("那份讲 KV Cache 的 PDF 在哪")
    names = [item["name"] for item in result["files"]]
    assert names[:2] == ["agents.pdf", "interview.pdf"]  # two KV Cache passages beat one
    assert len(names) == len(set(names))
    top = result["files"][0]
    assert top["snippets"][0]["locator"] in ("page:57", "page:58")
    assert len(top["snippets"]) == 2
    assert top["path"].endswith("agents.pdf") and top["ext"] == ".pdf"
    assert "travel.docx" not in names  # type filter


def test_topic_questions_are_answered_by_file_entries(store, library):
    result = service(store, library).search("有哪些关于 Agent 评估的资料")
    assert result["files"][0]["name"] == "agents.pdf"
    assert "summary" in result["files"][0]["matched_by"]


def test_metadata_only_questions_skip_content_search(store, library):
    reranker = FakeReranker()
    result = service(store, library, reranker).search("上个月放进来的PDF")
    assert [item["name"] for item in result["files"]] == ["agents.pdf"]
    assert reranker.calls == 0

    result = service(store, library).search("上周加的文件")  # added Oct 14
    assert [item["name"] for item in result["files"]] == ["interview.pdf"]

    result = service(store, library).search("上周改过的文件")  # modified Oct 13
    assert [item["name"] for item in result["files"]] == ["interview.pdf"]


def test_folder_filter_and_empty_results(store, library):
    result = service(store, library).search("大模型文件夹里的 PDF")
    assert {item["name"] for item in result["files"]} == {"agents.pdf", "interview.pdf"}
    assert service(store, library).search("去年加的 PDF")["files"] == []


def test_reranker_failure_still_ranks_by_retrieval_order(store, library):
    result = service(store, library, FakeReranker(fail=True)).search("KV Cache 缓存")
    assert result["files"] and result["files"][0]["name"] in ("agents.pdf", "interview.pdf")


def test_scope_limits_the_candidates(store, library):
    result = service(store, library).search("KV Cache", scope={DOC_B})
    assert [item["name"] for item in result["files"]] == ["interview.pdf"]


# ---------------- agent routing ----------------


class PlannerLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


class RecordingRetriever:
    def __init__(self):
        self.kwargs = None

    def retrieve(self, query, mode="hybrid_rerank", **kwargs):
        self.kwargs = kwargs
        return []


class RecordingFileSearch:
    def __init__(self, files=None):
        self.calls = []
        self.files = files if files is not None else [
            {"name": "agents.pdf", "path": r"C:\书\agents.pdf", "snippets": [{"locator": "page:57"}]}
        ]

    def search(self, query, filters, limit=8, scope=None):
        self.calls.append((query, filters, scope, limit))
        return {"files": self.files, "filters": filters.to_dict(), "elapsed_ms": 5, "total": len(self.files),
                "total_candidates": 3, "unreadable": 0}


def nodes(llm=None, retriever=None, file_search=None):
    return AgentNodes(
        llm=llm or PlannerLLM({"intent": "knowledge", "rewritten_query": "q"}),
        retriever=retriever or RecordingRetriever(),
        settings=SimpleNamespace(relevance_threshold=0.2, max_retrieval_attempts=2),
        file_search=file_search,
    )


def test_strong_find_file_questions_skip_the_planner():
    llm = PlannerLLM({"intent": "knowledge", "rewritten_query": "x"})
    update = nodes(llm=llm, file_search=RecordingFileSearch()).analyze_query(
        {"messages": [HumanMessage(content="那份讲 KV Cache 的 PDF 在哪")]}
    )
    assert update["intent"] == "find_file"
    assert llm.calls == 0
    assert route_after_analysis(update) == "search_files"


def test_a_bare_file_title_is_found_without_the_planner(store, library):
    llm = PlannerLLM({"intent": "knowledge", "rewritten_query": "x"})
    node = nodes(llm=llm, file_search=service(store, library))
    update = node.analyze_query({"messages": [HumanMessage(content="interview 资料")]})
    assert update["intent"] == "find_file" and llm.calls == 0
    update = node.analyze_query({"messages": [HumanMessage(content="interview 讲了什么？")]})
    # A first-turn question about content is searched as asked, without the planner.
    assert update["intent"] == "knowledge" and llm.calls == 0
    update = node.analyze_query({"messages": [HumanMessage(content="上一轮"), AIMessage(content="答"),
                                              HumanMessage(content="interview 讲了什么？")]})
    assert llm.calls == 1  # with earlier turns the planner resolves references


def test_type_requests_reach_the_file_list_without_the_planner():
    llm = PlannerLLM({"intent": "knowledge", "rewritten_query": "x"})
    node = nodes(llm=llm, file_search=RecordingFileSearch())
    for question in ("有哪些合同", "我的简历呢", "把会议纪要列一下", "最近的检测报告"):
        update = node.analyze_query({"messages": [HumanMessage(content=question)]})
        assert update["intent"] == "find_file", question
    assert llm.calls == 0
    # The planner cannot send a content question that names a type to file search either.
    llm = PlannerLLM({"intent": "find_file", "rewritten_query": "合同 违约条款"})
    node = nodes(llm=llm, file_search=RecordingFileSearch())
    for question in ("合同里有哪些违约条款", "检测报告的结论是什么", "申报书要写哪些内容", "办理入职需要提交哪些材料？"):
        assert node.analyze_query({"messages": [HumanMessage(content=question)]})["intent"] == "knowledge", question
    # "简历呢" may follow an answer: the planner, which sees the conversation, decides.
    llm = PlannerLLM({"intent": "find_file", "rewritten_query": "简历"})
    update = nodes(llm=llm, file_search=RecordingFileSearch()).analyze_query({"messages": [HumanMessage(content="简历呢")]})
    assert update["intent"] == "find_file" and llm.calls == 1


def test_planner_find_file_needs_some_cue_in_the_question():
    plan = QueryPlan(intent="find_file", rewritten_query="KV Cache 原理")
    assert AgentNodes._enforce_plan_policy("KV Cache 的原理是什么", plan).intent == "knowledge"
    plan = QueryPlan(intent="find_file", rewritten_query="华东区报销制度")
    assert AgentNodes._enforce_plan_policy("华东区的报销制度文件", plan).intent == "find_file"


def test_single_file_scope_is_always_a_content_question():
    llm = PlannerLLM({"intent": "find_file", "rewritten_query": "第三章讲了什么"})
    update = nodes(llm=llm, file_search=RecordingFileSearch()).analyze_query(
        {"messages": [HumanMessage(content="这份文档在哪一章讲 KV Cache")], "scope_doc_ids": [DOC_A]}
    )
    assert update["intent"] == "knowledge"


def test_retrieve_passes_the_scope_to_the_retriever():
    retriever = RecordingRetriever()
    node = nodes(retriever=retriever)
    node.retrieve({"rewritten_query": "KV Cache", "scope_doc_ids": [DOC_A], "answer_profile": "fact"})
    assert retriever.kwargs == {"doc_ids": {DOC_A}, "max_keep": 6}
    node.retrieve({"rewritten_query": "KV Cache", "scope_doc_ids": [], "answer_profile": "compare"})
    assert retriever.kwargs == {"max_keep": 12}


def test_a_file_kept_by_name_only_is_listed_but_not_remembered_for_the_model():
    named = {"file_id": DOC_A, "name": "ZJQQ护照扫描.txt", "path": "C:\\资料\\证件\\ZJQQ护照扫描.txt", "ext": ".txt",
             "status": "named", "snippets": []}
    update = nodes(file_search=RecordingFileSearch(files=[named])).search_files({"question": "找一下护照扫描", "plan": {}})
    assert "ZJQQ护照扫描" in update["answer"]
    remembered = update["messages"][0].content
    assert "ZJQQ" not in remembered and "〔只按文件名收录的文件〕" in remembered


def test_search_files_node_lists_files_without_generation():
    file_search = RecordingFileSearch()
    update = nodes(file_search=file_search).search_files(
        {"question": "上个月加的 KV Cache PDF", "plan": {"file_filter": {"topic": "KV Cache"}}}
    )
    assert update["files"][0]["name"] == "agents.pdf"
    assert "agents.pdf（第 57 页）" in update["answer"]
    assert not update["refused"]
    _, filters, _, _ = file_search.calls[0]
    assert filters.file_types == ["pdf"] and filters.time_field == "added" and filters.topic == "KV Cache"

    empty = nodes(file_search=RecordingFileSearch(files=[])).search_files({"question": "去年的 PDF", "plan": {}})
    assert empty["refused"] and "没有符合条件的 PDF 文件" in empty["answer"]

    empty = nodes(file_search=RecordingFileSearch(files=[])).search_files(
        {"question": "去年讲 KV Cache 的 PDF", "plan": {}}
    )
    assert empty["refused"] and "没有找到符合条件的文件" in empty["answer"] and "类型 PDF" in empty["answer"]


def test_scoped_refusal_mentions_the_scope():
    update = nodes().refuse({"scope_doc_ids": [DOC_A]})
    assert "限定的文件范围" in update["answer"]


def test_a_knowledge_question_with_where_is_searched_not_listed_as_files():
    llm = PlannerLLM({"intent": "knowledge", "rewritten_query": "福鼎 所属 地级市"})
    update = nodes(llm=llm, file_search=RecordingFileSearch()).analyze_query(
        {"messages": [HumanMessage(content="福鼎在哪个市？")]}
    )
    assert update["intent"] == "knowledge"
    # The planner cannot send it to file search either: nothing in it asks for a file.
    llm = PlannerLLM({"intent": "find_file", "rewritten_query": "福鼎"})
    update = nodes(llm=llm, file_search=RecordingFileSearch()).analyze_query(
        {"messages": [HumanMessage(content="福鼎在哪个市")]}
    )
    assert update["intent"] == "knowledge"
    update = nodes(file_search=RecordingFileSearch()).analyze_query(
        {"messages": [HumanMessage(content="会议终端的产品手册在哪")]}
    )
    assert update["intent"] == "find_file"


def test_retriever_scopes_hybrid_search(store):
    class Embeddings:
        def embed_query(self, text):
            return [1, 0, 0, 0]

    retriever = HybridRetriever(store, Embeddings(), FakeReranker(), StoreSettings("."))
    retriever.settings.rerank_top_k = 3
    hits = retriever.retrieve("KV Cache", doc_ids={DOC_B})
    assert hits and {hit.doc_id for hit in hits} == {DOC_B}
    assert retriever.retrieve("KV Cache", doc_ids=set()) == []


# ---------------- API ----------------


class ApiSystem:
    """Stands in for RAGSystem: records what the endpoints pass on."""

    def __init__(self, registry):
        self.library = SimpleNamespace(registry=registry)
        self.search_calls = []
        self.chat_calls = []

    def initialize(self):
        return self

    def start_library(self):
        return self.library

    def search_files(self, query, limit=8, filters=None, scope=None):
        self.search_calls.append((query, limit, filters, scope))
        return {"query": query, "filters": filters.to_dict(), "elapsed_ms": 1, "total_candidates": 0, "files": []}

    def resolve_scope(self, doc_ids=None, folder=""):
        if folder == "empty":
            raise ValueError("该文件夹下没有已入库的文件")
        return sorted(set(doc_ids or []) | ({DOC_A} if folder else set()))

    def new_thread_id(self):
        return "t1"

    def chat(self, question, thread_id, scope=None, received_at=None):
        self.chat_calls.append((question, scope))
        return {
            "thread_id": thread_id, "answer": "ok", "intent": "knowledge", "rewritten_query": question,
            "citations": [], "grounded": True, "refused": False, "retrieval_attempts": 1,
        }


@pytest.fixture
def api(tmp_path, monkeypatch, library):
    from fastapi.testclient import TestClient

    from api import main
    from config import Settings

    system = ApiSystem(library)
    monkeypatch.setattr(main, "system", system)
    monkeypatch.setattr(main, "settings", Settings(_env_file=None, data_dir=tmp_path))
    return TestClient(main.app), system


def test_file_search_endpoint_merges_explicit_parameters(api):
    client, system = api
    response = client.get("/api/v1/files/search", params={"q": "KV Cache", "types": "pdf", "time": "上个月"})
    assert response.status_code == 200
    _, _, filters, scope = system.search_calls[0]
    assert filters.file_types == ["pdf"] and filters.time_phrase == "上个月" and scope is None
    assert client.get("/api/v1/files/search", params={"q": "x", "types": "exe"}).status_code == 400
    assert client.get("/api/v1/files/search", params={"q": "x", "time": "春节前"}).status_code == 400
    assert client.get("/api/v1/files/search", params={"q": " "}).json()["files"] == []


def test_open_file_is_local_only_and_checks_the_original(api, monkeypatch):
    from api import main

    client, _ = api
    assert client.post(f"/api/v1/files/{DOC_A}/open").status_code == 404  # path does not exist here
    assert client.post("/api/v1/files/" + "f" * 32 + "/open").status_code == 404
    monkeypatch.setattr(main, "_local_only", lambda request: (_ for _ in ()).throw(main.HTTPException(403)))
    assert client.post(f"/api/v1/files/{DOC_A}/open").status_code == 403


def test_chat_scope_is_resolved_and_passed_on(api):
    client, system = api
    client.post("/api/v1/chat", json={"question": "讲了什么", "scope": {"doc_ids": [DOC_B]}})
    client.post("/api/v1/chat", json={"question": "讲了什么", "scope": {"folder": r"C:\书"}})
    client.post("/api/v1/chat", json={"question": "讲了什么"})
    assert [scope for _, scope in system.chat_calls] == [[DOC_B], [DOC_A], None]
    response = client.post("/api/v1/chat", json={"question": "q", "scope": {"folder": "empty"}})
    assert response.status_code == 400


def test_quick_search_shortcut_is_saved_and_validated(api):
    client, _ = api
    assert client.get("/api/v1/settings").json()["quick_search_shortcut"] == "Alt+Space"
    assert client.put("/api/v1/settings", json={"quick_search_shortcut": "Ctrl+Shift+K"}).status_code == 200
    assert client.get("/api/v1/settings").json()["quick_search_shortcut"] == "Ctrl+Shift+K"
    assert client.put("/api/v1/settings", json={"quick_search_shortcut": "rm -rf"}).status_code == 422


def test_exact_identifier_decides_between_same_template_files(store, library):
    store.insert([
        chunk("c2", DOC_C, "条款：专用业务标识 LOG-04-ZX11 仅用于西南区入库验收", [0, 0, 0, 1], "travel.docx"),
        chunk("a4", DOC_A, "条款：专用业务标识 LOG-01-ZX11 仅用于总部入库验收", [0, 0, 0, 1], "agents.pdf"),
        chunk("a5", DOC_A, "条款：总部入库验收须在 6 小时内完成", [0, 0, 0, 1], "agents.pdf"),
    ])

    class FlatReranker:
        def rerank(self, query, documents, top_n):
            return [SimpleNamespace(index=i, score=0.9) for i in range(len(documents))][:top_n]

    result = service(store, library, FlatReranker()).search("LOG-04-ZX11 入库验收")
    assert result["files"][0]["name"] == "travel.docx"


# ---------------- how many files, and which ----------------


@pytest.mark.parametrize(
    "question",
    [
        "知识库里一共有多少个 PDF 文件？",
        "我的知识库现在有几个 PDF",
        "知识库有多少 PDF",
        "有多少个 PDF？",
        "PDF 有多少个",
        "PDF 文件一共有几份？",
        "总共多少份pdf",
        "列出所有 PDF",
        "把全部 PDF 列出来",
        "所有 PDF 文件",
        "知识库里都有哪些 PDF？",
    ],
)
def test_counting_a_type_is_nothing_but_that_condition(question):
    filters = parse_query(question, NOW)
    assert filters.file_types == ["pdf"]
    assert filters.topic == "" and filters.name_keywords == []
    assert find_file_cue(question) == "strong"


@pytest.mark.parametrize(
    "question",
    ["一共有多少个文件？", "知识库有多少文件", "有几份文档", "文件总数是多少", "文档的数量", "列出全部文件", "都有哪些文件？"],
)
def test_counting_everything_has_no_condition_at_all(question):
    filters = parse_query(question, NOW)
    assert filters.topic == "" and not filters.has_metadata
    assert find_file_cue(question) == "strong"


@pytest.mark.parametrize(
    "question, types, field, folder",
    [
        ("上个月加了几个文件？", [], "added", ""),
        ("上周新增了多少个 PDF", ["pdf"], "added", ""),
        ("这个月放进来多少文件", [], "added", ""),
        ("最近改过多少文档？", [], "modified", ""),
        ("大模型文件夹里有多少文件？", [], "any", "大模型"),
        ("公司资料文件夹里面有几个 PDF？", ["pdf"], "any", "公司资料"),
        ("公司资料文件夹下有多少 Word 文档", ["word"], "any", "公司资料"),
        ("知识库里有多少张图片", ["image"], "any", ""),
    ],
)
def test_counting_with_time_and_folder_conditions(question, types, field, folder):
    filters = parse_query(question, NOW)
    assert filters.topic == "", filters.topic
    assert filters.file_types == types and filters.time_field == field and filters.folder == folder


@pytest.mark.parametrize(
    "question",
    [
        "PDF 解析有几个步骤？",
        "这份 PDF 有多少页？",
        "这个表格有多少个",
        "表格有多少列？",
        "文档有多少字？",
        "RAG 有多少个组件？",
        "纳米硅产品的价格是多少？",
        "公司有多少个部门？",
        "PDF 是什么？",
    ],
)
def test_counting_something_else_is_a_question_about_content(question):
    assert not is_inventory(question)


@pytest.mark.parametrize(
    "question",
    ["报销需要提交几份材料？", "办理备用金核销要准备多少文件？", "上周开会讨论了几个文件？"],
)
def test_a_count_about_something_else_is_not_a_look_at_the_library(question):
    assert find_file_cue(question) != "strong"


def test_a_month_or_week_said_in_full_is_still_a_time():
    assert parse_query("上一个月加了几个文件？", NOW).time_phrase == "上一个月"
    assert parse_query("上一周加了几个文件？", NOW).time_phrase == "上一周"
    assert find_file_cue("上一个月加了几个文件？") == "strong"
    # Still a reference to something earlier.
    assert find_file_cue("上一个文件有多少页？") != "strong"


def test_a_count_with_a_topic_keeps_the_topic():
    filters = parse_query("有多少个讲 KV Cache 的 PDF？", NOW)
    assert filters.file_types == ["pdf"] and filters.topic == "KV Cache"


def test_the_library_itself_is_not_a_topic_but_can_be_one():
    assert parse_query("知识库里有没有讲 Agent 的文档", NOW).topic == "Agent"
    assert "知识库" in parse_query("有没有讲知识库构建的文档", NOW).topic


def test_images_only_count_as_a_type_when_counting_files():
    assert parse_query("有哪些讲图片压缩的论文", NOW).file_types == []


def many_pdfs(registry, count, folder=r"C:\书\大模型"):
    for n in range(count):
        registry.insert_file(
            rf"{folder}\报告{n:03}.pdf", root_id=None, origin="upload", file_id=f"{n:032x}", status="indexed",
            created_at=utc(datetime(2026, 9, 10)), mtime=datetime(2026, 9, 1).timestamp() + n, size=100,
        )


def test_a_filter_only_question_gets_the_true_total_past_the_limit(store, library):
    many_pdfs(library, 51)
    reranker = FakeReranker()

    result = service(store, library, reranker).search("知识库里一共有多少个 PDF 文件？", limit=8)

    assert result["total"] == 53  # agents.pdf and interview.pdf too
    assert len(result["files"]) == 8
    assert reranker.calls == 0, "counting needs no content search"
    # Newest first: interview.pdf was modified last, then the 51 in turn.
    assert [item["name"] for item in result["files"][:2]] == ["interview.pdf", "报告050.pdf"]


def test_the_total_is_exact_for_each_kind_of_filter(store, library):
    many_pdfs(library, 12)
    search = service(store, library).search
    assert search("有多少个 Word 文档？")["total"] == 1
    assert search("一共有多少个文件？", limit=100)["total"] == 15
    assert search("大模型文件夹里有多少文件？")["total"] == 14
    assert search("工作文件夹里有几个 PDF？")["total"] == 0
    assert search("上周加了几个文件？")["total"] == 1  # interview.pdf, added Oct 14
    assert search("上周改过多少文件？")["total"] == 1  # interview.pdf, modified Oct 13


def test_files_that_could_not_be_read_are_counted_apart(store, library):
    library.insert_file(r"C:\书\大模型\太大.pdf", root_id=None, origin="upload", file_id="f" * 32, status="unsupported",
                        created_at=utc(datetime(2026, 9, 10)), mtime=1.0, size=100)

    result = service(store, library).search("知识库里一共有多少个 PDF 文件？")

    assert result["total"] == 2 and result["unreadable"] == 1
    assert "太大.pdf" not in [item["name"] for item in result["files"]]


def test_a_topic_question_is_not_a_count_of_the_filter(store, library):
    many_pdfs(library, 51)

    result = service(store, library).search("有多少个讲 KV Cache 的 PDF？")

    assert result["total"] < 53
    assert {"agents.pdf", "interview.pdf"} <= {item["name"] for item in result["files"]}


def describe(question, files=3, total=None, unreadable=0):
    rows = [{"name": f"{n}.pdf", "path": rf"C:\{n}.pdf"} for n in range(files)]
    return AgentNodes._describe_files(rows, parse_query(question, NOW), total, unreadable)


def test_the_answer_states_the_total_not_the_rows_listed():
    head = describe("知识库里一共有多少个 PDF 文件？", files=8, total=51, unreadable=1).split("\n")[0]
    assert head == "共 51 个 PDF 文件，另有 1 个没能读取，列出最新的 8 个："
    assert describe("有多少个 Word 文档？", files=3, total=3).split("\n")[0] == "共 3 个 Word 文件："
    assert describe("一共有多少个文件？", files=3, total=3).split("\n")[0] == "共 3 个文件："
    assert "文件夹含“大模型”" in describe("大模型文件夹里有多少文件？", files=3, total=3).split("\n")[0]
    assert describe("有多少张图片", files=2, total=2).split("\n")[0] == "共 2 个图片："
    assert describe("有多少个 PDF", files=0, total=0) == "没有符合条件的 PDF 文件。"


def test_the_written_list_stays_compact_while_the_rows_go_further():
    answer = describe("所有 PDF", files=51, total=51)
    assert answer.split("\n")[0] == "共 51 个 PDF 文件："
    assert len(answer.split("\n")) == 1 + 8


def test_a_topic_answer_still_speaks_of_related_files():
    head = describe("有多少个讲 KV Cache 的 PDF", files=8, total=12).split("\n")[0]
    assert head == "找到 12 个相关文件（条件：类型 PDF），列出最相关的 8 个："


def test_inventory_questions_reach_the_file_list_without_the_planner():
    llm = PlannerLLM({"intent": "knowledge", "rewritten_query": "x"})
    node = nodes(llm=llm, file_search=RecordingFileSearch())
    for question in ("知识库里一共有多少个 PDF 文件？", "列出所有 Excel 文件", "上个月加了几个文件？"):
        update = node.analyze_query({"messages": [HumanMessage(content=question)]})
        assert update["intent"] == "find_file", question
    assert llm.calls == 0


def test_a_question_that_is_nothing_but_conditions_is_a_listing():
    llm = PlannerLLM({"intent": "knowledge", "rewritten_query": "x"})
    node = nodes(llm=llm, file_search=RecordingFileSearch())
    assert node.analyze_query({"messages": [HumanMessage(content="Word 文档")]})["intent"] == "find_file"
    assert node.analyze_query({"messages": [HumanMessage(content="上周的 PDF")]})["intent"] == "find_file"
    assert llm.calls == 0
    assert node.analyze_query({"messages": [HumanMessage(content="PDF 是什么？")]})["intent"] == "knowledge"


def test_counting_in_a_conversation_ignores_the_history():
    llm = PlannerLLM({"intent": "knowledge", "rewritten_query": "x"})
    update = nodes(llm=llm, file_search=RecordingFileSearch()).analyze_query({"messages": [
        HumanMessage(content="纳米硅是什么？"), AIMessage(content="一种材料"), HumanMessage(content="一共有多少个 PDF？"),
    ]})
    assert update["intent"] == "find_file" and llm.calls == 0


def test_the_node_asks_for_a_long_list_only_without_a_topic():
    listing, topical = RecordingFileSearch(), RecordingFileSearch()
    nodes(file_search=listing).search_files({"question": "一共有多少个 PDF？", "plan": {}})
    nodes(file_search=topical).search_files({"question": "讲 KV Cache 的 PDF", "plan": {}})
    assert listing.calls[0][3] == 100
    assert topical.calls[0][3] == 8


def test_file_search_endpoint_reports_the_total(tmp_path, monkeypatch, store, library):
    from fastapi.testclient import TestClient

    from api import main
    from config import Settings

    many_pdfs(library, 51)
    search = service(store, library)
    system = SimpleNamespace(
        initialized=True, initialize=lambda: system, search_files=lambda q, limit, filters, scope: search.search(q, filters, limit=limit, scope=scope),
        start_library=lambda: None,
    )
    monkeypatch.setattr(main, "system", system)
    monkeypatch.setattr(main, "settings", Settings(_env_file=None, data_dir=tmp_path))

    body = TestClient(main.app).get("/api/v1/files/search", params={"q": "一共有多少个 PDF？", "limit": 100}).json()

    assert body["total"] == 53 and len(body["files"]) == 53
    assert TestClient(main.app).get("/api/v1/files/search", params={"q": "PDF", "limit": 200}).status_code == 200
    assert TestClient(main.app).get("/api/v1/files/search", params={"q": "PDF", "limit": 201}).status_code == 422


# ---------------- a document type listed whole, newest first ----------------


@pytest.mark.parametrize(
    "question",
    ["最近的检测报告", "最新的论文", "最近有哪些合同", "我最近的简历", "近期的合同", "最近改过的合同", "最近加的合同",
     "最新加的论文", "最新版的合同"],
)
def test_a_bare_latest_before_a_type_asks_for_an_order(question):
    assert parse_query(question, NOW).newest_first


@pytest.mark.parametrize(
    "question",
    # A window said in full, a described type, no type at all, or no 最近 / 最新.
    ["最近三天改过的合同", "上周加的合同", "最近一周的合同", "最近3个月的检测报告", "最近和华为签的合同", "最新进展的报告",
     "最近的PDF", "有哪些合同", "新加的论文", "2024年的合同"],
)
def test_other_times_and_questions_keep_no_order(question):
    assert not parse_query(question, NOW).newest_first


def test_windows_said_in_full_parse_as_before():
    filters = parse_query("最近三天改过的合同", NOW)
    assert filters.time_range == (datetime(2026, 10, 21), datetime(2026, 10, 25)) and filters.time_field == "modified"
    filters = parse_query("上周加的合同", NOW)
    assert filters.time_range == (datetime(2026, 10, 12), datetime(2026, 10, 19)) and filters.time_field == "added"
    # 最近 alone still parses to the last seven days: the search decides it is an order.
    assert parse_query("最近的检测报告", NOW).time_range == (datetime(2026, 10, 17), datetime(2026, 10, 25))


REPORTS = [  # name, modified, added: modified 3 days ago, a month ago, two months ago
    ("水质检测报告-3.pdf", datetime(2026, 10, 21, 9), datetime(2026, 9, 1)),
    ("水质检测报告-2.pdf", datetime(2026, 9, 24, 9), datetime(2026, 10, 14)),
    ("水质检测报告-1.pdf", datetime(2026, 8, 24, 9), datetime(2026, 10, 22)),
]


@pytest.fixture
def typed_library(tmp_path, store):
    registry = FileRegistry(tmp_path / "typed.sqlite3")
    for n, (name, modified, added) in enumerate(REPORTS):
        registry.insert_file(str(tmp_path / "报告" / name), root_id=None, origin="upload", file_id=f"{0xa0 + n:032x}",
                             status="indexed", created_at=utc(added), mtime=modified.timestamp(), size=100)
    for n in range(12):
        registry.insert_file(str(tmp_path / "合同" / f"合同{n:02}.pdf"), root_id=None, origin="upload",
                             file_id=f"{0xc0 + n:032x}", status="indexed", created_at=utc(datetime(2026, 9, 1)),
                             mtime=datetime(2026, 9, 1).timestamp() + n, size=100)
    registry.insert_file(str(tmp_path / "杂项" / "报价单.xlsx"), root_id=None, origin="upload", file_id="e" * 32,
                         status="indexed", created_at=utc(datetime(2026, 10, 20)),
                         mtime=datetime(2026, 10, 20).timestamp(), size=100)
    # The oldest report reads best for "检测报告": by score it comes first.
    store.insert([chunk("r1", f"{0xa2:032x}", "检测报告 检测报告 水质", [0, 0, 0, 1], "水质检测报告-1.pdf")])
    yield registry
    registry.close()


def typed_service(store, registry, on=True):
    return FileSearchService(store, FakeEmbeddings(), FakeReranker(), registry,
                             SimpleNamespace(doc_type_search=on), clock=lambda: NOW)


def names(result):
    return [item["name"] for item in result["files"]]


def test_latest_reports_are_listed_newest_first_not_filtered_to_a_week(store, typed_library):
    search = typed_service(store, typed_library).search
    result = search("最近的检测报告")
    assert names(result) == ["水质检测报告-3.pdf", "水质检测报告-2.pdf", "水质检测报告-1.pdf"]
    assert result["filters"]["time_range"] is None and result["filters"]["newest_first"]
    assert names(search("最新的检测报告"))[0] == "水质检测报告-3.pdf"
    # Without 最近 / 最新 the best match comes first, as before.
    assert names(search("有哪些检测报告"))[0] == "水质检测报告-1.pdf"
    # "最近加的…": by when each was added.
    assert names(search("最近加的检测报告")) == ["水质检测报告-1.pdf", "水质检测报告-2.pdf", "水质检测报告-3.pdf"]


def test_windows_said_in_full_still_filter(store, typed_library):
    search = typed_service(store, typed_library).search
    assert names(search("最近三天改过的检测报告")) == ["水质检测报告-3.pdf"]
    assert names(search("上周加的检测报告")) == ["水质检测报告-2.pdf"]
    assert set(names(search("最近一周的检测报告"))) == {"水质检测报告-3.pdf", "水质检测报告-1.pdf"}  # modified or added
    # An explicit window (the time wheel's "最近7天") is kept, and the order still applies.
    filters = parse_query("最近的检测报告", NOW)
    filters.time_range, filters.time_phrase = resolve_time("最近7天", NOW), "最近7天"
    assert names(search("最近的检测报告", filters)) == ["水质检测报告-3.pdf", "水质检测报告-1.pdf"]


def test_without_the_type_switch_latest_is_still_the_last_week(store, typed_library):
    result = typed_service(store, typed_library, on=False).search("最近的检测报告")
    assert result["filters"]["time_phrase"] == "最近"
    assert "水质检测报告-2.pdf" not in names(result)  # a month old on both dates


def test_a_type_alone_is_listed_whole_in_chat(store, typed_library):
    file_search = typed_service(store, typed_library)
    update = nodes(file_search=file_search).search_files({"question": "有哪些合同", "plan": {}})
    assert len(update["files"]) == 12 and all("type" in item["matched_by"] for item in update["files"])
    assert update["answer"].split("\n")[0] == "共 12 份合同："
    assert len(update["answer"].split("\n")) == 1 + 8  # the written list stays compact; the rows go on
    for question in ("列出全部合同", "合同有哪些", "我的合同呢"):
        assert len(nodes(file_search=file_search).search_files({"question": question, "plan": {}})["files"]) == 12
    # Without the switch, or with more than the type, the list stays short.
    off = nodes(file_search=typed_service(store, typed_library, on=False)).search_files(
        {"question": "有哪些合同", "plan": {}})
    assert len(off["files"]) <= 8 and off["answer"].startswith("找到")
    assert not file_search.lists_type(parse_query("有哪些和华为签的合同", NOW))


def test_the_latest_reports_answer_says_the_order(store, typed_library):
    def head(question):
        update = nodes(file_search=typed_service(store, typed_library)).search_files({"question": question, "plan": {}})
        return update["answer"].split("\n")[0], update["file_search"]["filters"]

    first, filters = head("最近的检测报告")
    assert first == "共 3 份检测报告，按修改时间从新到旧："
    assert filters["time_phrase"] == ""  # the time wheel starts at 全部
    assert head("最近加的检测报告")[0] == "共 3 份检测报告，按加入时间从新到旧："
    assert head("最近三天改过的检测报告")[0] == "共 1 份检测报告（修改时间 2026-10-21 至 2026-10-24）："


def typed_rows(count, others=0):
    rows = [{"name": f"{n}.pdf", "path": rf"C:\{n}.pdf", "matched_by": ["type"], "doc_type": "合同"}
            for n in range(count)]
    return rows + [{"name": f"x{n}.pdf", "path": rf"C:\x{n}.pdf", "matched_by": ["content", "mention"]}
                   for n in range(others)]


@pytest.mark.parametrize(
    "question, typed, others, head",
    [
        ("有哪些合同", 12, 0, "共 12 份合同："),
        ("有哪些协议", 3, 2, "共 3 份协议，另有 2 个相关文件："),
        ("所有论文", 4, 0, "共 4 篇论文："),
        ("最新的论文", 4, 0, "共 4 篇论文，按修改时间从新到旧："),
        ("最新的论文", 1, 0, "共 1 篇论文："),
        ("上个月的发票", 2, 0, "共 2 张发票（时间 2026-09-01 至 2026-09-30）："),
        ("有哪些 PDF 合同", 2, 0, "共 2 份 PDF 合同："),
        ("我的 CV", 1, 0, "共 1 份简历："),
    ],
)
def test_a_type_list_counts_the_documents_of_the_type(question, typed, others, head):
    rows = typed_rows(typed, others)
    answer = AgentNodes._describe_files(rows, parse_query(question, NOW), len(rows), 0, typed=typed)
    assert answer.split("\n")[0] == head


def test_a_type_list_with_none_of_the_type_speaks_of_related_files():
    rows = typed_rows(0, 3)
    answer = AgentNodes._describe_files(rows, parse_query("有哪些合同", NOW), 3, 0, typed=0)
    assert answer.split("\n")[0] == "找到 3 个相关文件："
