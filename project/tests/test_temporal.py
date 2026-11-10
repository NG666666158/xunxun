"""Time-scoped answers: no cloud calls, fixed clocks and independent expected dates."""
from datetime import datetime
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from config import Settings
from rag_agent.edges import route_after_analysis
from rag_agent.nodes import AgentNodes
from rag_agent.graph import create_agent_graph
from retrieval.temporal import DocumentDates, request_time
from retrieval.types import SearchHit


def row(number, name, **extra):
    return {"file_id": f"{number:024x}", "name": name, "status": "indexed", **extra}


@pytest.mark.parametrize("now,query,start,end", [
    ("2026-11-09", "这周例会有什么重点", "2026-11-09", "2026-11-16"),
    ("2026-11-08", "这周例会有什么重点", "2026-11-02", "2026-11-09"),
    ("2026-01-01", "本周进展", "2025-12-29", "2026-01-05"),
    ("2026-01-01", "上周进展", "2025-12-22", "2025-12-29"),
    ("2026-03-01", "上个月的检测报告", "2026-02-01", "2026-03-01"),
    ("2026-11-10", "昨天的生产情况", "2026-11-09", "2026-11-10"),
])
def test_calendar_boundaries(now, query, start, end):
    result = request_time(query, datetime.fromisoformat(now))
    assert (result["start"], result["end"]) == (start, end)


@pytest.mark.parametrize("query", ["最新一份报告", "每周报告怎么写", "2023年是谁发明了它", "最近的检测报告", "本周与上周的变化"])
def test_non_calendar_queries_are_not_invented_as_date_ranges(query):
    assert not request_time(query, datetime(2026, 11, 9))


def test_document_date_ignores_directory_and_mtime_and_checks_explicit_fields(tmp_path):
    index = DocumentDates(tmp_path)
    old = row(1, "生产会议纪要_2026年07月13日.docx", path="C:/2026-11-09/a.docx", mtime=1794182400)
    assert index.get(old) == "2026-07-13"
    unknown = row(2, "报告.docx", path="C:/2026-11-09/报告.docx", mtime=1794182400)
    assert index.get(unknown) == ""
    path = tmp_path / f"{unknown['file_id']}.json"
    path.write_text(json.dumps({"units": [{"text": "发布日期：2026年11月09日\n内容提到2020年01月01日"}]}, ensure_ascii=False), encoding="utf-8")
    assert index.get(unknown) == "2026-11-09"  # invalidate the missing-file cache
    path.write_text(json.dumps({"units": [{"text": "会议时间：2026年11月10日"}]}, ensure_ascii=False), encoding="utf-8")
    assert index.get(unknown) == "2026-11-10"
    assert index.get({**unknown, "name": "会议_2026-11-09.docx"}) == ""  # disagreement
    assert index.get(row(3, "对比_2026-11-09_2026-11-10.docx")) == ""
    assert index.get(row(4, "报告_20260230.docx")) == ""


def test_explicit_date_and_large_parsed_unit(tmp_path):
    request = request_time("2026年11月09日的报告有什么重点", datetime(2026, 12, 1))
    assert (request["start"], request["end"]) == ("2026-11-09", "2026-11-10")
    report = row(1, "报告.docx")
    (tmp_path / f"{report['file_id']}.json").write_text(json.dumps({"units": [{"text": "报告日期：2026年11月09日\n" + "正文" * 20000}]}, ensure_ascii=False), encoding="utf-8")
    assert DocumentDates(tmp_path).get(report) == "2026-11-09"


def select(tmp_path, query="这周例会有什么重点", now="2026-11-09", rows=None, scope=None):
    return DocumentDates(tmp_path).select(request_time(query, datetime.fromisoformat(now)), rows or [], scope or [])


def test_monday_fallback_is_explicit_relevant_and_never_arbitrary_old_files(tmp_path):
    previous = row(1, "生产会议纪要_2026年11月06日.docx")
    result = select(tmp_path, rows=[previous, row(2, "销售报告_2026-11-06.docx"), row(3, "会议纪要_2026-07-13.docx")])
    assert result["doc_ids"] == [previous["file_id"]]
    assert result["fallback"]
    assert (result["start"], result["end"]) == ("2026-11-02", "2026-11-09")
    assert "星期一" in result["note"] and "可能" in result["note"] and "周日" in result["note"]
    assert not select(tmp_path, now="2026-11-10", rows=[previous])["doc_ids"]
    assert not select(tmp_path, rows=[row(3, "会议纪要_2026-07-13.docx")])["doc_ids"]
    assert not select(tmp_path, rows=[row(2, "销售报告_2026-11-06.docx")])["doc_ids"]


def test_current_week_wins_and_scope_and_privacy_remain_effective(tmp_path):
    current, previous = row(1, "生产会议纪要_2026-11-09.docx"), row(2, "会议纪要_2026-11-06.docx")
    result = select(tmp_path, rows=[current, previous])
    assert result["doc_ids"] == [current["file_id"]] and not result["fallback"]
    assert not select(tmp_path, rows=[current, previous], scope=["f" * 24])["doc_ids"]
    assert not select(tmp_path, rows=[{**current, "content_mode": "name"}, {**previous, "status": "recycled"}])["doc_ids"]


def test_import_and_modified_dates_only_when_explicit(tmp_path):
    old = row(1, "报告_2026-07-13.docx", created_at="2026-11-09T10:00:00+08:00", mtime=datetime(2026, 11, 9, 12).timestamp())
    assert not select(tmp_path, query="这周报告的重点", rows=[old])["doc_ids"]
    assert select(tmp_path, query="这周导入的报告重点", rows=[old])["doc_ids"] == [old["file_id"]]
    assert select(tmp_path, query="这周修改的报告重点", rows=[old])["doc_ids"] == [old["file_id"]]


class SpyRetriever:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def retrieve(self, query, **options):
        self.calls.append(options)
        return [SimpleNamespace(to_dict=lambda r=r: {"doc_id": r["file_id"], "source": r["name"], "text": "设备巡检", "rerank_score": .9}) for r in self.rows]


def make_nodes(tmp_path, rows):
    retriever = SpyRetriever(rows)
    nodes = AgentNodes(None, retriever, Settings(_env_file=None, data_dir=tmp_path), documents=SimpleNamespace(rows=lambda: rows))
    nodes.clock = lambda: datetime(2026, 11, 9, 12)
    return nodes, retriever


def test_node_retries_keep_dates_and_override_previous_conversation(tmp_path):
    current, old = row(1, "会议纪要_2026-11-09.docx"), row(2, "会议纪要_2026-07-13.docx")
    nodes, retriever = make_nodes(tmp_path, [current, old])
    state = nodes.analyze_query({"question": "这周例会有什么重点", "citations": [{"doc_id": old["file_id"]}],
                                 "messages": [HumanMessage(content="七月重点"), AIMessage(content="七月设备维修"), HumanMessage(content="这周例会有什么重点")]})
    assert state["intent"] == "knowledge"
    for query in [state["question"], "设备重点"]:
        result = nodes.retrieve({**state, "rewritten_query": query})
        assert [d["doc_id"] for d in result["documents"]] == [current["file_id"]]
        assert retriever.calls[-1]["doc_ids"] == {current["file_id"]}
        assert "prefer_doc_ids" not in retriever.calls[-1]
    # No stale time restriction survives a new, ordinary turn.
    assert nodes.analyze_query({"question": "光合作用是什么", "messages": []})["temporal"] == {}


def test_empty_time_scope_does_not_call_retriever_or_model(tmp_path):
    nodes, retriever = make_nodes(tmp_path, [row(1, "会议纪要_2026-07-13.docx")])
    state = nodes.analyze_query({"question": "这周例会有什么重点", "messages": []})
    assert route_after_analysis(state) == "refuse"
    assert not nodes.retrieve(state)["documents"] and not retriever.calls
    answer = nodes.refuse(state)["answer"]
    assert "2026-11-09" in answer and "2026-11-15" in answer


def test_date_check_cannot_be_bypassed_by_fast_or_literal_checks(tmp_path):
    nodes, _ = make_nodes(tmp_path, [row(1, "会议纪要_2026-11-09.docx")])
    state = nodes.analyze_query({"question": "这周例会有什么重点", "messages": []})
    result = nodes.fact_check({**state, "answer": "设备巡检", "context_docs": [{"doc_id": "f" * 24, "text": "设备巡检"}]})
    assert not result["grounded"] and result["check_mode"] == "temporal"


def test_correction_survives_model_omission(tmp_path):
    nodes, _ = make_nodes(tmp_path, [row(1, "会议纪要_2026-11-06.docx")])
    state = nodes.analyze_query({"question": "这周例会有什么重点", "messages": []})
    result = nodes.finish({**state, "answer": "设备巡检", "grounded": True})
    assert result["answer"].startswith("今天是 2026-11-09，星期一")
    assert "2026-11-02 至 2026-11-08" in result["answer"]
    assert result["messages"][0].content == result["answer"]


def test_file_search_uses_business_date_without_reapplying_mtime(tmp_path):
    document = row(1, "报告_2026-11-09.docx", mtime=datetime(2026, 7, 1).timestamp())
    nodes, _ = make_nodes(tmp_path, [document])
    calls = []

    def search(question, filters, **options):
        calls.append(options)
        assert filters.time_range is None
        assert options["scope"] == {document["file_id"]}
        return {"files": [document], "filters": filters.to_dict(), "elapsed_ms": 0,
                "total": 1, "total_candidates": 1, "unreadable": 0}

    nodes.file_search = SimpleNamespace(search=search, clock=nodes.clock, best_name_match=lambda _: 0)
    state = nodes.analyze_query({"question": "找一下本周的报告", "messages": []})
    assert state["intent"] == "find_file"
    result = nodes.search_files(state)
    assert calls and result["grounded"] and "2026-11-09" in result["answer"]


def test_empty_elliptical_followup_does_not_need_a_planner(tmp_path):
    nodes, _ = make_nodes(tmp_path, [])
    state = nodes.analyze_query({"messages": [HumanMessage(content="上周情况"),
                                             AIMessage(content="设备维修"), HumanMessage(content="这周呢")]})
    assert route_after_analysis(state) == "refuse"


def test_business_time_does_not_filter_on_certificate_issue_date(tmp_path):
    rows = [row(1, "上年度纳税证明.pdf"), row(2, "报告_2026-08-11.pdf"),
            row(3, "隐私证明.pdf", content_mode="name")]
    nodes, retriever = make_nodes(tmp_path, rows)
    state = nodes.analyze_query({"question": "星河科技今年交了多少税", "messages": []})
    assert state["temporal"]["basis"] == "content"
    assert state["temporal"]["doc_ids"] == [rows[0]["file_id"], rows[1]["file_id"]]
    result = nodes.retrieve(state)
    assert [d["doc_id"] for d in result["documents"]] == state["temporal"]["doc_ids"]
    assert not state["temporal"]["dates"]  # do not invent a payment date
    scoped = select(tmp_path, query="今年交了多少税", rows=rows, scope=[rows[0]["file_id"]])
    assert scoped["doc_ids"] == [rows[0]["file_id"]]
    assert not scoped["fallback"]


def test_finding_certificates_still_uses_document_time(tmp_path):
    certificate = row(1, "上年度纳税证明.pdf")
    (tmp_path / f"{certificate['file_id']}.json").write_text(json.dumps({"units": [
        {"text": "填发日期 | 2026-08-11\n入库日期 | 2025-07-15\n金额 | 123.45"}
    ]}, ensure_ascii=False), encoding="utf-8")
    result = select(tmp_path, query="找今年的纳税证明", rows=[certificate])
    assert result["basis"] == "document"
    assert result["dates"][certificate["file_id"]] == "2026-08-11"


@pytest.mark.parametrize("query", ["本周会议纪要，本周有什么重点", "这周会议纪要，本周有什么重点",
                                   "本周会议纪要讨论了哪些交付问题", "本周报告中纳税情况的重点"])
def test_time_modifying_document_wins_over_its_business_topic(tmp_path, query):
    rows = [row(1, "会议纪要报告_2026-11-09.docx"), row(2, "会议纪要报告_2026-07-13.docx")]
    result = select(tmp_path, query=query, rows=rows)
    assert result["basis"] == "document" and result["doc_ids"] == [rows[0]["file_id"]]


def test_monday_short_question_ignores_unrelated_current_documents(tmp_path):
    previous = row(1, "会议纪要_2026-11-06.docx")
    result = select(tmp_path, query="本周例会重点", rows=[previous, row(2, "采购报告_2026-11-09.docx")])
    assert result["fallback"] and result["doc_ids"] == [previous["file_id"]]
    assert "星期一" in result["note"] and "2026-11-02 至 2026-11-08" in result["note"]


def test_event_date_outranks_drafting_date_but_conflicting_events_are_rejected(tmp_path):
    r = row(1, "会议纪要_2026-11-09.docx")
    path = tmp_path / f"{r['file_id']}.json"
    path.write_text(json.dumps({"units": [{"text": "会议日期：2026-11-09\n编制日期：2026-11-10"}]}, ensure_ascii=False), encoding="utf-8")
    assert select(tmp_path, rows=[r])["dates"] == {r["file_id"]: "2026-11-09"}
    path.write_text(json.dumps({"units": [{"text": "会议日期：2026-11-06\n编制日期：2026-11-10"}]}, ensure_ascii=False), encoding="utf-8")
    assert not select(tmp_path, rows=[r])["doc_ids"]


def test_year_precision_is_not_presented_as_january_first(tmp_path):
    r = row(1, "2026年度报告.pdf")
    result = select(tmp_path, query="今年的报告", rows=[r])
    assert result["dates"] == {r["file_id"]: "2026"}
    assert not select(tmp_path, query="本周报告", rows=[r])["doc_ids"]
    assert not select(tmp_path, query="今年的报告", rows=[row(2, "2025年与2026年报告.pdf")])["doc_ids"]
    nodes, _ = make_nodes(tmp_path, [r])
    checked = nodes.fact_check({"temporal": result, "answer": "", "context_docs": [{"doc_id": r["file_id"], "text": ""}]})
    assert checked.get("check_mode") != "temporal"  # the year itself is valid evidence metadata


def test_business_time_keeps_fast_search_history_prior_and_explicit_scope(tmp_path):
    rows = [row(1, "纳税证明.pdf"), row(2, "隐私证明.pdf", content_mode="name")]
    nodes, retriever = make_nodes(tmp_path, rows)
    state = {"question": "那它今年交了多少税", "rewritten_query": "星河公司今年交了多少税",
             "recent_doc_ids": [rows[0]["file_id"]],
             "temporal": select(tmp_path, query="今年交了多少税", rows=rows)}
    result = nodes.retrieve(state)
    assert "doc_ids" not in retriever.calls[-1]
    assert retriever.calls[-1]["prefer_doc_ids"] == [rows[0]["file_id"]]
    assert [d["doc_id"] for d in result["documents"]] == [rows[0]["file_id"]]
    nodes.retrieve({**state, "scope_doc_ids": [rows[0]["file_id"]]})
    assert retriever.calls[-1]["doc_ids"] == {rows[0]["file_id"]}


def test_business_date_check_receives_absolute_year_and_rejects_wrong_year(tmp_path):
    document = row(1, "上年度纳税证明.pdf")
    nodes, _ = make_nodes(tmp_path, [document])
    state = nodes.analyze_query({"question": "星河科技今年交了多少税", "messages": []})
    calls = []

    def check(schema, messages, **options):
        calls.append(messages)
        assert "2026-01-01 至 2026-12-31" in messages[0].content
        assert "开具日期不等于业务日期" in messages[0].content
        return SimpleNamespace(grounded=False, unsupported_claims=["金额属于2025年"])

    nodes._invoke_structured = check
    result = nodes.fact_check({**state, "answer": "今年交税123.45元。",
                               "documents": [{"doc_id": document["file_id"]}],
                               "evidence": "填发日期2026-08-11；入库日期2025-07-15；实缴123.45元。"})
    assert calls and not result["grounded"]


@pytest.mark.parametrize("day,expected_calls", [("2026-11-06", 1), ("2026-07-13", 0)])
def test_real_graph_correction_or_empty_result_without_extra_model_calls(tmp_path, day, expected_calls):
    document = row(1, f"会议纪要_{day}.docx")
    calls = []

    class Model:
        def bind(self, **kwargs):
            return self

        def invoke(self, messages):
            calls.append(messages)
            return AIMessage(content=f"设备巡检是重点。[来源：{document['name']}｜位置：section:1]")

    class Retriever:
        def retrieve(self, query, **kwargs):
            assert kwargs["doc_ids"] == {document["file_id"]}
            return [SearchHit("c1", .9, "设备巡检是重点。", document["file_id"], "p1", document["name"], "", "section:1", "hybrid", .9)]

    settings = Settings(_env_file=None, data_dir=tmp_path, answer_mode="fast", term_note=False,
                        anchor_note=False, conflict_finder=False, rejection_note="off", conclusion_line="off")
    access = SimpleNamespace(rows=lambda: [document])
    graph = create_agent_graph(Model(), Retriever(), settings, documents=access,
                               file_search=SimpleNamespace(clock=lambda: datetime(2026, 11, 9, 12), best_name_match=lambda _: 0))
    result = graph.invoke({"messages": [HumanMessage(content="这周例会有什么重点")]},
                          config={"configurable": {"thread_id": day}})
    assert len(calls) == expected_calls
    if expected_calls:
        assert result["grounded"] and "星期一" in result["answer"] and "上周" in result["answer"]
        assert result["context_docs"][0]["date"] == day
    else:
        assert result["refused"] and "2026-11-09" in result["answer"]
