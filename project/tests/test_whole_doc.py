"""Whole-file questions: which file, notes with a cache, saying what was covered, and the graph."""

import re
import threading

from langchain_core.messages import AIMessage, HumanMessage
import httpx
import openai
from openpyxl import Workbook
import pytest

from rag_agent.document_nodes import DocumentAccess
from core.conversations import ConversationCheckpointer
from rag_agent.graph import create_agent_graph
from rag_agent.whole_doc import (
    NoteStore,
    ReadResult,
    compress_notes,
    describe_coverage,
    distinct_files,
    estimate_reading,
    files_named_in,
    match_files,
    merge_locators,
    narrow_by_folder,
    parse_notes,
    pick_choice,
    plan_batches,
    read_passages,
    wants_whole_file,
    Note,
)
from suggestions.followups import FollowupService
from types import SimpleNamespace

DOC = "a" * 32


def row(name, file_id=None, path=None, sha="", mtime=1.0):
    return {"file_id": file_id or (name.encode().hex() + "0" * 32)[:32], "name": name, "path": path or f"C:\\库\\{name}", "sha256": sha or file_id or name, "mtime": mtime}


# ---------------------------------------------------------------- the wording


@pytest.mark.parametrize("question", [
    "总结一下《01 申报书》这份文件的主要内容",
    "《精密级制造技术白皮书》主要讲了什么？",
    "概括一下《X》这份文件",
    "总结一下《马斯克》这篇文章",
    "《图解大模型》这本书主要讲了什么？",
    "总结一下这份文件的主要内容",
    "这份文件主要讲了什么？",
    "总结一下这份 PPT 的主要内容",
    "帮我梳理一下这个文档的要点",
    "总结《X》",
    "这份资料的大意是什么",
])
def test_asks_about_the_whole_file(question):
    assert wants_whole_file(question)


@pytest.mark.parametrize("question", [
    "总结一下 Agent 的记忆机制有哪些类型",
    "KV Cache 的原理是什么？",
    "《01 申报书》里的项目总投资是多少？",
    "第3章的主要内容讲了 Transformer 的什么？",
    "《X》第3章讲了什么",
    "总结一下《X》的风险部分",
    "《X》的预算是多少",
    "文件里提到了哪些风险",
    "",
])
def test_asks_about_a_part_or_something_else(question):
    assert not wants_whole_file(question)


def test_a_file_name_written_out_counts_as_naming_it():
    assert wants_whole_file("总结一下铜精密级粉体计划书", ["铜精密级粉体计划书"])
    assert not wants_whole_file("总结一下铜精密级粉体计划书")


def test_match_files_prefers_exact_then_without_number_then_partial():
    rows = [row("01 申报书.pdf"), row("0.政策补贴汇总.xlsx"), row("政策补贴汇总表说明.docx"), row("铜精密级粉体生产项目商业计划书.pdf")]
    assert [r["name"] for r in match_files("01 申报书", rows)] == ["01 申报书.pdf"]
    assert [r["name"] for r in match_files("01 申报书.pdf", rows)] == ["01 申报书.pdf"]
    assert [r["name"] for r in match_files("0.政策补贴汇总", rows)] == ["0.政策补贴汇总.xlsx"]
    assert [r["name"] for r in match_files("政策补贴汇总", rows)] == ["0.政策补贴汇总.xlsx"]
    assert [r["name"] for r in match_files("生产项目商业计划书", rows)] == ["铜精密级粉体生产项目商业计划书.pdf"]
    assert match_files("书", rows) == []  # too short to mean anything


def test_files_named_in_a_question():
    rows = [row("0.政策补贴汇总.xlsx"), row("01 申报书.pdf"), row("申报书模板.docx")]
    found, names = files_named_in("《0.政策补贴汇总》有多少条", rows)
    assert [r["name"] for r in found] == ["0.政策补贴汇总.xlsx"] and names == ["0.政策补贴汇总"]
    found, names = files_named_in("政策补贴汇总里一共多少条", rows)
    assert [r["name"] for r in found] == ["0.政策补贴汇总.xlsx"] and names == ["政策补贴汇总"]
    assert files_named_in("大模型面试题一共有多少道", rows) == ([], [])


def test_copies_with_the_same_content_count_once():
    rows = [row("a.pdf", "1" * 32, sha="same", mtime=1), row("a.pdf", "2" * 32, sha="same", mtime=9), row("a.pdf", "3" * 32, sha="other")]
    kept = distinct_files(rows)
    assert {r["file_id"] for r in kept} == {"2" * 32, "3" * 32}


def test_same_named_files_are_told_apart_by_their_folder():
    a = row("01 申报书.pdf", "1" * 32, "C:\\资料\\01 面向精密级对准的工艺研究\\01 申报书.pdf")
    b = row("01 申报书.pdf", "2" * 32, "C:\\资料\\02 精密级合金粉末宏量开发\\01 申报书.pdf")
    assert narrow_by_folder("总结《01 申报书》（面向精密级对准）", [a, b]) == [a]
    assert narrow_by_folder("总结《01 申报书》", [a, b]) == []
    assert pick_choice("第二份", [a, b]) == b
    assert pick_choice("1", [a, b]) == a
    assert pick_choice("第 2 份", [a, b]) == b
    assert pick_choice("合金粉末那份", [a, b]) == b
    assert pick_choice("第五份", [a, b]) is None
    assert pick_choice("算了，换个问题", [a, b]) is None


# ---------------------------------------------------------------- reading


def parents_of(*texts, prefix="page"):
    return [
        {"parent_id": f"{DOC}-p{i}", "locator": f"{prefix}:{i + 1}", "heading": f"第 {i + 1} 节", "text": text}
        for i, text in enumerate(texts)
    ]


class Reader:
    """A model that writes one note per passage and records what it was asked."""

    def __init__(self, fail_on=None, error=None):
        self.batches = []
        self.fail_on = fail_on
        self.error = error
        self.lock = threading.Lock()

    def __call__(self, messages, max_tokens=None):
        body = str(messages[-1].content)
        numbers = [int(n) for n in re.findall(r"\[P(\d+)｜", body)]
        with self.lock:
            self.batches.append(numbers)
        if self.fail_on and any(text in body for text in self.fail_on):
            raise self.error or httpx.ConnectError("断线")
        return AIMessage(
            content="\n".join(f"P{n}: 第{n}段的笔记" for n in numbers),
            usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
        )


def status_error(code):
    request = httpx.Request("POST", "http://x")
    return openai.APIStatusError("x", response=httpx.Response(code, request=request), body=None)


def test_batches_stay_in_order_and_within_budget():
    parents = parents_of("字" * 100, "字" * 100, "字" * 100, "字" * 100)
    assert plan_batches([0, 1, 2, 3], parents, 250) == [[0, 1], [2, 3]]
    # A passage already read splits the batch: what is sent stays consecutive.
    assert plan_batches([0, 2, 3], parents, 1000) == [[0], [2, 3]]
    # A passage over the budget still goes, alone.
    assert plan_batches([0, 1], parents_of("字" * 900, "字"), 100) == [[0], [1]]


def test_every_passage_gets_read_and_noted(tmp_path):
    reader = Reader()
    result = read_passages(DOC, parents_of("一", "二", "三"), reader, NoteStore(tmp_path), concurrency=2)
    assert [note.text for note in result.notes] == ["第1段的笔记", "第2段的笔记", "第3段的笔记"]
    assert [note.locator for note in result.notes] == ["page:1", "page:2", "page:3"]
    assert (result.read, result.unread, result.cached) == (3, [], 0)
    assert result.input_tokens == 100 * result.calls


def test_second_read_costs_nothing(tmp_path):
    parents = parents_of("一", "二", "三")
    store = NoteStore(tmp_path)
    read_passages(DOC, parents, Reader(), store)
    again = Reader()
    result = read_passages(DOC, parents, again, store)
    assert again.batches == []
    assert (result.calls, result.cached, result.read) == (0, 3, 3)
    assert len(result.notes) == 3


def test_after_an_edit_only_the_changed_passage_is_read(tmp_path):
    store = NoteStore(tmp_path)
    read_passages(DOC, parents_of("一", "二", "三"), Reader(), store)
    again = Reader()
    result = read_passages(DOC, parents_of("一", "改过的二", "三"), again, store)
    assert again.batches == [[2]]
    assert (result.cached, result.read) == (2, 3)
    assert "改过的二" not in str(store.load(DOC)["notes"])
    assert len(store.load(DOC)["notes"]) == 3  # the old passage's note is gone


def test_a_passage_with_nothing_to_note_is_read_but_not_evidence(tmp_path):
    def model(messages, max_tokens=None):
        return AIMessage(content="P1: 有内容的一段\nP2: -")

    result = read_passages(DOC, parents_of("一", "29"), model, NoteStore(tmp_path))
    assert [note.index for note in result.notes] == [0]
    assert (result.read, result.unread) == (2, [])


def test_parse_notes_tolerates_how_models_write_them():
    assert parse_notes("[P3]: 甲\nP4｜page:4: 乙\n  p5：丙\n别的话\nP99: 不在这批", [2, 3, 4]) == {2: "甲", 3: "乙", 4: "丙"}


def test_a_failed_batch_is_reported_and_kept_for_next_time(tmp_path):
    parents = parents_of("好的一段", "坏的一段", "又一段好的")
    store = NoteStore(tmp_path)
    reader = Reader(fail_on=["坏的一段"])
    result = read_passages(DOC, parents, reader, store, batch_tokens=3, concurrency=1)
    assert result.unread == [1]
    assert result.read == 2
    assert "连接中断" in result.error
    assert [note.index for note in result.notes] == [0, 2]
    # It was tried again before giving up.
    assert reader.batches.count([2]) == 2
    # The next question continues from what was read.
    later = Reader()
    resumed = read_passages(DOC, parents, later, store, batch_tokens=3)
    assert later.batches == [[2]]
    assert resumed.unread == [] and len(resumed.notes) == 3


def test_a_reply_without_notes_does_not_count_as_read(tmp_path):
    result = read_passages(DOC, parents_of("一"), lambda m, max_tokens=None: AIMessage(content="好的。"), NoteStore(tmp_path))
    assert result.unread == [0] and result.notes == []


def test_no_balance_stops_reading_at_once(tmp_path):
    parents = parents_of(*[f"第{i}段" for i in range(12)])
    reader = Reader(fail_on=["第"], error=status_error(402))
    result = read_passages(DOC, parents, reader, NoteStore(tmp_path), batch_tokens=3, concurrency=1)
    assert result.fatal and "余额不足" in result.error
    assert len(reader.batches) < 12
    assert len(result.unread) == 12


def test_progress_reaches_the_total(tmp_path):
    seen = []
    read_passages(DOC, parents_of("一", "二"), Reader(), NoteStore(tmp_path), batch_tokens=1, progress=lambda d, t: seen.append((d, t)))
    assert seen[0] == (0, 2) and seen[-1] == (2, 2)


def test_store_refuses_odd_document_ids(tmp_path):
    store = NoteStore(tmp_path)
    assert store.load("../../secret") == {"notes": {}, "compressed": {}}
    with pytest.raises(ValueError):
        store.save("../../secret", {})
    store.remove("../../secret")


# ---------------------------------------------------------------- long files


@pytest.mark.parametrize("locators,expected", [
    (["page:3", "page:4-5"], "page:3-5"),
    (["page:7"], "page:7"),
    (["slide:1", "slide:3"], "slide:1-3"),
    (["sheet:A;rows:2-31", "sheet:A;rows:32-61"], "sheet:A;rows:2-61"),
    (["第一节", "第二节"], "第一节"),
    (["page:1", "slide:2"], "page:1"),
])
def test_merging_locators(locators, expected):
    assert merge_locators(locators) == expected


def test_compress_merges_neighbours_keeps_locators_and_caches(tmp_path):
    notes = [Note(i, f"page:{i + 1}", "", f"{i}" + "字" * 100, f"q{i}") for i in range(40)]
    calls = []

    def model(messages, max_tokens=None):
        count = len(re.findall(r"\[N\d+\]", str(messages[-1].content)))
        calls.append(count)
        return AIMessage(content=f"N1-N{count}: 合并的内容", usage_metadata={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10})

    store = NoteStore(tmp_path)
    result = ReadResult([])
    merged = compress_notes(DOC, notes, model, store, result, limit_tokens=2000, concurrency=1)
    assert sum(calls) == 40 and len(merged) == len(calls)
    assert merged[0].locator.startswith("page:1-") and merged[0].quote == "q0"
    assert merged[-1].locator.endswith("-40")
    assert result.calls == len(calls)
    # Asking again reads the cache.
    calls.clear()
    assert len(compress_notes(DOC, notes, model, store, ReadResult([]), limit_tokens=2000, concurrency=1)) == len(merged)
    assert calls == []


def test_short_notes_are_not_compressed(tmp_path):
    notes = [Note(0, "page:1", "", "短", "")]
    assert compress_notes(DOC, notes, None, NoteStore(tmp_path), ReadResult([])) == notes


def test_a_group_that_will_not_compress_keeps_its_notes(tmp_path):
    notes = [Note(i, f"page:{i + 1}", "", "字" * 100, "") for i in range(30)]
    merged = compress_notes(DOC, notes, lambda m, max_tokens=None: AIMessage(content="不知道"), NoteStore(tmp_path), ReadResult([]), limit_tokens=2000)
    assert merged == notes


# ---------------------------------------------------------------- what was covered


def test_coverage_of_a_full_read_says_pages_and_those_without_text():
    parents = parents_of("一", "二", "三")
    parents[2]["locator"] = "page:5"
    text = describe_coverage(parents, ReadResult([], read=3), {"pages": 5})["text"]
    assert text == "已通读全文：共 5 页，其中第 3–4 页没有可读文字。"


def test_coverage_of_a_partial_read_names_what_is_missing_and_why():
    parents = parents_of("一", "二", "三", "四")
    result = ReadResult([], read=2, unread=[1, 2], error="与模型服务的连接中断了，请检查网络后重试。")
    coverage = describe_coverage(parents, result, {"pages": 4})
    assert coverage["text"] == "这次没有读完：第 2–3 页没读到（与模型服务的连接中断了，请检查网络后重试）。再问一次会接着读。"
    assert coverage["unread"] == 2


def test_coverage_of_a_deck_says_pictures_were_not_read():
    parents = parents_of("一", "二", prefix="slide")
    assert describe_coverage(parents, ReadResult([], read=2), {"slides": 2, "pictures": 7})["text"] == "已通读全部 2 张幻灯片（7 张图片里的文字没有读取）。"
    assert describe_coverage(parents, ReadResult([], read=2), {"slides": 2})["text"] == "已通读全部 2 张幻灯片。"


def test_coverage_of_a_workbook_and_of_a_plain_document():
    parents = [{"locator": "sheet:甲;rows:2-31", "text": "x"}, {"locator": "sheet:乙 ;rows:2-9", "text": "y"}]
    assert describe_coverage(parents, ReadResult([], read=2), {})["text"] == "已通读全部 2 个工作表。"
    plain = [{"locator": "", "heading": "节", "text": "x"}]
    assert describe_coverage(plain, ReadResult([], read=1), {})["text"] == "已通读全文（1 段）。"


def test_estimate_counts_only_what_is_still_unread(tmp_path):
    parents = parents_of("字" * 4000, "文" * 4000)
    store = NoteStore(tmp_path)
    first = estimate_reading(parents, store, DOC, "Qwen/Qwen3-30B-A3B-Instruct-2507")
    assert first["tokens"] == 8000 and first["yuan"] > 0
    read_passages(DOC, parents[:1], Reader(), store)
    assert estimate_reading(parents, store, DOC, "unknown-model")["tokens"] == 4000
    assert estimate_reading(parents, store, DOC, "unknown-model")["yuan"] is None


# ---------------------------------------------------------------- through the graph


class Settings:
    relevance_threshold = 0.2
    max_retrieval_attempts = 2
    context_token_budget = 500
    answer_mode = "strict"
    llm_model = "m"
    summary_model = ""
    summary_concurrency = 2
    summary_batch_tokens = 4000
    summary_reduce_tokens = 12000
    summary_confirm_tokens = 150000


class Model:
    """Notes, a summary citing what it was given, table queries and the fact check."""

    def __init__(self, grounded=True, query='{"op":"count"}'):
        self.grounded = grounded
        self.query = query
        self.prompts = []
        self.lock = threading.Lock()

    def invoke(self, messages):
        system = str(messages[0].content)
        with self.lock:
            self.prompts.append(system[:12])
        if "为后面的全文总结做笔记" in system:
            numbers = re.findall(r"\[P(\d+)｜", str(messages[-1].content))
            return AIMessage(content="\n".join(f"P{n}: 第{n}段讲的内容" for n in numbers))
        if "翻译成一个查询" in system:
            return AIMessage(content=self.query)
        if "事实一致性检查器" in system:
            value = "true" if self.grounded else "false"
            claims = "[]" if self.grounded else '["金额不对"]'
            return AIMessage(content=f'{{"grounded":{value},"unsupported_claims":{claims}}}')
        if "基于用户知识库文档回答问题" in system:
            evidence = re.findall(r"来源：([^｜]+)｜位置：([^\]]+)\]", str(messages[-1].content))
            cited = "".join(f"第{i}点。[来源：{source}｜位置：{where}]\n" for i, (source, where) in enumerate(evidence[:2]))
            return AIMessage(content=f"这份文件讲了几件事。\n{cited}")
        if "查询规划器" in system:
            return AIMessage(content='{"intent":"knowledge","rewritten_query":"改写"}')
        raise AssertionError(system)


class Retriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, mode, doc_ids=None, max_keep=None):
        self.calls.append(doc_ids)
        return []


def access(tmp_path, rows, parents, original=None):
    return DocumentAccess(
        rows=lambda: rows,
        parents=lambda doc_id: parents.get(doc_id, []),
        metadata=lambda doc_id: {"pages": len(parents.get(doc_id, []))},
        original=lambda doc_id: original.get(doc_id) if original else None,
        notes=NoteStore(tmp_path / "summaries"),
    )


def build(tmp_path, rows, parents, model=None, original=None, retriever=None):
    return create_agent_graph(
        llm=model or Model(), retriever=retriever or Retriever(), settings=Settings(),
        documents=access(tmp_path, rows, parents, original),
    )


def ask(graph, question, thread="t", scope=None):
    state = {"messages": [HumanMessage(content=question)], "question": question, "scope_doc_ids": scope or []}
    return graph.invoke(state, config={"configurable": {"thread_id": thread}})


def test_summary_reads_every_passage_cites_them_and_says_what_was_covered(tmp_path):
    report = row("报告.pdf", DOC)
    graph = build(tmp_path, [report], {DOC: parents_of("一", "二", "三")})
    result = ask(graph, "总结一下《报告》的主要内容")
    assert result["intent"] == "summarize_file"
    assert result["grounded"] is True and result["refused"] is False
    assert result["answer"].endswith("已通读全文：共 3 页。")
    assert result["coverage"]["read"] == 3 and result["coverage"]["unread"] == 0
    assert [c["locator"] for c in result["citations"]] == ["page:1", "page:2"]  # only what the answer cites
    assert all(c["doc_id"] == DOC for c in result["citations"])
    # What the reader highlights is the passage's own text, not the note.
    assert result["citations"][0]["quote"] == "一"
    # Written from the notes, checked against the passages.
    assert "第1段讲的内容" in result["evidence"]
    assert result["context_docs"][0]["text"] == "一"
    assert result["coverage"]["checked_against"] == "passages"
    assert result["messages"][-1].content == result["answer"]


def _ask_durably(graph, question, thread="t"):
    state = {"messages": [HumanMessage(content=question)], "question": question, "scope_doc_ids": []}
    return graph.invoke(state, config={"configurable": {"thread_id": thread}}, durability="exit")


def test_a_question_asked_back_is_still_waiting_after_a_restart(tmp_path):
    """The reply to "哪一份？" reaches a new process: the state lives in conversations.sqlite3."""
    a = row("01 申报书.pdf", "1" * 32, "C:\\资料\\甲甲甲甲\\01 申报书.pdf")
    b = row("01 申报书.pdf", "2" * 32, "C:\\资料\\乙乙乙乙\\01 申报书.pdf")
    parents = {"1" * 32: parents_of("甲"), "2" * 32: parents_of("乙", "丙")}
    path = tmp_path / "conversations.sqlite3"

    def start():
        checkpointer = ConversationCheckpointer(path)
        graph = create_agent_graph(
            llm=Model(), retriever=Retriever(), settings=Settings(), checkpointer=checkpointer,
            documents=access(tmp_path, [a, b], parents),
        )
        return graph, checkpointer

    graph, checkpointer = start()
    assert _ask_durably(graph, "总结一下《01 申报书》")["choices"] == ["第 1 份", "第 2 份"]
    checkpointer.close()

    graph, checkpointer = start()
    second = _ask_durably(graph, "第 2 份")
    assert second["intent"] == "summarize_file" and second["target_file"]["file_id"] == "2" * 32
    assert second["coverage"]["read"] == 2 and not second["pending_file"]
    checkpointer.close()

    # What the finished summary left behind can be read back too, for its follow-ups.
    graph, checkpointer = start()
    values = graph.get_state({"configurable": {"thread_id": "t"}}).values
    assert values["coverage"]["read"] == 2 and values["answer"].endswith("已通读全文：共 2 页。")
    checkpointer.close()


def test_a_table_answer_goes_through_the_saved_state(tmp_path, policy):
    path = tmp_path / "conversations.sqlite3"
    checkpointer = ConversationCheckpointer(path)
    graph = create_agent_graph(
        llm=Model(), retriever=Retriever(), settings=Settings(), checkpointer=checkpointer,
        documents=access(tmp_path, [row(policy.name, TABLE, str(policy))], {}, original={TABLE: policy}),
    )
    assert "共 13 条" in _ask_durably(graph, "《0.政策补贴汇总》一共多少条")["answer"]
    checkpointer.close()

    checkpointer = ConversationCheckpointer(path)
    graph = create_agent_graph(llm=Model(), retriever=Retriever(), settings=Settings(), checkpointer=checkpointer)
    values = graph.get_state({"configurable": {"thread_id": "t"}}).values
    assert values["intent"] == "table_query" and "共 13 条" in values["answer"]
    checkpointer.close()


def test_a_very_long_file_is_checked_against_its_merged_notes(tmp_path):
    class Long(Model):
        def invoke(self, messages):
            if "合并压缩" in str(messages[0].content):
                count = len(re.findall(r"\[N\d+\]", str(messages[-1].content)))
                return AIMessage(content=f"N1-N{count}: 合并的内容")
            return super().invoke(messages)

    settings = Settings()
    settings.summary_reduce_tokens = 20
    parents = {DOC: parents_of(*[f"第{i}段很长的原文" for i in range(30)])}
    graph = create_agent_graph(
        llm=Long(), retriever=Retriever(), settings=settings,
        documents=access(tmp_path, [row("报告.pdf", DOC)], parents),
    )
    result = ask(graph, "总结一下《报告》")
    assert result["coverage"]["checked_against"] == "notes"
    assert result["context_docs"][0]["text"] == "合并的内容"
    assert result["context_docs"][0]["locator"].startswith("page:1-")


def test_a_retired_notes_model_falls_back_to_the_answer_model(tmp_path):
    class Retired:
        def invoke(self, messages):
            request = httpx.Request("POST", "http://x")
            raise openai.NotFoundError("gone", response=httpx.Response(404, request=request), body=None)

    graph = create_agent_graph(
        llm=Model(), retriever=Retriever(), settings=Settings(), summary_llm=Retired(),
        documents=access(tmp_path, [row("报告.pdf", DOC)], {DOC: parents_of("一")}),
    )
    result = ask(graph, "总结一下《报告》")
    assert result["coverage"]["read"] == 1 and result["refused"] is False


def test_summary_of_the_file_being_asked_within_needs_no_name(tmp_path):
    report = row("报告.pdf", DOC)
    graph = build(tmp_path, [report], {DOC: parents_of("一")})
    result = ask(graph, "这份文件主要讲了什么？", scope=[DOC])
    assert result["intent"] == "summarize_file" and result["target_file"]["file_id"] == DOC


def test_a_question_about_a_part_of_the_file_is_searched(tmp_path):
    retriever = Retriever()
    graph = build(tmp_path, [row("报告.pdf", DOC)], {DOC: parents_of("一")}, retriever=retriever)
    result = ask(graph, "《报告》里的项目总投资是多少？")
    assert result["intent"] == "knowledge" and retriever.calls


def test_failed_fact_check_is_flagged_and_does_not_search_again(tmp_path):
    retriever = Retriever()
    graph = build(tmp_path, [row("报告.pdf", DOC)], {DOC: parents_of("一", "二")}, model=Model(grounded=False), retriever=retriever)
    result = ask(graph, "总结一下《报告》")
    assert retriever.calls == []
    assert result["grounded"] is False and result["refused"] is False
    assert "1 处没能和文件内容对上" in result["answer"] and "已通读全文" in result["answer"]


def test_no_balance_is_said_plainly(tmp_path):
    class Broke(Model):
        def invoke(self, messages):
            raise status_error(402)

    graph = build(tmp_path, [row("报告.pdf", DOC)], {DOC: parents_of("一")}, model=Broke())
    result = ask(graph, "总结一下《报告》")
    assert result["refused"] is True and "余额不足" in result["answer"]


def test_a_file_without_text_says_so(tmp_path):
    graph = build(tmp_path, [row("报告.pdf", DOC)], {})
    result = ask(graph, "总结一下《报告》")
    assert result["refused"] is True and "还没有读取完成" in result["answer"]


def test_same_named_files_ask_which_and_the_reply_reads_it(tmp_path):
    a = row("01 申报书.pdf", "1" * 32, "C:\\资料\\01 面向精密级对准\\01 申报书.pdf")
    b = row("01 申报书.pdf", "2" * 32, "C:\\资料\\02 精密级合金粉末\\01 申报书.pdf")
    model = Model()
    graph = build(tmp_path, [a, b], {"1" * 32: parents_of("甲"), "2" * 32: parents_of("乙", "丙")}, model=model)
    first = ask(graph, "总结一下《01 申报书》这份文件的主要内容")
    assert "有 2 份《01 申报书》，要总结哪一份？" in first["answer"]
    assert "01 面向精密级对准" in first["answer"] and "02 精密级合金粉末" in first["answer"]
    assert first["choices"] == ["第 1 份", "第 2 份"]
    assert model.prompts == []  # asking back costs nothing
    second = ask(graph, "第 2 份")
    assert second["intent"] == "summarize_file" and second["target_file"]["file_id"] == "2" * 32
    assert second["coverage"]["read"] == 2
    assert not second["pending_file"]


def test_naming_the_folder_avoids_the_question(tmp_path):
    a = row("01 申报书.pdf", "1" * 32, "C:\\资料\\01 面向精密级对准\\01 申报书.pdf")
    b = row("01 申报书.pdf", "2" * 32, "C:\\资料\\02 精密级合金粉末\\01 申报书.pdf")
    graph = build(tmp_path, [a, b], {"1" * 32: parents_of("甲"), "2" * 32: parents_of("乙")})
    result = ask(graph, "总结一下精密级合金粉末里的《01 申报书》")
    assert result["target_file"].get("file_id", "") in ("", "2" * 32)


def test_another_question_after_asking_which_is_just_a_new_question(tmp_path):
    a, b = row("x.pdf", "1" * 32, "C:\\甲甲甲甲\\x.pdf"), row("x.pdf", "2" * 32, "C:\\乙乙乙乙\\x.pdf")
    retriever = Retriever()
    graph = build(tmp_path, [a, b], {}, retriever=retriever)
    ask(graph, "总结一下《x》")
    result = ask(graph, "KV Cache 的原理是什么？")
    assert result["intent"] == "knowledge" and not result["pending_file"] and retriever.calls


def test_a_very_long_file_says_what_it_costs_and_reads_on_a_yes(tmp_path):
    long_book = row("大书.pdf", DOC)
    model = Model()
    settings = Settings()
    settings.summary_confirm_tokens = 10
    graph = create_agent_graph(
        llm=model, retriever=Retriever(), settings=settings,
        documents=access(tmp_path, [long_book], {DOC: parents_of("字" * 40, "字" * 40)}),
    )
    first = ask(graph, "总结一下《大书》")
    assert "要开始吗" in first["answer"] and first["choices"] == ["开始通读"]
    assert model.prompts == []
    second = ask(graph, "开始通读")
    assert second["coverage"]["read"] == 2 and "已通读全文" in second["answer"]


def test_a_long_file_already_read_does_not_ask_again(tmp_path):
    book = row("大书.pdf", DOC)
    parents = {DOC: parents_of("字" * 40)}
    settings = Settings()
    documents = access(tmp_path, [book], parents)
    graph = create_agent_graph(llm=Model(), retriever=Retriever(), settings=settings, documents=documents)
    ask(graph, "总结一下《大书》")
    settings.summary_confirm_tokens = 10
    result = ask(graph, "总结一下《大书》", thread="other")
    assert "已通读全文" in result["answer"] and result["coverage"]["cached"] == 1


def test_follow_ups_offer_the_replies_without_a_model_call():
    service = FollowupService(SimpleNamespace(), chat_model=None, pool=lambda n: [], describe_file=lambda doc_id: None)
    result = service.suggest({"choices": ["第 1 份", "第 2 份"], "question": "总结"})
    assert [item.text for item in result.items] == ["第 1 份", "第 2 份"]
    assert result.source == "choices"


# ---------------------------------------------------------------- tables through the graph


TABLE = "b" * 32


@pytest.fixture
def policy(tmp_path):
    book = Workbook()
    sheet = book.active
    sheet.title = "科技研发"
    sheet.append(["序号", "类别", "政策方向"])
    for n in range(1, 13):
        sheet.append([n, "科技补贴", f"方向{n}"])
    other = book.create_sheet("商贸")
    other.append(["序号", "类别", "政策方向"])
    other.append([1, "商贸产业", "冷链"])
    path = tmp_path / "0.政策补贴汇总.xlsx"
    book.save(path)
    return path


def table_graph(tmp_path, path, model=None, retriever=None, original=True):
    rows = [row(path.name, TABLE, str(path))]
    return build(tmp_path, rows, {TABLE: parents_of("片段", prefix="page")}, model=model, retriever=retriever,
                 original={TABLE: path} if original else None)


def test_the_question_that_used_to_be_refused_gets_the_exact_count(tmp_path, policy):
    graph = table_graph(tmp_path, policy)
    result = ask(graph, "《0.政策补贴汇总》这张表里一共列了多少条政策？")
    assert result["intent"] == "table_query"
    assert "共 13 条" in result["answer"]
    assert result["grounded"] is True and result["refused"] is False
    assert {c["locator"] for c in result["citations"]} == {"sheet:科技研发;rows:2-13", "sheet:商贸;rows:2-2"}
    assert all(c["doc_id"] == TABLE for c in result["citations"])
    assert result["messages"][-1].content == result["answer"]


def test_the_file_name_given_as_a_sheet_counts_every_sheet(tmp_path, policy):
    # Qwen3-8B now and then answers {"sheet": "0.政策补贴汇总"} (answer_speed_v2, T01).
    model = Model(query='{"op":"count","sheet":"0.政策补贴汇总"}')
    result = ask(table_graph(tmp_path, policy, model=model), "《0.政策补贴汇总》这张表里一共列了多少条政策？")
    assert "共 13 条" in result["answer"] and result["refused"] is False
    # A sheet the question asks for by name that isn't there is still reported.
    model = Model(query='{"op":"count","sheet":"财务"}')
    result = ask(table_graph(tmp_path, policy, model=model), "《0.政策补贴汇总》的财务表有多少条？")
    assert "没有叫“财务”的工作表" in result["answer"]


@pytest.mark.parametrize("query", ['{"op":"count","sheet":"科技研发"}', '{"op":"count","sheet":"不存在"}'])
def test_a_sheet_the_question_never_names_is_not_what_it_counts(tmp_path, policy, query):
    # Qwen3-8B fills in the first sheet: "一共列了多少条政策" was answered 26 of 104 (T01).
    result = ask(table_graph(tmp_path, policy, model=Model(query=query)), "《0.政策补贴汇总》这张表里一共列了多少条政策？")
    answer = result["answer"]
    assert answer.startswith("共 13 条（2 个工作表")
    assert "- 科技研发：12 条" in answer and "- 商贸：1 条" in answer
    assert {c["locator"] for c in result["citations"]} == {"sheet:科技研发;rows:2-13", "sheet:商贸;rows:2-2"}
    assert result["refused"] is False


def test_a_sheet_the_question_names_is_the_only_one_counted(tmp_path, policy):
    model = Model(query='{"op":"count","sheet":"科技研发"}')
    result = ask(table_graph(tmp_path, policy, model=model), "《0.政策补贴汇总》的科技研发工作表有多少条政策？")
    assert result["answer"].startswith("「科技研发」共 12 条")
    assert {c["locator"] for c in result["citations"]} == {"sheet:科技研发;rows:2-13"}


def test_table_question_within_a_scoped_file(tmp_path, policy):
    model = Model(query='{"op":"count","sheet":"科技研发"}')
    graph = table_graph(tmp_path, policy, model=model)
    assert "「科技研发」共 12 条" in ask(graph, "科技研发这张表一共多少条？", scope=[TABLE])["answer"]
    # "这张表" within the file is the whole workbook, whichever sheet the model filled in (T21).
    assert "共 13 条（2 个工作表" in ask(graph, "这张表一共多少条？", thread="t2", scope=[TABLE])["answer"]


def test_a_question_about_content_goes_to_search(tmp_path, policy):
    retriever = Retriever()
    graph = table_graph(tmp_path, policy, retriever=retriever)
    result = ask(graph, "《0.政策补贴汇总》里研发费补贴的标准是什么？")
    assert result["intent"] == "knowledge" and retriever.calls


def test_when_the_query_says_none_search_answers_within_the_file(tmp_path, policy):
    retriever = Retriever()
    graph = table_graph(tmp_path, policy, model=Model(query='{"op":"none"}'), retriever=retriever)
    result = ask(graph, "《0.政策补贴汇总》有哪些事需要注意？")
    assert result["intent"] == "knowledge"
    assert retriever.calls and retriever.calls[0] == {TABLE}


def test_a_query_the_model_garbled_goes_to_search_too(tmp_path, policy):
    retriever = Retriever()
    graph = table_graph(tmp_path, policy, model=Model(query="我不知道"), retriever=retriever)
    assert ask(graph, "《0.政策补贴汇总》一共多少条")["intent"] == "knowledge" and retriever.calls


def test_a_query_outside_the_fixed_operations_never_runs(tmp_path, policy):
    retriever = Retriever()
    graph = table_graph(tmp_path, policy, model=Model(query='{"op":"exec","sheet":"__import__"}'), retriever=retriever)
    result = ask(graph, "《0.政策补贴汇总》一共多少条")
    assert result["intent"] == "knowledge" and retriever.calls


def test_a_column_that_does_not_exist_is_said_plainly(tmp_path, policy):
    model = Model(query='{"op":"sum","sheet":"科技研发","column":"金额"}')
    graph = table_graph(tmp_path, policy, model=model)
    result = ask(graph, "《0.政策补贴汇总》科技研发的金额合计多少")
    assert result["refused"] is True and "工作表“科技研发”没有“金额”这一列" in result["answer"]
    # No sheet named: none of them has it.
    result = ask(graph, "《0.政策补贴汇总》金额合计多少", thread="t2")
    assert result["refused"] is True and "没有工作表同时包含：金额" in result["answer"]


def test_a_missing_original_is_said_plainly(tmp_path, policy):
    result = ask(table_graph(tmp_path, policy, original=False), "《0.政策补贴汇总》一共多少条")
    assert result["refused"] is True and "找不到" in result["answer"]


def test_a_very_big_workbook_is_refused_with_the_reason(tmp_path, policy, monkeypatch):
    monkeypatch.setattr("rag_agent.document_nodes.TABLE_MAX_MB", 0)
    result = ask(table_graph(tmp_path, policy), "《0.政策补贴汇总》一共多少条")
    assert result["refused"] is True and "暂时不能整表统计" in result["answer"]


def test_a_broken_workbook_is_said_plainly(tmp_path):
    path = tmp_path / "坏.xlsx"
    path.write_bytes(b"not a workbook")
    result = ask(table_graph(tmp_path, path), "《坏》一共多少条")
    assert result["refused"] is True and "打不开" in result["answer"]


def test_same_named_workbooks_ask_which_and_keep_the_question(tmp_path, policy):
    second = tmp_path / "别处"
    second.mkdir()
    copy = second / policy.name
    book = Workbook()
    book.active.append(["序号", "名称"])
    book.active.append([1, "只有一行"])
    book.save(copy)
    rows = [row(policy.name, TABLE, str(policy)), row(copy.name, "c" * 32, str(copy))]
    graph = build(tmp_path, rows, {}, original={TABLE: policy, "c" * 32: copy})
    first = ask(graph, "《0.政策补贴汇总》一共多少条")
    assert "要统计哪一份" in first["answer"]
    result = ask(graph, "第 2 份")
    assert result["intent"] == "table_query" and "共 1 条" in result["answer"]


def test_a_pdf_named_with_a_count_question_is_not_a_table_question(tmp_path):
    retriever = Retriever()
    graph = build(tmp_path, [row("01 申报书.pdf", DOC)], {}, retriever=retriever)
    assert ask(graph, "《01 申报书》一共多少页？")["intent"] == "knowledge" and retriever.calls


def test_without_a_library_nothing_changes(tmp_path):
    graph = create_agent_graph(llm=Model(), retriever=Retriever(), settings=Settings())
    assert ask(graph, "总结一下《报告》的主要内容")["intent"] == "knowledge"
