"""Follow-up questions after an answer: scenarios, the model call and its fallbacks, rewording, the switch."""

import json
from pathlib import Path
import time
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from config import Settings
from suggestions.followups import (
    SCENE_ANSWER, SCENE_CHAT, SCENE_FILES, SCENE_NONE, SCENE_REFUSED, SCENE_SCOPED, SCOPE_CLEAR, SCOPE_KEEP,
    SCOPE_SET, Followup, FollowupService, Turn, dedupe, on_topic, similarity,
)

BOOK = "b" * 32
DEAL = "d" * 32
MEMO = "e" * 32

BOOK_UNITS = [
    {"heading": "第 3 章 推理加速 > 3.1 KV Cache", "locator": "page:40", "text": "KV Cache 缓存每一层的键和值。"},
    {"heading": "第 3 章 推理加速 > 3.2 PagedAttention 机制", "locator": "page:41-42", "text": "按页管理显存。"},
    {"heading": "第 3 章 推理加速 > 3.3 投机解码", "locator": "page:43", "text": "小模型先猜。"},
    {"heading": "第 3 章 推理加速 > 本章小结", "locator": "page:44", "text": "小结。"},
    {"heading": "第 4 章 微调 > 4.1 LoRA 方法", "locator": "page:45", "text": "低秩适配。"},
]


class FakeChat:
    """Streams its reply line by line: ``delay`` before the first line, ``gap`` between lines."""

    def __init__(self, reply="", delay=0.0, error=None, gap=0.0):
        self.reply, self.delay, self.error, self.gap = reply, delay, error, gap
        self.prompts = []

    def stream(self, prompt):
        self.prompts.append(prompt)
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        for index, line in enumerate(self.reply.splitlines(keepends=True)):
            if index and self.gap:
                time.sleep(self.gap)
            yield SimpleNamespace(content=line, usage_metadata=None)
        yield SimpleNamespace(content="", usage_metadata={"input_tokens": 420, "output_tokens": 48})


@pytest.fixture
def settings(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, siliconflow_api_key="sk-test-key-123",
                        followup_timeout_seconds=1.0)
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    (settings.parsed_dir / f"{BOOK}.json").write_text(
        json.dumps({"source": "图解大模型.pdf", "units": BOOK_UNITS}, ensure_ascii=False), encoding="utf-8"
    )
    return settings


def service(settings, chat=None, pool=None, names=None):
    names = names or {BOOK: "图解大模型.pdf", DEAL: "苏州三方协议.docx"}
    return FollowupService(
        settings,
        chat_model=(lambda: chat) if chat is not None else None,
        pool=pool,
        describe_file=lambda doc_id: {"name": names[doc_id]} if doc_id in names else None,
    )


def doc(doc_id=BOOK, source="图解大模型.pdf", heading="第 3 章 推理加速 > 3.1 KV Cache", locator="page:40", text="KV Cache 缓存每一层的键和值。"):
    return {"doc_id": doc_id, "source": source, "heading": heading, "locator": locator, "text": text}


def history(*questions):
    messages = []
    for question in questions:
        messages += [HumanMessage(content=question), AIMessage(content="……")]
    return messages


def answer_state(**extra):
    unused = doc(heading="第 4 章 微调 > 4.1 LoRA 方法", locator="page:45", text="低秩适配。")
    state = {
        "question": "什么是 KV Cache？",
        "intent": "knowledge",
        "rewritten_query": "什么是 KV Cache？",
        "answer": "KV Cache 缓存了每一层注意力的键和值，避免重复计算 [来源：图解大模型.pdf｜位置：page:40]。",
        "context_docs": [doc(), unused],
        "documents": [doc(), unused],
        "messages": history("什么是 KV Cache？"),
        "scope_doc_ids": [],
    }
    state.update(extra)
    return state


def files_state(**extra):
    state = {
        "question": "找一下苏州三方协议",
        "intent": "find_file",
        "answer": "找到 2 个相关文件：…",
        "files": [
            {"file_id": DEAL, "name": "苏州三方协议.docx", "ext": ".docx",
             "snippets": [{"heading": "二、合作内容", "text": "三方就电池回收开展合作……"}]},
            {"file_id": MEMO, "name": "杭州电池合作纪要.pdf", "ext": ".pdf", "snippets": []},
        ],
        "file_search": {"filters": {"file_types": []}},
        "messages": history("找一下苏州三方协议"),
        "grounded": True,
    }
    state.update(extra)
    return state


def refused_state(**extra):
    weak = doc(heading="第 3 章 推理加速 > 3.3 投机解码", locator="page:43", text="小模型先猜。")
    state = {
        "question": "推理怎么加速到十倍？",
        "intent": "knowledge",
        "rewritten_query": "推理加速 十倍",
        "answer": "当前知识库没有足够依据回答这个问题。",
        "refused": True,
        "documents": [weak],
        "context_docs": [],
        "messages": history("推理怎么加速到十倍？"),
    }
    state.update(extra)
    return state


# ---------------------------------------------------------------- scenarios


def test_each_kind_of_turn_gets_its_scenario():
    assert Turn.from_state(answer_state()).scenario == SCENE_ANSWER
    assert Turn.from_state(answer_state(scope_doc_ids=[BOOK])).scenario == SCENE_SCOPED
    assert Turn.from_state(files_state()).scenario == SCENE_FILES
    assert Turn.from_state(files_state(files=[])).scenario == SCENE_REFUSED
    assert Turn.from_state(refused_state()).scenario == SCENE_REFUSED
    assert Turn.from_state({"question": "你好", "intent": "chitchat", "answer": "你好！"}).scenario == SCENE_CHAT
    clarify = answer_state(plan={"needs_clarification": True})
    assert Turn.from_state(clarify).scenario == SCENE_NONE
    assert Turn.from_state(answer_state(model_error="服务繁忙")).scenario == SCENE_NONE


def test_cited_and_unused_passages_are_told_apart():
    turn = Turn.from_state(answer_state())
    assert [p.section for p in turn.cited] == ["KV Cache"]
    assert [p.section for p in turn.unused] == ["LoRA 方法"]
    assert turn.asked == ["什么是 KV Cache？"]


# ---------------------------------------------------------------- templates (no key, timeout, error)


def test_without_a_key_templates_ask_about_the_neighbouring_sections(settings):
    settings.siliconflow_api_key = ""
    chat = FakeChat("不应调用？")
    result = service(settings, chat).suggest(answer_state())
    texts = [item.text for item in result.items]
    assert result.source == "template" and result.fallback == "no_key" and not chat.prompts
    assert texts[0] == "PagedAttention 机制是如何工作的？"
    assert "《图解大模型》中“投机解码”部分讲了什么？" in texts
    # The retrieved-but-unused passage, and never the boilerplate "本章小结".
    assert "什么是LoRA 方法？" in texts and not any("小结" in t for t in texts)
    assert all(item.scope_action == SCOPE_KEEP for item in result.items)


def test_found_files_are_asked_about_within_that_file(settings):
    result = service(settings, chat=None).suggest(files_state(), use_model=False)
    first = result.items[0]
    assert first.text == "《苏州三方协议》约定了哪些合作内容？"
    assert first.scope_action == SCOPE_SET and first.scope == {"doc_ids": [DEAL], "label": "苏州三方协议.docx"}
    assert first.label == "文件内"
    assert any(i.text == "《杭州电池合作纪要》安排了哪些工作重点？" and i.scope["doc_ids"] == [MEMO] for i in result.items)
    # Two formats among the results: narrowing to the first one's is offered.
    assert any(i.text == "找一下苏州三方协议，只要 Word 文档" and i.kind == "refine_search" for i in result.items)


def test_one_file_found_still_gets_a_full_set_from_its_outline(settings):
    units = [{"heading": "一、合作内容", "text": "…"}, {"heading": "二、各方责任", "text": "…"},
             {"heading": "三、违约责任", "text": "…"}]
    (settings.parsed_dir / f"{DEAL}.json").write_text(json.dumps({"units": units}, ensure_ascii=False), encoding="utf-8")
    state = files_state(files=files_state()["files"][:1])
    result = service(settings).suggest(state, use_model=False)
    texts = [item.text for item in result.items]
    assert len(texts) == 4 and all(item.scope["doc_ids"] == [DEAL] for item in result.items)
    # The matched section ("合作内容") would only repeat the first question in other words.
    assert texts == ["《苏州三方协议》约定了哪些合作内容？", "《苏州三方协议》中“各方责任”部分讲了什么？",
                     "《苏州三方协议》中“违约责任”部分讲了什么？", "《苏州三方协议》主要讲了什么？"]


def test_after_a_refusal_rephrase_find_files_or_widen(settings):
    result = service(settings).suggest(refused_state(), use_model=False)
    kinds = [item.kind for item in result.items]
    assert kinds == ["find_files", "rephrase", "related"]
    assert result.items[0].text == "找一下和推理加速有关的文件" and result.items[0].label == "找文件"
    assert result.items[1].text == "什么是推理加速？"
    assert result.items[2].text == "《图解大模型》中“投机解码”部分讲了什么？"

    scoped = service(settings).suggest(refused_state(scope_doc_ids=[BOOK]), use_model=False)
    widen = scoped.items[0]
    assert widen.kind == "widen" and widen.text == "推理怎么加速到十倍？" and widen.scope_action == SCOPE_CLEAR
    # Finding files and the broader question cannot work inside one file: they leave it too.
    assert [(i.kind, i.scope_action) for i in scoped.items[1:3]] == [("find_files", SCOPE_CLEAR), ("rephrase", SCOPE_CLEAR)]
    model = service(settings, FakeChat("找一下推理加速相关的文件")).suggest(refused_state(scope_doc_ids=[BOOK]))
    assert model.items[0].kind == "find_files" and model.items[0].scope_action == SCOPE_CLEAR


def test_after_a_refusal_unrelated_passages_are_not_suggested(settings):
    funding = doc(doc_id=MEMO, source="前海政策资金支持梳理.docx", heading="一、条件适用", text="资金支持条件……")
    state = refused_state(question="Manus 项目的初始投资金额是多少？", rewritten_query="Manus 项目 初始投资 金额",
                          documents=[funding], messages=history("Manus 项目的初始投资金额是多少？"))
    result = service(settings).suggest(state, use_model=False)
    assert [item.text for item in result.items] == ["找一下和 Manus 有关的文件", "什么是 Manus？"]
    # The model wrote questions about the funding documents only: none is kept.
    chat = FakeChat("前海政策对专精特新企业的资金支持是多少？\n找一下前海政策资金支持梳理里的资金总表\nManus 是哪家公司开发的？")
    result = service(settings, chat).suggest(state)
    assert [item.text for item in result.items] == [
        "Manus 是哪家公司开发的？", "找一下和 Manus 有关的文件", "什么是 Manus？"]


def test_no_file_found_suggests_looking_differently(settings):
    result = service(settings).suggest(files_state(files=[], question="找一下去年的报价单 PDF",
                                                   messages=history("找一下去年的报价单 PDF")), use_model=False)
    assert result.scenario == SCENE_REFUSED
    assert result.items and result.items[-1].kind == "find_files"


def test_asking_within_one_file_moves_on_to_its_next_section(settings):
    state = answer_state(scope_doc_ids=[BOOK])
    result = service(settings).suggest(state, use_model=False)
    assert result.scenario == SCENE_SCOPED
    assert result.items[0].text == "PagedAttention 机制是如何工作的？"
    # Only that file: the other chapter's retrieved passage is not suggested here.
    assert not any("LoRA" in item.text for item in result.items)


def test_scoped_question_without_citations_starts_at_the_files_beginning(settings):
    state = answer_state(scope_doc_ids=[BOOK], answer="根据资料……", context_docs=[], documents=[])
    texts = [item.text for item in service(settings).suggest(state, use_model=False).items]
    # The first section's question ("什么是KV Cache？") was just asked, so the next ones follow.
    assert texts[:2] == ["PagedAttention 机制是如何工作的？", "《图解大模型》中“投机解码”部分讲了什么？"]


def test_small_talk_draws_from_the_home_page_pool(settings):
    chat = FakeChat("不应调用？")
    cards = [{"question": "什么是 KV Cache？", "qtype": "ask"}, {"question": "找一下苏州三方协议", "qtype": "find"},
             {"question": "LoRA 是什么？", "qtype": "ask"}]
    state = {"question": "你好", "intent": "chitchat", "answer": "你好！", "messages": history("什么是 KV Cache？", "你好")}
    result = service(settings, chat, pool=lambda n: cards).suggest(state)
    assert result.source == "pool" and not chat.prompts
    # Already asked in this conversation: dropped.
    assert [item.text for item in result.items] == ["找一下苏州三方协议", "LoRA 是什么？"]
    assert result.items[0].label == "找文件"


def test_clarifying_questions_get_no_followups(settings):
    chat = FakeChat("A？")
    result = service(settings, chat).suggest(answer_state(plan={"needs_clarification": True}))
    assert result.items == [] and not chat.prompts


def test_a_slow_model_is_abandoned_for_templates(settings):
    settings.followup_timeout_seconds = 0.2
    chat = FakeChat("PagedAttention 是怎么解决 KV Cache 显存碎片的？", delay=1.0)
    started = time.perf_counter()
    result = service(settings, chat).suggest(answer_state())
    assert time.perf_counter() - started < 0.8
    assert result.fallback == "timeout" and result.source == "template" and result.items


def test_at_the_deadline_the_questions_already_written_are_kept(settings):
    settings.followup_timeout_seconds = 0.3
    chat = FakeChat("PagedAttention 是怎么解决 KV Cache 显存碎片的？\nKV Cache 会占用多少显存？\n投机解码", gap=1.0)
    svc = service(settings, chat)
    result = svc.suggest(answer_state())
    assert result.source == "model" and result.fallback == "partial" and result.usage is None
    # The first line arrived in time; templates add the rest.
    assert result.items[0].text == "PagedAttention 是怎么解决 KV Cache 显存碎片的？"
    assert result.items[1].kind == "next_section"
    # The call still finishes in the background and its tokens are counted.
    time.sleep(2.2)
    assert svc.stats == {"calls": 1, "completed": 1, "input_tokens": 420, "output_tokens": 48}


def test_a_model_error_falls_back_to_templates(settings):
    result = service(settings, FakeChat(error=RuntimeError("429"))).suggest(answer_state())
    assert result.fallback == "error" and result.source == "template"
    assert result.items[0].text == "PagedAttention 机制是如何工作的？"


# ---------------------------------------------------------------- the model call


def test_model_questions_come_first_and_stay_on_the_material(settings):
    chat = FakeChat(
        "1. PagedAttention 是怎么解决 KV Cache 显存碎片的？\n"
        "- 量子计算机会取代 GPU 吗？\n"
        "KV Cache 会占用多少显存？\n"
        "说明："
    )
    result = service(settings, chat).suggest(answer_state())
    texts = [item.text for item in result.items]
    assert result.source == "model" and result.usage == {"input_tokens": 420, "output_tokens": 48}
    assert texts[:2] == ["PagedAttention 是怎么解决 KV Cache 显存碎片的？", "KV Cache 会占用多少显存？"]
    assert not any("量子" in t for t in texts)
    # Templates fill the rest.
    assert len(texts) == 4


def test_details_the_passages_never_mention_are_not_suggested_first(settings):
    chat = FakeChat("KV Cache 显存占用的计算公式是什么？\nKV Cache 缓存了哪些内容？")
    texts = [item.text for item in service(settings, chat).suggest(answer_state(), n=6).items]
    # The template questions (sections that exist) come before it: it is an alternative only.
    assert texts[0] == "KV Cache 缓存了哪些内容？" and texts[-1] == "KV Cache 显存占用的计算公式是什么？"
    # When the answer does give one, asking for it is fine.
    state = answer_state(answer="KV Cache 显存可按公式 2×层数×隐藏维度 估算 [来源：图解大模型.pdf｜位置：page:40]。")
    texts = [item.text for item in service(settings, chat).suggest(state).items]
    assert texts[0] == "KV Cache 显存占用的计算公式是什么？"


def test_how_exactly_probes_are_never_the_grey_suggestion(settings):
    chat = FakeChat("KV Cache 具体是怎么分配显存的？\nPagedAttention 和 KV Cache 有什么关系？")
    result = service(settings, chat).suggest(answer_state())
    assert result.items[0].text == "PagedAttention 和 KV Cache 有什么关系？" and result.source == "model"
    assert "KV Cache 具体是怎么分配显存的？" not in [item.text for item in result.items]
    # With only probes from the model, the first suggestion is a template one.
    only = service(settings, FakeChat("KV Cache 具体怎么实现？\n两种方法的差距是多少？")).suggest(answer_state(), n=6)
    assert only.source == "template" and only.items[0].kind == "next_section"
    # The probe stays as the last alternative; "两种方法…" names nothing from this turn and is dropped.
    assert only.items[-1].text == "KV Cache 具体怎么实现？" and len(only.items) == 5
    from suggestions.followups import DETAIL_PROBE

    # Run 4's one miss, and questions that are fine.
    assert DETAIL_PROBE.search("多参考音和单一参考音效果差多少？")
    assert DETAIL_PROBE.search("PagedAttention 能把吞吐提高多少？")
    assert not DETAIL_PROBE.search("在轨蒸馏中一条rollout能产生多少组监督信号？")
    assert not DETAIL_PROBE.search("难负例和简单负例在训练时有什么区别？")


def test_the_prompt_holds_only_this_turn(settings):
    chat = FakeChat("PagedAttention 是怎么解决 KV Cache 显存碎片的？")
    service(settings, chat).suggest(answer_state(messages=history("LoRA 是什么？", "什么是 KV Cache？")))
    prompt = chat.prompts[0]
    assert "本轮问题：什么是 KV Cache？" in prompt
    assert "回答摘要：KV Cache 缓存了每一层注意力的键和值，避免重复计算" in prompt and "[来源" not in prompt
    assert "- 《图解大模型》KV Cache" in prompt
    assert "同一文件里相邻的小节：\n- 《图解大模型》PagedAttention 机制" in prompt
    assert "检索到但回答没用上的片段：\n- 《图解大模型》LoRA 方法" in prompt
    assert "已经问过：LoRA 是什么？" in prompt
    # Passage text is not repeated to the model for an answered question; section names are enough.
    assert "低秩适配" not in prompt


def test_a_found_file_kept_by_name_only_is_not_told_to_the_model(settings):
    chat = FakeChat("《杭州电池合作纪要》讲了什么？")
    state = files_state()
    state["files"][0] = {**state["files"][0], "status": "named", "name": "ZJQQ护照扫描.txt"}
    service(settings, chat).suggest(state)
    prompt = chat.prompts[0]
    assert "ZJQQ" not in prompt and "三方就电池回收" not in prompt
    assert "1. 《杭州电池合作纪要》" in prompt


def test_model_questions_about_a_found_file_are_scoped_to_it(settings):
    chat = FakeChat("《苏州三方协议》里三方各自负责什么？\n找一下杭州电池合作相关的 PDF\n《杭州电池合作纪要》的会议时间是哪天？")
    result = service(settings, chat).suggest(files_state())
    first, second, third = result.items[:3]
    assert first.scope_action == SCOPE_SET and first.scope["doc_ids"] == [DEAL]
    assert second.kind == "refine_search" and second.scope_action == SCOPE_KEEP and second.label == "找文件"
    assert third.scope["doc_ids"] == [MEMO]


def test_after_a_scoped_refusal_widening_is_second(settings):
    chat = FakeChat("投机解码能把推理加速多少倍？")
    result = service(settings, chat).suggest(refused_state(scope_doc_ids=[BOOK]))
    assert [item.kind for item in result.items[:2]] == ["rephrase", "widen"]


def test_after_a_refusal_rephrasings_keep_the_question_s_topic(settings):
    # The weak passages were about another subject: questions about it are not offered.
    chat = FakeChat("小模型先猜的原理是什么？\n投机解码能让推理加速多少？\n找一下推理加速相关的文件")
    result = service(settings, chat).suggest(refused_state())
    texts = [item.text for item in result.items]
    assert texts[:2] == ["投机解码能让推理加速多少？", "找一下推理加速相关的文件"]
    # Without a search query to name the subject, sharing a word with the question is enough.
    state = refused_state(rewritten_query="", question="?")
    assert service(settings, FakeChat("小模型先猜的原理是什么？")).suggest(state).source == "template"


# ---------------------------------------------------------------- rewording


def test_asked_questions_and_rewordings_of_the_last_one_are_dropped():
    asked = ["LoRA 是什么？", "PagedAttention 是怎么解决 KV Cache 显存碎片的？"]
    items = [
        Followup("PagedAttention 如何解决 KV Cache 的显存碎片问题？", "ask"),  # rewording of the last
        Followup("lora是什么", "ask"),  # asked before, other punctuation and case
        Followup("KV Cache 有什么作用？", "ask"),
        Followup("KV Cache 有什么作用呢？", "ask"),  # near-duplicate of the one above
        Followup("PagedAttention 是怎么解决 KV Cache 显存碎片的？", "widen", scope_action=SCOPE_CLEAR),
        Followup("投机解码是如何工作的？", "ask"),
    ]
    kept = [item.text for item in dedupe(items, asked, 4)]
    assert kept == ["KV Cache 有什么作用？", "PagedAttention 是怎么解决 KV Cache 显存碎片的？", "投机解码是如何工作的？"]
    assert similarity("什么是 KV Cache？", "KV Cache 有什么作用？") < 0.6
    # Same topic, question words moved around: a rewording.
    assert [i.text for i in dedupe([Followup("KV Cache 是什么？", "ask")], ["什么是 KV Cache？"], 4)] == []
    assert [i.text for i in dedupe([Followup("《苏州三方协议》约定了哪些合作内容？", "ask_file"),
                                    Followup("《苏州三方协议》提到的合作内容有哪些？", "ask_file")], [], 4)] == [
        "《苏州三方协议》约定了哪些合作内容？"]


def test_off_topic_check():
    from suggestions.followups import _content_bigrams

    material = _content_bigrams("KV Cache 缓存键值 PagedAttention 显存 图解大模型")
    assert on_topic("PagedAttention 是怎么管理显存的？", material)
    assert not on_topic("量子计算机会取代 GPU 吗？", material)


# ---------------------------------------------------------------- system and API


def test_followups_belong_to_the_turn_that_finished(settings):
    from core.rag_system import RAGSystem

    system = RAGSystem(settings)
    system.initialized = True
    system.followup_service = service(settings)
    values = answer_state()
    system.agent_graph = SimpleNamespace(get_state=lambda config: SimpleNamespace(values=values))
    assert system.followups("t1", "什么是 KV Cache？")["items"]
    # A newer question is running in this conversation: nothing.
    assert system.followups("t1", "别的问题")["items"] == []
    system.agent_graph = SimpleNamespace(get_state=lambda config: SimpleNamespace(values={}))
    assert system.followups("t1", "什么是 KV Cache？") == {
        "scenario": "", "source": "", "fallback": "", "elapsed_ms": 0, "usage": None, "items": []}


def test_the_switch_turns_followups_off(settings, monkeypatch):
    from fastapi.testclient import TestClient

    from api import main

    calls = []

    class System:
        initialized = True

        def followups(self, thread_id, question="", n=4):
            calls.append((thread_id, question, n))
            return {"scenario": "answer", "source": "template", "items": [{"text": "下一问？"}]}

    settings.app_token = ""
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "system", System())
    client = TestClient(main.app)

    assert client.get("/api/v1/settings").json()["followups_enabled"] is True
    on = client.post("/api/v1/followups", json={"thread_id": "t1", "question": "什么是 KV Cache？"}).json()
    assert on["enabled"] is True and on["items"] == [{"text": "下一问？"}] and calls == [("t1", "什么是 KV Cache？", 4)]

    assert client.put("/api/v1/settings", json={"followups_enabled": False}).json()["followups_enabled"] is False
    off = client.post("/api/v1/followups", json={"thread_id": "t1", "question": "什么是 KV Cache？"}).json()
    assert off == {"enabled": False, "scenario": "", "items": []} and len(calls) == 1
    stored = json.loads((Path(settings.data_dir) / "preferences.json").read_text(encoding="utf-8"))
    assert stored["followups_enabled"] is False
    assert client.post("/api/v1/followups", json={"thread_id": ""}).status_code == 422


def test_a_conversation_with_nothing_answered_has_nothing_to_follow(settings, monkeypatch):
    from fastapi.testclient import TestClient

    from api import main
    from core.rag_system import RAGSystem

    settings.app_token = ""
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "system", RAGSystem(settings))
    client = TestClient(main.app)
    body = client.post("/api/v1/followups", json={"thread_id": "t1"}).json()
    assert body["enabled"] is True and body["scenario"] == "" and body["items"] == []
