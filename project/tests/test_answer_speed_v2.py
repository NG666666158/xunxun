"""answer_speed_v2: literal statements skip the check model, small talk skips the planner,
the planner writes less, background builders wait, follow-ups are written ahead, and model
calls take the direct route when it connects."""

from __future__ import annotations

from collections import OrderedDict
import json
import threading

from langchain_core.messages import AIMessage, HumanMessage
import httpx
import pytest

from rag_agent.answering import literally_supported
from rag_agent.nodes import AgentNodes
from services import model_clients
from services.model_clients import RoutedTransport


PASSAGE = "纳米硅粉体产品的粒径为 2~5 nm，纯度不低于 99.5%，可分散于水和乙醇。"


@pytest.mark.parametrize(
    ("claim", "supported"),
    [
        ("纳米硅粉体产品的粒径为 2~5 nm", True),
        ("纳米硅粉体的粒径是2-5nm", True),  # joining words and spacing differ
        ("纯度不低于 99.5%", True),
        ("可分散于水和乙醇", True),
        ("纳米硅粉体产品的粒径为 2~8 nm", False),  # a number the passage does not have
        ("纯度低于 99.5%", False),  # the negation dropped
        ("不可分散于水和乙醇", False),  # a negation added
        ("纯度最高可达 99.5%", False),  # a degree word added
        ("纳米硅粉体具有优异的导电性能", False),  # reworded or new: left to the model
        ("粒径", False),  # too short to judge by its words
    ],
)
def test_literal_statements_are_recognised_strictly(claim, supported):
    assert literally_supported(claim, [PASSAGE]) is supported


class CheckerLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(str(messages[1].content))
        return AIMessage(content=json.dumps({"grounded": True, "unsupported_claims": []}))


class StrictSettings:
    answer_mode = "strict"
    max_retrieval_attempts = 2


BLOCKS = [{"source": "规格书.pdf", "locator": "page:2", "text": PASSAGE}]


def test_an_answer_that_copies_its_passages_needs_no_check_model():
    llm = CheckerLLM()
    answer = "纳米硅粉体产品的粒径为 2~5 nm [来源：规格书.pdf｜位置：page:2]，纯度不低于 99.5% [来源：规格书.pdf｜位置：page:2]。"
    result = AgentNodes(llm, None, StrictSettings()).fact_check({"answer": answer, "context_docs": BLOCKS})
    assert result == {"grounded": True, "unsupported_claims": [], "check_skipped": False, "check_mode": "literal"}
    assert llm.prompts == []


def test_only_reworded_statements_go_to_the_check_model():
    llm = CheckerLLM()
    answer = ("纳米硅粉体产品的粒径为 2~5 nm [来源：规格书.pdf｜位置：page:2]。\n"
              "它非常适合锂电池导电剂 [来源：规格书.pdf｜位置：page:2]。")
    result = AgentNodes(llm, None, StrictSettings()).fact_check({"answer": answer, "context_docs": BLOCKS})
    assert result["check_mode"] == "cited"
    assert len(llm.prompts) == 1
    assert "锂电池导电剂" in llm.prompts[0] and "粒径为 2~5 nm（依据" not in llm.prompts[0]


def test_a_citation_to_missing_evidence_still_fails_without_the_literal_shortcut():
    answer = "纳米硅粉体产品的粒径为 2~5 nm [来源：别的文件.pdf｜位置：page:9]。"
    result = AgentNodes(CheckerLLM(), None, StrictSettings()).fact_check({"answer": answer, "context_docs": BLOCKS})
    assert result["grounded"] is False


class CountingPlanner:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content='{"intent":"knowledge","rewritten_query":"纳米硅胶体产品的粒径"}')


@pytest.mark.parametrize("greeting", ["你好", "谢谢！", "你能做什么？"])
def test_small_talk_skips_the_planner_even_in_a_conversation(greeting):
    llm = CountingPlanner()
    messages = [HumanMessage(content="纳米硅的粒径？"), AIMessage(content="2~5 nm"), HumanMessage(content=greeting)]
    update = AgentNodes(llm, None).analyze_query({"messages": messages})
    assert update["intent"] == "chitchat" and llm.calls == []


def test_the_planner_is_asked_for_the_fields_it_uses_only():
    llm = CountingPlanner()
    messages = [HumanMessage(content="纳米硅粉体的粒径是多少？"), AIMessage(content="2~5 nm"),
                HumanMessage(content="那胶体产品呢？")]
    update = AgentNodes(llm, None).analyze_query({"messages": messages})
    assert update["rewritten_query"] == "纳米硅胶体产品的粒径"
    assert "不要输出" in str(llm.calls[0][-1].content)


def test_the_suggestion_builder_waits_for_a_question_being_answered(tmp_path):
    from suggestions.service import SuggestionService

    waited = []
    service = SuggestionService.__new__(SuggestionService)
    service.pause = lambda: waited.append(True)
    service._yield_to_questions()
    assert waited == [True]
    service.pause = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    service._yield_to_questions()  # a failing pause never stops the builder


# --- the route to the model service ---------------------------------------------------


class Recorder(httpx.BaseTransport):
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.seen = []

    def handle_request(self, request):
        self.seen.append(request.extensions.get("timeout", {}).get("connect"))
        if self.fail:
            raise httpx.ConnectTimeout("no route")
        return httpx.Response(200, json={"ok": True})


def routed(mode: str, direct_fails: bool):
    transport = RoutedTransport("http://127.0.0.1:7892", mode, httpx.Limits())
    transport.direct, transport.proxied = Recorder(direct_fails), Recorder()
    return transport, httpx.Client(transport=transport, timeout=httpx.Timeout(30.0, connect=10.0))


def test_auto_goes_direct_when_a_direct_connection_opens():
    transport, client = routed("auto", direct_fails=False)
    client.get("https://api.example.cn/v1/models")
    client.get("https://api.example.cn/v1/models")
    assert transport.route == "direct"
    # Only the first, undecided request had to connect within the short limit.
    assert transport.direct.seen == [model_clients.DIRECT_CONNECT_SECONDS, 10.0]
    assert transport.proxied.seen == []


def test_auto_falls_back_to_the_proxy_and_resends_the_request():
    transport, client = routed("auto", direct_fails=True)
    assert client.get("https://api.example.cn/v1/models").status_code == 200
    assert transport.route == "proxy"
    assert transport.proxied.seen == [10.0]  # the proxy gets the normal connect timeout
    client.get("https://api.example.cn/v1/models")
    assert len(transport.direct.seen) == 1  # direct is not retried for a while


def test_a_fixed_route_is_kept():
    transport, client = routed("proxy", direct_fails=False)
    client.get("https://api.example.cn/v1/models")
    assert transport.direct.seen == [] and transport.route == "proxy"
    transport, client = routed("direct", direct_fails=True)
    with pytest.raises(httpx.ConnectTimeout):
        client.get("https://api.example.cn/v1/models")
    assert transport.proxied.seen == []


def test_without_a_proxy_everything_goes_direct():
    transport = RoutedTransport(None, "auto", httpx.Limits())
    assert transport.route == "direct" and transport.proxied is None


# --- follow-ups written while the answer is checked -----------------------------------


def test_follow_ups_written_ahead_are_served_for_the_same_answer(monkeypatch):
    from core.rag_system import RAGSystem

    system = RAGSystem.__new__(RAGSystem)
    system._followup_jobs = OrderedDict()
    system._followup_lock = threading.Lock()
    system._followup_pool = None
    written = []

    class Result:
        source = "model"
        fallback = ""

        def __init__(self, label):
            self.label = label

        def to_dict(self):
            return {"items": [self.label]}

    def suggest(values, n, ahead=False):
        written.append((values["answer"], ahead))
        return Result("ahead" if ahead else "now")

    monkeypatch.setattr(system, "_suggest_followups", suggest)
    values = {"question": "粒径？", "answer": "2~5 nm"}
    system._start_followups("t1", dict(values))
    system._followup_jobs["t1"][1].result(timeout=5)

    class Graph:
        def __init__(self, state):
            self.state = state

        def get_state(self, config):
            return type("Snapshot", (), {"values": self.state})()

    class Settings:
        followup_timeout_seconds = 4.0
        log_dir = None

    system.settings = Settings()
    system.initialize = lambda: system
    system._restore_thread = lambda thread_id: None
    monkeypatch.setattr("core.timings.save", lambda record, log_dir: None)
    system.agent_graph = Graph(values)
    assert system.followups("t1", "粒径？") == {"items": ["ahead"]}
    # A different final answer (the check sent it back) is written anew.
    system._start_followups("t2", dict(values))
    system._followup_jobs["t2"][1].result(timeout=5)
    system.agent_graph = Graph({"question": "粒径？", "answer": "改写后的回答"})
    assert system.followups("t2", "粒径？") == {"items": ["now"]}


# --- a request with no first token is sent again alongside -----------------------------


def _fake_streams(monkeypatch, plans):
    """``plans``: per call, (seconds before the first chunk, text or an exception)."""
    import time as clock

    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_openai import ChatOpenAI

    calls = []

    def stream(self, *args, **kwargs):
        delay, text = plans[len(calls)]
        calls.append(kwargs.get("run_manager"))
        clock.sleep(delay)
        if isinstance(text, BaseException):
            raise text
        yield ChatGenerationChunk(message=AIMessageChunk(content=""))
        for char in text:
            yield ChatGenerationChunk(message=AIMessageChunk(content=char))

    monkeypatch.setattr(ChatOpenAI, "_stream", stream)
    return calls


def _chat(hedge_after):
    from services.model_clients import SiliconFlowChat

    return SiliconFlowChat(model="m", api_key="sk-test-123456", base_url="http://x", streaming=True, hedge_after=hedge_after)


def test_a_stuck_request_is_overtaken_by_a_second_one(monkeypatch):
    calls = _fake_streams(monkeypatch, [(3.0, "迟到的回答"), (0.0, "答案")])
    started = __import__("time").monotonic()
    assert _chat(0.2).invoke("hi").content == "答案"
    assert __import__("time").monotonic() - started < 2.0
    assert len(calls) == 2 and calls == [None, None]  # callbacks only see the chunks used


def test_a_quick_request_is_never_sent_twice(monkeypatch):
    calls = _fake_streams(monkeypatch, [(0.0, "答案"), (0.0, "不该发出")])
    assert _chat(5.0).invoke("hi").content == "答案"
    assert len(calls) == 1


def test_an_error_before_the_hedge_is_raised_as_before(monkeypatch):
    _fake_streams(monkeypatch, [(0.0, RuntimeError("服务出错"))])
    with pytest.raises(RuntimeError, match="服务出错"):
        _chat(5.0).invoke("hi")


def test_the_second_request_answers_when_the_first_fails_late(monkeypatch):
    _fake_streams(monkeypatch, [(0.6, RuntimeError("超时")), (0.0, "备用的回答")])
    assert _chat(0.2).invoke("hi").content == "备用的回答"


def test_typing_asks_the_backend_to_reopen_its_connections(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from api import main
    from config import Settings

    calls = []
    monkeypatch.setattr(main, "system", type("System", (), {"warm_up_soon": lambda self: calls.append(1)})())
    monkeypatch.setattr(main, "settings", Settings(_env_file=None, data_dir=tmp_path))
    assert TestClient(main.app).post("/api/v1/warm").json() == {"ok": True}
    assert calls == [1]


def test_warm_up_reopens_connections_at_most_every_20_seconds(monkeypatch):
    import core.rag_system as rag_system

    system = rag_system.RAGSystem.__new__(rag_system.RAGSystem)
    system._warm_lock = threading.Lock()
    system._warmed_at = float("-inf")
    system.settings = object()
    opened = []
    monkeypatch.setattr(rag_system, "open_connections", lambda settings: opened.append(1) or 2)
    assert "connections" in system.warm_up(connections_only=True)
    assert system.warm_up(connections_only=True) == {}  # within 20 s: nothing
    assert opened == [1]
