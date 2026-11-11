"""Asking again after a failure: what failed (error kinds), the turn's waiting budget, the busy
pause for background work, "重新提问" without the question twice in the history, and what the
stream tells the UI."""

import json
import time
from types import SimpleNamespace

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import openai
import pytest

from api.main import _sse_response
from core.rag_system import RAGSystem
from rag_agent.graph import create_agent_graph
from rag_agent.nodes import AgentNodes
from services.model_clients import (
    AUTO_RETRY_KINDS,
    MANUAL_RETRY_KINDS,
    ServiceBusy,
    chat_busy,
    error_kind,
    retry_after,
    within_budget,
)
from tests.test_graph_integration import FakeParentStore, FakeRetriever, GraphSettings
from tests.test_model_failures import FailingLLM, rate_limit_error, run

REQUEST = httpx.Request("POST", "https://api.siliconflow.cn/v1/chat/completions")
ANSWER = "基于用户知识库文档回答问题"


def status(code, message="x", body=None, headers=None):
    response = httpx.Response(code, request=REQUEST, headers=headers or {})
    kind = {429: openai.RateLimitError, 401: openai.AuthenticationError, 403: openai.PermissionDeniedError}
    return kind.get(code, openai.APIStatusError)(message, response=response, body=body)


def too_busy():
    return status(429, "Error code: 429 - System is too busy now.", body={"code": 50609})


# ---------------------------------------------------------------- what failed

def test_error_kinds_tell_what_to_do_next():
    assert error_kind(too_busy()) == "busy"
    assert error_kind(status(429, "TPM limit reached")) == "rate_limit"
    assert error_kind(status(401)) == error_kind(status(403)) == "auth"
    assert error_kind(status(402)) == "balance"
    assert error_kind(status(503)) == "server"
    assert error_kind(status(400)) == "bad_request"
    assert error_kind(openai.APITimeoutError(request=REQUEST)) == "network"
    assert error_kind(httpx.RemoteProtocolError("peer closed connection")) == "network"
    assert error_kind(RuntimeError("database is locked")) == "unknown"
    # Asking again may help after anything but a bad key, no balance or a bad request.
    assert MANUAL_RETRY_KINDS == {"busy", "rate_limit", "network", "server", "unknown"}
    assert "unknown" not in AUTO_RETRY_KINDS


def test_retry_after_is_honoured_up_to_a_minute():
    assert retry_after(status(429, headers={"Retry-After": "7"})) == 7
    assert retry_after(status(503, headers={"Retry-After": "600"})) == 60
    assert retry_after(status(429)) is None
    assert retry_after(RuntimeError()) is None


# ---------------------------------------------------------------- the busy pause

def test_busy_pause_waits_only_for_that_model_and_never_past_its_limit():
    now, slept = [100.0], []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    busy = ServiceBusy(clock=lambda: now[0], sleep=sleep)
    assert busy.left("qwen") == 0 and busy.wait("qwen") == 0
    busy.note("qwen", 5)
    assert busy.left("qwen") == 5 and busy.left("glm") == 0
    assert busy.wait("qwen") == 5 and sum(slept) == 5
    busy.note("qwen", 500)
    assert busy.wait("qwen", limit=3) == 3


def test_a_busy_answer_makes_background_work_step_aside(monkeypatch):
    monkeypatch.setattr(AgentNodes, "rate_limit_backoff", (0.0,))
    settings = GraphSettings()
    settings.llm_model = "Qwen/Qwen3-8B"
    graph = create_agent_graph(llm=FailingLLM(ANSWER, too_busy, times=1), retriever=FakeRetriever(),
                               settings=settings, parent_store=FakeParentStore())
    result = graph.invoke({"messages": [HumanMessage(content="住宿费超标怎么办？")]},
                          config={"configurable": {"thread_id": "t"}})
    assert result["grounded"] is True  # the retry answered
    assert chat_busy.left("Qwen/Qwen3-8B") > 0


def test_follow_ups_use_templates_while_the_model_is_busy():
    from suggestions.followups import FollowupService

    chat_busy.note("Qwen/Qwen3-8B", 30)
    service = FollowupService.__new__(FollowupService)
    service.chat_model = lambda: SimpleNamespace(model_name="Qwen/Qwen3-8B")
    with pytest.raises(RuntimeError, match="busy"):
        service._call("问题", None)


# ---------------------------------------------------------------- the turn's budget

def test_the_turn_budget_ends_waiting_and_says_why(monkeypatch):
    # Patience alone would wait out these rate limits; the turn's budget is already spent.
    monkeypatch.setattr(AgentNodes, "rate_limit_backoff", (0.5, 0.5, 0.5))
    llm = FailingLLM(ANSWER, rate_limit_error, times=99)

    def turn():
        yield run(llm)

    started = time.monotonic()
    result = next(within_budget(turn(), 0.1))
    assert time.monotonic() - started < 0.5
    assert result["refused"] is True and result["error_kind"] == "rate_limit"
    assert sum(call.startswith("你是基于用户知识库") for call in llm.calls) == 1


def test_without_a_budget_the_usual_retries_still_answer(monkeypatch):
    monkeypatch.setattr(AgentNodes, "rate_limit_backoff", (0.0,))
    llm = FailingLLM(ANSWER, rate_limit_error, times=1)
    assert run(llm)["grounded"] is True


# ---------------------------------------------------------------- "重新提问"

def failing_then_answering():
    llm = FailingLLM(ANSWER, lambda: httpx.RemoteProtocolError("peer closed connection"), times=1)
    graph = create_agent_graph(llm=llm, retriever=FakeRetriever(), settings=GraphSettings(),
                               parent_store=FakeParentStore(), checkpointer=MemorySaver())
    return graph, {"configurable": {"thread_id": "t"}}


def ask(graph, config, question):
    return graph.invoke({"messages": [HumanMessage(content=question)], "question": question}, config=config)


def test_asking_again_replaces_the_failed_turn_instead_of_repeating_it():
    graph, config = failing_then_answering()
    failed = ask(graph, config, "住宿费超标怎么办？")
    assert failed["refused"] is True and failed["error_kind"] == "network"
    system = SimpleNamespace(agent_graph=graph)
    assert RAGSystem._drop_failed_turn(system, config, "住宿费超标怎么办？") == 2
    answered = ask(graph, config, "住宿费超标怎么办？")
    humans = [m.content for m in answered["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["住宿费超标怎么办？"]
    assert answered["grounded"] is True and not answered["error_kind"]


def test_asking_again_never_drops_a_turn_that_succeeded():
    graph, config = failing_then_answering()
    graph.update_state(config, {"messages": [HumanMessage(content="住宿费超标怎么办？"), AIMessage(content="要说明原因")]},
                       as_node="finish")
    system = SimpleNamespace(agent_graph=graph)
    # Nothing failed in the saved state (a crashed turn is not saved): keep the history.
    assert RAGSystem._drop_failed_turn(system, config, "住宿费超标怎么办？") == 0
    # A failed turn about another question is not this one.
    ask(graph, config, "出差补贴多少？")
    assert RAGSystem._drop_failed_turn(system, config, "住宿费超标怎么办？") == 0


# ---------------------------------------------------------------- what the stream says

async def events(response):
    body = b""
    async for chunk in response.body_iterator:
        body += chunk.encode() if isinstance(chunk, str) else chunk
    return [json.loads(line[5:]) for line in body.decode().split("\n") if line.startswith("data:")]


class RecordingRag:
    def __init__(self, error=None):
        self.error, self.options = error, None

    def stream(self, question, thread_id, **options):
        self.options = options
        if self.error:
            raise self.error
        yield {"event": "final", "answer": "好"}


@pytest.mark.asyncio
async def test_the_stream_says_what_failed_and_whether_asking_again_helps():
    error = (await events(_sse_response(RecordingRag(RuntimeError("database is locked")), "问题", "t")))[-1]
    assert error["event"] == "error" and error["kind"] == "unknown" and error["retryable"] is True
    error = (await events(_sse_response(RecordingRag(status(402)), "问题", "t")))[-1]
    assert error["kind"] == "balance" and error["retryable"] is False


@pytest.mark.asyncio
async def test_a_retry_request_reaches_the_answering_system():
    rag = RecordingRag()
    await events(_sse_response(rag, "问题", "t", retry=True))
    assert rag.options.get("retry") is True
    rag = RecordingRag()
    await events(_sse_response(rag, "问题", "t"))
    assert "retry" not in rag.options
