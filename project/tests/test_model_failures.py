"""Model-service failures (rate limits, dropped connections) must not look like bad evidence."""

import httpx
import openai
from langchain_core.messages import HumanMessage

from rag_agent.graph import create_agent_graph
from rag_agent.nodes import AgentNodes
from services.model_clients import describe_model_error
from tests.test_graph_integration import FakeLLM, FakeParentStore, FakeRetriever, GraphSettings


def rate_limit_error():
    request = httpx.Request("POST", "https://api.siliconflow.cn/v1/chat/completions")
    return openai.RateLimitError("TPM limit reached", response=httpx.Response(429, request=request), body=None)


class FailingLLM(FakeLLM):
    """Fails the call whose system prompt contains ``fail_on``, ``times`` times."""

    def __init__(self, fail_on, error, times=99):
        super().__init__(grounded=True)
        self.fail_on, self.error, self.remaining = fail_on, error, times
        self.calls = []

    def invoke(self, messages):
        system = str(messages[0].content)
        self.calls.append(system[:12])
        if self.fail_on in system and self.remaining:
            self.remaining -= 1
            raise self.error()
        return super().invoke(messages)


def run(llm):
    graph = create_agent_graph(
        llm=llm,
        retriever=FakeRetriever(),
        settings=GraphSettings(),
        parent_store=FakeParentStore(),
    )
    return graph.invoke(
        {"messages": [HumanMessage(content="住宿费超标怎么办？")]},
        config={"configurable": {"thread_id": "t"}},
    )


def no_wait(monkeypatch):
    monkeypatch.setattr(AgentNodes, "rate_limit_backoff", (0.0,))


def test_unreachable_fact_checker_shows_answer_unverified_without_second_search(monkeypatch):
    no_wait(monkeypatch)
    result = run(FailingLLM("事实一致性检查器", rate_limit_error))

    assert result["check_skipped"] is True
    assert result["grounded"] is False and not result.get("refused")
    assert result["retrieval_attempts"] == 1
    assert "部门负责人审批" in result["answer"]


def test_dropped_generation_stream_refuses_with_the_real_reason():
    def dropped():
        return httpx.RemoteProtocolError("peer closed connection without sending complete message body")

    result = run(FailingLLM("基于用户知识库文档回答问题", dropped))

    assert result["refused"] is True
    assert result["retrieval_attempts"] == 1
    assert "连接中断" in result["answer"]


def test_rate_limit_is_retried_before_giving_up(monkeypatch):
    no_wait(monkeypatch)
    llm = FailingLLM("基于用户知识库文档回答问题", rate_limit_error, times=1)

    result = run(llm)

    assert result["grounded"] is True
    assert sum(call.startswith("你是基于用户知识库") for call in llm.calls) == 2


def test_error_messages_name_the_cause():
    assert "限流" in describe_model_error(rate_limit_error())
    request = httpx.Request("POST", "https://api.siliconflow.cn/v1/chat/completions")
    busy = openai.RateLimitError("Error code: 429 - System is too busy now.", response=httpx.Response(429, request=request),
                                 body={"code": 50609, "message": "Request was rejected due to rate limiting. Details: System is too busy now."})
    assert "繁忙" in describe_model_error(busy) and "额度" in describe_model_error(busy)
    assert "连接中断" in describe_model_error(httpx.RemoteProtocolError("closed"))
    assert "暂时不可用" in describe_model_error(RuntimeError("boom"))
