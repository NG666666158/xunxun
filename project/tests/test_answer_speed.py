"""Answer profiles, the cited-statement fact check, fast mode, live API keys, error details."""

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
import pytest

from api import main
from api.errors import ErrorDetailMiddleware
from config import MissingApiKeyError, Settings
from core import user_settings
from rag_agent.answering import PROFILES, cited_claims, classify_question, evidence_for
from rag_agent.edges import route_after_fact_check
from rag_agent.nodes import AgentNodes
from services.model_clients import SiliconFlowChat, SiliconFlowEmbeddings, SiliconFlowReranker, describe_model_error


# --- question types ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "profile"),
    [
        ("在斯坦福AI小镇的实验中，Isabella 想在什么时间举办派对？", "fact"),
        ("知识更新的流水线应该明确分开哪三层？", "fact"),
        ("什么是提示注入？为什么它对Agent系统特别危险？", "explain"),
        ("在微调BERT模型时，DataCollatorWithPadding工具的主要作用是什么？", "explain"),
        ("关于RAG技术，这份面试题主要考察了哪些方面的知识点？", "explain"),
        ("ORM 和 PRM 有什么区别？", "compare"),
        ("《AI Agent 实战》和《图解大模型》对RAG技术的描述侧重点有何不同？", "compare"),
    ],
)
def test_questions_are_classified_by_wording(question, profile):
    assert classify_question(question) == profile


def test_fact_questions_get_the_smallest_budget_and_shortest_answers():
    assert PROFILES["fact"].context_budget < PROFILES["explain"].context_budget <= PROFILES["compare"].context_budget
    assert PROFILES["fact"].max_tokens < PROFILES["explain"].max_tokens < PROFILES["compare"].max_tokens


class BudgetSettings:
    context_token_budget = 9000


class SizedParents:
    def get(self, parent_id):
        return {"parent_id": parent_id, "source": "a.pdf", "locator": parent_id, "text": "字" * 1500}


def test_context_budget_follows_the_answer_profile():
    nodes = AgentNodes(None, None, BudgetSettings(), SizedParents())
    children = [{"id": str(i), "parent_id": f"p{i}", "text": "x", "source": "a.pdf"} for i in range(6)]
    fact = nodes.build_context({"retrieve_chunks": children, "answer_profile": "fact"})["context_docs"]
    compare = nodes.build_context({"retrieve_chunks": children, "answer_profile": "compare"})["context_docs"]
    assert len(fact) == 2 and len(compare) == 6


# --- planner skip --------------------------------------------------------------------


class CountingPlanner:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content='{"intent":"knowledge","rewritten_query":"ReAct 的原理和示例"}')


def test_first_turn_knowledge_question_skips_the_planner():
    llm = CountingPlanner()
    update = AgentNodes(llm, None).analyze_query({"messages": [HumanMessage(content="什么是 ReAct？")]})
    assert llm.calls == 0
    assert update["intent"] == "knowledge" and update["rewritten_query"] == "什么是 ReAct？"
    assert update["answer_profile"] == "explain"


@pytest.mark.parametrize(
    "messages",
    [
        [HumanMessage(content="什么是 ReAct？"), AIMessage(content="ReAct 是……"), HumanMessage(content="展开讲讲")],
        [HumanMessage(content="它的质保期多久？")],
    ],
)
def test_follow_ups_still_use_the_planner(messages):
    llm = CountingPlanner()
    update = AgentNodes(llm, None).analyze_query({"messages": messages})
    assert llm.calls == 1
    if len(messages) == 3:
        # "展开讲讲" takes the profile of what it expands into.
        assert update["answer_profile"] == "explain"


# --- citations -----------------------------------------------------------------------


ANSWER = """ReAct 把推理和行动交替进行。

1. **推理**：模型先写出思考 [来源：AI-Agents.pdf｜位置：page:12]。
2. 行动：调用工具并读取观察结果 [来源：AI-Agents.pdf｜位置：page:13][来源：图解大模型.pdf｜位置：page:5]
知识库中没有找到关于性能对比的依据。"""


def test_cited_claims_take_the_text_before_each_citation():
    claims = cited_claims(ANSWER)
    assert claims == [
        ("推理：模型先写出思考", [("AI-Agents.pdf", "page:12")]),
        ("行动：调用工具并读取观察结果", [("AI-Agents.pdf", "page:13"), ("图解大模型.pdf", "page:5")]),
    ]


BLOCKS = [
    {"source": "AI-Agents.pdf", "locator": "page:12", "text": "模型先写出思考，再决定行动。"},
    {"source": "AI-Agents.pdf", "locator": "page:13", "text": "行动之后读取观察结果。"},
    {"source": "图解大模型.pdf", "locator": "page:5", "text": "工具调用。"},
    {"source": "其他.pdf", "locator": "page:1", "text": "无关内容。"},
]


def test_citations_resolve_to_their_evidence_blocks():
    assert evidence_for([("AI-Agents.pdf", "page:13")], BLOCKS) == ([1], True)
    # Unknown locator: the source's passages are used.
    assert evidence_for([("AI-Agents.pdf", "page:99")], BLOCKS) == ([0, 1], True)
    assert evidence_for([("差旅制度.docx", "section:2")], BLOCKS) == ([], False)
    # "[来源：a.pdf，page:12]" without the 位置 label.
    assert evidence_for([("AI-Agents.pdf，page:12", "")], BLOCKS) == ([0], True)


class CheckerLLM:
    def __init__(self, grounded=True):
        self.prompts = []
        self.grounded = grounded

    def invoke(self, messages):
        self.prompts.append((str(messages[0].content), str(messages[1].content)))
        return AIMessage(content=json.dumps({"grounded": self.grounded, "unsupported_claims": []}))


class ModeSettings:
    def __init__(self, mode):
        self.answer_mode = mode
        self.max_retrieval_attempts = 2


def check_state(answer=ANSWER):
    return {"answer": answer, "context_docs": BLOCKS, "evidence": "全部证据"}


def test_strict_check_sends_only_cited_statements_and_their_passages():
    llm = CheckerLLM()
    result = AgentNodes(llm, None, ModeSettings("strict")).fact_check(check_state())
    assert result == {"grounded": True, "unsupported_claims": [], "check_skipped": False, "check_mode": "cited"}
    system, content = llm.prompts[0]
    assert "引用的证据编号" in system
    assert "无关内容" not in content  # the uncited passage is not sent
    assert "没有找到关于性能对比" not in content  # nor the uncited sentence
    assert "1. 推理：模型先写出思考（依据：E1）" in content
    assert "2. 行动：调用工具并读取观察结果（依据：E2、E3）" in content


def test_strict_check_without_citations_checks_the_whole_answer():
    llm = CheckerLLM()
    result = AgentNodes(llm, None, ModeSettings("strict")).fact_check(check_state("当前知识库没有足够依据。"))
    assert result["check_mode"] == "full" and "全部证据" in llm.prompts[0][1]


def test_strict_check_fails_a_citation_to_missing_evidence():
    answer = "报销需要审批 [来源：差旅制度.docx｜位置：section:2]。"
    result = AgentNodes(CheckerLLM(), None, {}, ModeSettings("strict")).fact_check(check_state(answer))
    assert result["grounded"] is False
    assert any("不在检索证据中" in claim for claim in result["unsupported_claims"])


def test_fast_mode_checks_citations_without_a_model_call():
    llm = CheckerLLM()
    nodes = AgentNodes(llm, None, ModeSettings("fast"))
    good = nodes.fact_check(check_state())
    assert good["grounded"] is True and good["check_mode"] == "citations" and llm.prompts == []
    bad = nodes.fact_check(check_state("没有引用的回答。"))
    assert bad["grounded"] is False
    # Fast mode never searches again, whatever the citations say.
    assert route_after_fact_check({**bad, "retrieval_attempts": 1}, ModeSettings("fast")) == "finish"


class BindingLLM:
    def __init__(self):
        self.bound = None

    def bind(self, **kwargs):
        self.bound = kwargs
        return self

    def invoke(self, messages):
        self.prompt = str(messages[1].content)
        return AIMessage(content="答案 [来源：a.pdf｜位置：page:1]")


def test_answer_length_follows_the_profile():
    llm = BindingLLM()
    state = {"question": "什么时候举办？", "answer_profile": "fact",
             "context_docs": [{"source": "a.pdf", "locator": "page:1", "text": "周五"}]}
    AgentNodes(llm, None).generate_answer(state)
    assert llm.bound == {"max_tokens": PROFILES["fact"].max_tokens}
    assert "250 字以内" in llm.prompt


# --- streamed token usage -----------------------------------------------------------


def test_streamed_usage_is_counted_once(monkeypatch):
    def cumulative(self, *args, **kwargs):
        for n in range(1, 4):
            usage = {"input_tokens": 20, "output_tokens": n, "total_tokens": 20 + n}
            yield ChatGenerationChunk(message=AIMessageChunk(content="字", usage_metadata=usage))

    monkeypatch.setattr(ChatOpenAI, "_stream", cumulative)
    llm = SiliconFlowChat(model="m", api_key="sk-test-123456", base_url="http://x", streaming=True)
    result = llm.invoke("hi")
    assert result.content == "字字字"
    assert result.usage_metadata["input_tokens"] == 20 and result.usage_metadata["output_tokens"] == 3


# --- API key read per request -------------------------------------------------------


def test_clients_use_the_key_current_at_request_time(tmp_path):
    seen = []

    def handler(request):
        seen.append(request.headers["authorization"])
        if request.url.path.endswith("/rerank"):
            return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 4}], "model": "m",
                                         "object": "list", "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    settings = Settings(_env_file=None, data_dir=tmp_path, siliconflow_api_key="sk-first-123456",
                        siliconflow_base_url="http://sf.test/v1", embedding_dimension=4)
    reranker = SiliconFlowReranker(settings)
    # The reranker sends its own auth on each request through the shared pool.
    reranker.client = httpx.Client(transport=httpx.MockTransport(handler))
    embeddings = SiliconFlowEmbeddings(settings)
    embeddings.client = embeddings.client.with_options(http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    reranker.rerank("q", ["d"], top_n=1)
    embeddings.embed_query("q")
    settings.siliconflow_api_key = "sk-second-123456"
    reranker.rerank("q", ["d"], top_n=1)
    embeddings.embed_query("q")
    assert seen == ["Bearer sk-first-123456"] * 2 + ["Bearer sk-second-123456"] * 2

    settings.siliconflow_api_key = ""
    with pytest.raises(MissingApiKeyError):
        reranker.rerank("q", ["d"], top_n=1)
    with pytest.raises(MissingApiKeyError):
        embeddings.embed_query("q")
    assert "设置" in describe_model_error(MissingApiKeyError("缺少 SILICONFLOW_API_KEY。请在“设置”页填写"))


def test_answer_mode_and_tray_choice_are_saved(tmp_path, monkeypatch):
    class System:
        initialized = True

        def reload_models(self):
            pass

    settings = Settings(_env_file=None, data_dir=tmp_path, siliconflow_api_key="sk-test-1234567890")
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "system", System())
    client = TestClient(main.app)

    before = client.get("/api/v1/settings").json()
    assert before["answer_mode"] == "strict" and before["close_to_tray"] is True
    body = client.put("/api/v1/settings", json={"answer_mode": "fast", "close_to_tray": False}).json()
    assert body["answer_mode"] == "fast" and body["close_to_tray"] is False
    assert settings.answer_mode == "fast"  # the graph reads it on the next turn
    assert client.put("/api/v1/settings", json={"answer_mode": "slow"}).status_code == 422
    assert user_settings.apply_user_settings(Settings(_env_file=None, data_dir=tmp_path)).answer_mode == "fast"


def test_missing_key_is_named_instead_of_a_generic_503(tmp_path, monkeypatch):
    from core.rag_system import RAGSystem

    settings = Settings(_env_file=None, data_dir=tmp_path, siliconflow_api_key="")
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "system", RAGSystem(settings))
    response = TestClient(main.app).post("/api/v1/chat", json={"question": "什么是 RAG？"})
    assert response.status_code == 503
    assert "API_KEY" in response.json()["detail"]


# --- error details reach the webview ------------------------------------------------


def test_unhandled_errors_answer_with_reason_and_cors_headers():
    app = FastAPI()
    app.add_middleware(ErrorDetailMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["http://tauri.localhost"], allow_methods=["*"])

    @app.get("/boom")
    def boom():
        raise ValueError("索引文件损坏")

    @app.get("/locked")
    def locked():
        raise PermissionError(13, "denied", "C:/kb/knowledge.sqlite3")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom", headers={"Origin": "http://tauri.localhost"})
    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == "http://tauri.localhost"
    assert "ValueError" in response.json()["detail"] and "索引文件损坏" in response.json()["detail"]
    assert "knowledge.sqlite3" in client.get("/locked").json()["detail"]


def test_real_app_includes_the_error_middleware():
    assert any(m.cls is ErrorDetailMiddleware for m in main.app.user_middleware)


def test_bare_titles_still_go_through_the_planner():
    # "北方区考勤办法" may name a file rather than ask about its content.
    llm = CountingPlanner()
    AgentNodes(llm, None).analyze_query({"messages": [HumanMessage(content="北方区考勤办法")]})
    assert llm.calls == 1


class SequenceChecker(CheckerLLM):
    """Fails the first (cited) check, then answers the second look with ``second``."""

    def __init__(self, second):
        super().__init__()
        self.second = second

    def invoke(self, messages):
        self.prompts.append((str(messages[0].content), str(messages[1].content)))
        if len(self.prompts) == 1:
            return AIMessage(content=json.dumps({"grounded": False, "unsupported_claims": ["推理：模型先写出思考"]}))
        return AIMessage(content=json.dumps({"grounded": self.second, "unsupported_claims": [] if self.second else ["推理：模型先写出思考"]}))


@pytest.mark.parametrize("second", [True, False])
def test_failed_cited_check_gets_a_second_look_against_all_evidence(second):
    llm = SequenceChecker(second)
    result = AgentNodes(llm, None, ModeSettings("strict")).fact_check(check_state())
    assert len(llm.prompts) == 2
    assert "无关内容" in llm.prompts[1][1]  # the second look sees every passage
    assert "1. 推理：模型先写出思考" in llm.prompts[1][1]
    assert result["grounded"] is second and result["check_mode"] == "cited+all"
