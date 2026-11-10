from types import SimpleNamespace

from pydantic import ValidationError
import pytest
from langchain_core.messages import AIMessage, HumanMessage


from rag_agent.edges import route_after_analysis
from rag_agent.nodes import AgentNodes
from rag_agent.schemas import QueryPlan


def test_refusal_is_not_reported_as_grounded():
    result = AgentNodes(None, None).refuse({"retrieval_attempts": 2})

    assert result["refused"] is True
    assert result["grounded"] is False


class ContextSettings:
    context_token_budget = 100


class FakeParentStore:
    def get(self, parent_id):
        return {
            "parent_id": parent_id,
            "doc_id": "a" * 24,
            "source": "制度.txt",
            "heading": "报销制度",
            "locator": "section:1",
            "text": "这是父块完整正文，包含比命中子块更多的上下文。",
        }


def test_context_build_expands_and_deduplicates_parents():
    nodes = AgentNodes(None, None, ContextSettings(), FakeParentStore())
    child = {
        "id": "c1",
        "parent_id": f"{'a' * 24}-p0",
        "source": "制度.txt",
        "text": "命中子块",
        "rerank_score": 0.9,
    }
    result = nodes.build_context({"retrieve_chunks": [child, {**child, "id": "c2"}]})

    assert len(result["context_docs"]) == 1
    assert result["context_docs"][0]["text"].startswith("这是父块完整正文")
    assert result["context_docs"][0]["child_text"] == "命中子块"


def test_plan_policy_drops_clarification_without_a_reference():
    plan = QueryPlan(
        intent="knowledge",
        rewritten_query="西南区备用金核销 提前 工作日",
        needs_clarification=True,
        clarification_question="请补充产品",
    )

    fixed = AgentNodes._enforce_plan_policy("办理西南区备用金核销最晚需提前几个工作日提交材料？", plan)

    assert fixed.intent == "knowledge"
    assert fixed.needs_clarification is False


def test_plan_policy_preserves_real_ambiguity():
    ambiguous = QueryPlan(
        intent="knowledge",
        rewritten_query="该产品保修期",
        needs_clarification=True,
        clarification_question="请问是哪款产品？",
    )

    assert AgentNodes._enforce_plan_policy("该产品的保修期呢？", ambiguous) == ambiguous


@pytest.mark.parametrize("intent", ["structured_data", "business_api"])
def test_planner_can_no_longer_choose_the_demo_tools(intent):
    with pytest.raises(ValidationError):
        QueryPlan(intent=intent, rewritten_query="NX-MEET-PRO 价格")


class CountingPlanner:
    def __init__(self, reply='{"intent":"knowledge","rewritten_query":"改写后的问题"}'):
        self.reply = reply
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.reply)


@pytest.mark.parametrize(
    "question",
    ["人事部门的联系电话是多少？", "纳米硅产品的价格是多少？", "你们产品的质保期是多久？", "实时库存还有多少件？", "服务状态是否在线？"],
)
def test_questions_the_demo_tools_took_are_searched_in_the_documents(question):
    llm = CountingPlanner()
    update = AgentNodes(llm, None).analyze_query({"messages": [HumanMessage(content=question)]})

    assert update["intent"] == "knowledge"
    assert route_after_analysis(update) == "retrieve"
    # Nothing in the wording needs the planner any more: searched as asked.
    assert llm.calls == 0 and update["rewritten_query"] == question


def test_a_planner_reply_naming_a_removed_intent_falls_back_to_searching():
    llm = CountingPlanner('{"intent":"structured_data","rewritten_query":"纳米硅 价格","tool_name":"product"}')
    update = AgentNodes(llm, None).analyze_query(
        {"messages": [HumanMessage(content="它的价格呢？")]}
    )

    assert update["intent"] == "knowledge"
    assert update["rewritten_query"] == "它的价格呢？"


def test_a_retry_answers_from_the_new_passages_not_the_first_ones():
    class RecordingLLM:
        def invoke(self, messages):
            self.prompt = str(messages[-1].content)
            return AIMessage(content="回答 [来源：新.pdf｜位置：page:2]")

    llm = RecordingLLM()
    nodes = AgentNodes(llm, retriever=None, settings=SimpleNamespace(context_token_budget=500))
    result = nodes.generate_answer({
        "question": "问题",
        "evidence": "[证据 1｜来源：旧.pdf｜位置：page:1]\n旧的段落",
        "citations": [{"source": "旧.pdf", "locator": "page:1"}],
        "context_docs": [{"source": "新.pdf", "locator": "page:2", "text": "新的段落"}],
    })

    assert "新的段落" in llm.prompt and "旧的段落" not in llm.prompt
    assert [c["source"] for c in result["citations"]] == ["新.pdf"]


@pytest.mark.parametrize("question", ["agent是什么", "能再详细展开讲讲吗", "第二点是什么意思？", "hi, what is ReAct"])
def test_plan_policy_searches_questions_the_planner_mislabels_as_chitchat(question):
    plan = QueryPlan(intent="chitchat", rewritten_query=question)

    assert AgentNodes._enforce_plan_policy(question, plan).intent == "knowledge"


@pytest.mark.parametrize("question", ["你好", "谢谢！", "你能做什么？", "Hello", "再见啦"])
def test_plan_policy_keeps_plain_small_talk(question):
    plan = QueryPlan(intent="chitchat", rewritten_query=question)

    assert AgentNodes._enforce_plan_policy(question, plan).intent == "chitchat"


def test_generate_answer_passes_recent_conversation_for_follow_ups():
    class RecordingLLM:
        def invoke(self, messages):
            self.prompt = str(messages[-1].content)
            return AIMessage(content="ReAct 是推理与行动交替的模式 [来源：a.pdf｜位置：page:1]")

    llm = RecordingLLM()
    nodes = AgentNodes(llm, retriever=None, settings=SimpleNamespace(context_token_budget=500))
    nodes.generate_answer({
        "question": "能再详细展开讲讲吗",
        "messages": [
            HumanMessage(content="ReAct 是什么"),
            AIMessage(content="ReAct 是一种 Agent 推理模式"),
            HumanMessage(content="能再详细展开讲讲吗"),
        ],
        "context_docs": [{"source": "a.pdf", "locator": "page:1", "text": "ReAct 交替进行思考、行动和观察。"}],
    })

    assert "最近对话" in llm.prompt
    assert "ReAct 是一种 Agent 推理模式" in llm.prompt
