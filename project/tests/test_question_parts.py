"""Questions of several parts get more passages from more documents; no evidence is said plainly.

evaluation/suite_v1/REPORT.md, fixes 2 and 3; the measurements are in
evaluation/suite_v1/FIXES.md.
"""

from types import SimpleNamespace

from langchain_core.messages import HumanMessage
import pytest

from rag_agent.answering import PROFILES, classify_question, multi_part_question, retrieval_limits
from rag_agent.graph import create_agent_graph
from rag_agent.nodes import AgentNodes
from tests.test_graph_integration import FakeLLM, FakeParentStore, GraphSettings


@pytest.mark.parametrize(
    "question",
    [
        "2023世界机器人大会将在哪个地点举办，并且有多少家企业参展？",
        "在2023年8月15日的武装冲突中，有多少人死亡，又有多少人受伤？",
        "生成调度器由哪个团队负责维护，失败后最多重试几次？",
        "纳米硅的试用价格是多少，报价有效期多久？",
        "俄罗斯“月球-25”号探测器的发射时间、预计着陆地点以及未来探月计划的发射年份是什么？",
        "给水方式选择原则?水泵的扬程怎么计算?高层建筑给水系统竖向分区基本形式有哪几种",
        "Who founded the company, and when did it go public?",
        "Which company, covered by TechCrunch, faces an antitrust suit, and is also accused of harming developers?",
        "Does the TechCrunch article say the release was incomplete, while The Verge calls it finished?",
        "What is RAG and how does it work?",
    ],
)
def test_several_asks_in_one_question(question):
    assert multi_part_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "知识更新的流水线应该明确分开哪三层？",
        "4什么加9什么等于1什么",  # two interrogatives, one ask
        "怎么看网卡多少兆",
        "小孩扁桃体发炎怎么办吃什么药",
        "霍山县开展安全生产大检查的目的是什么？",
        "What is the capital of France?",
        "Who is the person who founded OpenAI?",  # a relative "who" is not a second ask
        "How does attention work in transformers?",
        "Is RAG better than fine-tuning for private documents?",
    ],
)
def test_one_ask(question):
    assert not multi_part_question(question)


def test_only_fact_questions_become_multi_part():
    assert classify_question("2023世界机器人大会将在哪个地点举办，并且有多少家企业参展？") == "multi"
    assert classify_question("Who founded the company, and when did it go public?") == "multi"
    # An explanation or a comparison of several parts keeps its own type (as many passages or more).
    assert classify_question("什么是提示注入？为什么它对Agent系统特别危险？") == "explain"
    assert classify_question("在评估RAG效果时，忠实度和答案相关性分别指什么？") == "compare"
    assert classify_question("知识更新的流水线应该明确分开哪三层？") == "fact"


def test_multi_part_questions_keep_ten_passages_two_per_document():
    multi, fact = PROFILES["multi"], PROFILES["fact"]
    assert multi.max_passages == 10 and multi.per_document == 2
    # Room in the context for the extra passages (an explanation's budget).
    assert multi.context_budget == PROFILES["explain"].context_budget > fact.context_budget
    assert retrieval_limits("multi") == {"max_keep": 10, "per_document": 2}
    # Every other type keeps its cap and the setting's per-document cap.
    assert retrieval_limits("fact") == {"max_keep": 6}
    assert retrieval_limits("explain") == {"max_keep": 10}
    assert retrieval_limits("compare") == {"max_keep": 12}


class Recording:
    def __init__(self, hits=None):
        self.calls = []
        self.hits = hits or []

    def retrieve(self, query, mode="hybrid_rerank", **kwargs):
        self.calls.append(kwargs)
        return list(self.hits)


def settings(**extra):
    return SimpleNamespace(relevance_threshold=0.2, max_retrieval_attempts=2, context_token_budget=9000, **extra)


def test_the_retrieve_node_passes_the_multi_part_caps():
    retriever = Recording()
    node = AgentNodes(None, retriever, settings())
    node.retrieve({"rewritten_query": "x", "answer_profile": "multi"})
    node.retrieve({"rewritten_query": "x", "answer_profile": "fact", "scope_doc_ids": ["d1"]})
    assert retriever.calls == [{"max_keep": 10, "per_document": 2}, {"max_keep": 6, "doc_ids": {"d1"}}]


def test_analyze_query_marks_a_multi_part_question():
    class Planner:
        def invoke(self, messages):
            raise AssertionError("a complete first-turn question needs no planner")

    node = AgentNodes(Planner(), Recording(), settings())
    update = node.analyze_query({"messages": [HumanMessage(content="大会在哪个城市举办，并且有多少家企业参展？")]})
    assert update["intent"] == "knowledge" and update["answer_profile"] == "multi"


# ---------------- no evidence ----------------


def test_no_evidence_is_a_plain_reply():
    node = AgentNodes(None, None, settings())
    answer = node.refuse({"documents": []})["answer"]
    assert answer.startswith("资料里没有找到") and "检索" not in answer
    # Passages too weak for the relevance gate are no evidence either.
    weak = [{"rerank_score": 0.15, "text": "t"}]
    assert node.refuse({"documents": weak})["answer"].startswith("资料里没有找到")
    assert "限定的文件范围" in node.refuse({"documents": [], "scope_doc_ids": ["d1"]})["answer"]
    # A table question that fell back to its file's passages names the file.
    table = {"documents": [], "scope_doc_ids": ["d1"], "table_fallback": True, "target_file": {"name": "销量.xlsx"}}
    assert node.refuse(table)["answer"].startswith("《销量》里没有找到")


def test_a_failed_fact_check_is_not_reported_as_nothing_found():
    node = AgentNodes(None, None, settings())
    answer = node.refuse({"documents": [{"rerank_score": 0.9, "text": "t"}]})["answer"]
    assert "没有找到" not in answer and "不足以可靠地回答" in answer
    assert "检索服务暂时不可用" in node.refuse({"documents": [], "last_error": "TimeoutError"})["answer"]


class NoEvidenceRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve(self, query, mode, max_keep=None, **kwargs):
        self.calls += 1
        return []


def test_the_graph_answers_no_evidence_without_writing_an_answer():
    retriever = NoEvidenceRetriever()
    graph = create_agent_graph(llm=FakeLLM(), retriever=retriever, settings=GraphSettings(), parent_store=FakeParentStore())
    result = graph.invoke(
        {"messages": [HumanMessage(content="住宿费超标怎么办？")]}, config={"configurable": {"thread_id": "none"}}
    )
    # One rewrite is still tried; then the plain reply, with no generation and no fact check.
    assert retriever.calls == 2 and result["retrieval_attempts"] == 2
    assert result["refused"] and result["answer"].startswith("资料里没有找到")
    assert not result.get("evidence") and not result.get("citations")
