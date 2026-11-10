from types import SimpleNamespace

from core.rag_system import RAGSystem


class StreamSettings:
    sse_chunk_size = 2


class FakeGraph:
    def stream(self, state, config, stream_mode, durability=None):
        yield {"analyze_query": {"intent": "knowledge", "rewritten_query": "制度"}}

    def get_state(self, config):
        return SimpleNamespace(
            values={
                "answer": "完整答案",
                "intent": "knowledge",
                "rewritten_query": "制度",
                "citations": [],
                "grounded": True,
                "refused": False,
                "retrieval_attempts": 1,
            }
        )


def test_stream_emits_verified_answer_chunks():
    system = RAGSystem(StreamSettings())
    system.initialized = True
    system._begin_turn = lambda thread_id: None
    system.agent_graph = FakeGraph()

    events = list(system.stream("问题", "thread-1"))
    assert "".join(event["content"] for event in events if event["event"] == "token") == "完整答案"
    assert events[-1]["event"] == "final"
    assert events[-1]["refused"] is False


class DraftSettings:
    sse_chunk_size = 2
    stream_draft = True


class DraftGraph(FakeGraph):
    def __init__(self, answer="完整答案"):
        self.answer = answer

    def stream(self, state, config, stream_mode, durability=None):
        assert stream_mode == ["updates", "messages", "custom"]
        yield "updates", {"analyze_query": {"intent": "knowledge", "rewritten_query": "制度"}}
        # Tokens from other nodes (planner, fact check) must not leak into the draft.
        yield "messages", (SimpleNamespace(content='{"grounded"'), {"langgraph_node": "fact_check"})
        yield "messages", (SimpleNamespace(content="完整"), {"langgraph_node": "generate_answer"})
        yield "messages", (SimpleNamespace(content="答案"), {"langgraph_node": "generate_answer"})
        yield "updates", {"generate_answer": {"answer": "完整答案"}}
        yield "updates", {"fact_check": {"grounded": True}}

    def get_state(self, config):
        values = super().get_state(config).values
        return SimpleNamespace(values={**values, "answer": self.answer})


def test_stream_forwards_answer_draft_then_final_without_duplicate_tokens():
    system = RAGSystem(DraftSettings())
    system.initialized = True
    system._begin_turn = lambda thread_id: None
    system.agent_graph = DraftGraph()

    events = list(system.stream("问题", "thread-1"))
    assert "".join(e["content"] for e in events if e["event"] == "draft") == "完整答案"
    assert not [e for e in events if e["event"] == "token"]
    assert [e["node"] for e in events if e["event"] == "node"] == ["analyze_query", "generate_answer", "fact_check"]
    assert events[-1]["event"] == "final" and events[-1]["answer"] == "完整答案"


def test_final_answer_overrides_a_rejected_draft():
    system = RAGSystem(DraftSettings())
    system.initialized = True
    system._begin_turn = lambda thread_id: None
    system.agent_graph = DraftGraph(answer="当前知识库没有足够依据回答这个问题。")

    events = list(system.stream("问题", "thread-1"))
    assert events[-1]["answer"].startswith("当前知识库没有足够依据")
