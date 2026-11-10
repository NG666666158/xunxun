"""Conversations in the data folder: the history, the agent state behind it, and moving them."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sqlite3
import subprocess
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
import pytest

from config import Settings
from core.conversations import ConversationCheckpointer, ConversationStore, state_from_text
from core.rag_system import RAGSystem
from rag_agent.graph import create_agent_graph
from retrieval.types import SearchHit

NODE = shutil.which("node")
JS_TESTS = Path(__file__).resolve().parent / "js" / "convstore.test.cjs"


class GraphSettings:
    relevance_threshold = 0.2
    max_retrieval_attempts = 2
    context_token_budget = 500


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, mode, max_keep=None):
        self.queries.append(query)
        return [
            SearchHit(
                id="child-1", score=0.8, text="纳米硅的平均粒径为 0.8 nm", doc_id="a" * 24, parent_id=f"{'a' * 24}-p0",
                source="产品规格.docx", heading="粒径", locator="section:2", retrieval_method="hybrid", rerank_score=0.9,
            )
        ]


class HistoryAwareLLM:
    """Rewrites "它" only when the planner prompt shows what it refers to."""

    def __init__(self):
        self.plans = []

    def invoke(self, messages):
        system = str(messages[0].content)
        if "查询规划器" in system:
            prompt = str(messages[1].content)
            self.plans.append(prompt)
            if "纳米硅" in prompt.split("当前问题")[0]:
                return AIMessage(content='{"intent":"knowledge","rewritten_query":"纳米硅的粒径"}')
            return AIMessage(content='{"intent":"knowledge","rewritten_query":"粒径","needs_clarification":true,'
                                     '"clarification_question":"请问您指的是哪种物质？"}')
        if "事实一致性检查器" in system:
            return AIMessage(content='{"grounded":true,"unsupported_claims":[]}')
        return AIMessage(content="纳米硅的平均粒径为 0.8 nm [来源：产品规格.docx｜位置：section:2]")


def graph(checkpointer, llm=None):
    return create_agent_graph(llm=llm or HistoryAwareLLM(), retriever=FakeRetriever(), settings=GraphSettings(), checkpointer=checkpointer)


def config(thread_id="t1"):
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}


def rows(db, table, thread_id):
    with sqlite3.connect(db) as connection:
        return connection.execute(f"SELECT COUNT(*) FROM {table} WHERE thread_id = ?", (thread_id,)).fetchone()[0]


# ---------------- the agent state ----------------


def test_a_follow_up_keeps_its_context_across_a_restart(tmp_path):
    db = tmp_path / "conversations.sqlite3"
    before = ConversationCheckpointer(db)
    first = graph(before)
    first.invoke({"messages": [HumanMessage(content="纳米硅是什么？")]}, config=config(), durability="exit")
    before.close()

    # A new process: new saver, new graph, nothing in memory.
    llm = HistoryAwareLLM()
    after = ConversationCheckpointer(db)
    result = graph(after, llm).invoke({"messages": [HumanMessage(content="那它的粒径有多大？")]}, config=config(), durability="exit")

    assert "纳米硅是什么" in llm.plans[0]
    assert result["rewritten_query"] == "纳米硅的粒径"
    assert "0.8 nm" in result["answer"]
    assert [m.type for m in result["messages"]] == ["human", "ai", "human", "ai"]


def test_the_last_turn_can_be_read_after_a_restart(tmp_path):
    db = tmp_path / "conversations.sqlite3"
    before = ConversationCheckpointer(db)
    graph(before).invoke({"messages": [HumanMessage(content="纳米硅是什么？")]}, config=config(), durability="exit")
    before.close()

    values = graph(ConversationCheckpointer(db)).get_state(config()).values

    assert values["question"] == "纳米硅是什么？"
    assert values["context_docs"][0]["source"] == "产品规格.docx"
    assert "0.8 nm" in values["answer"]


def test_only_the_newest_checkpoint_stays(tmp_path):
    db = tmp_path / "conversations.sqlite3"
    agent = graph(ConversationCheckpointer(db))
    for question in ("纳米硅是什么？", "那它的粒径有多大？", "它还有什么特点？"):
        # Also with a checkpoint per step, the default outside the app.
        agent.invoke({"messages": [HumanMessage(content=question)]}, config=config(), durability="sync")

    assert rows(db, "checkpoints", "t1") == 1
    assert len(agent.get_state(config()).values["messages"]) == 6


def test_conversations_do_not_share_state(tmp_path):
    agent = graph(ConversationCheckpointer(tmp_path / "c.sqlite3"))
    agent.invoke({"messages": [HumanMessage(content="纳米硅是什么？")]}, config=config("a"), durability="exit")

    assert agent.get_state(config("b")).values == {}


def test_deleting_a_thread_removes_every_row(tmp_path):
    db = tmp_path / "c.sqlite3"
    saver = ConversationCheckpointer(db)
    agent = graph(saver)
    agent.invoke({"messages": [HumanMessage(content="纳米硅是什么？")]}, config=config(), durability="exit")
    assert rows(db, "checkpoints", "t1") == 1

    saver.delete_thread("t1")

    assert rows(db, "checkpoints", "t1") == 0 and rows(db, "writes", "t1") == 0
    assert agent.get_state(config()).values == {}


def test_an_answer_still_running_cannot_bring_a_deleted_thread_back(tmp_path):
    db = tmp_path / "c.sqlite3"
    saver = ConversationCheckpointer(db)
    agent = graph(saver)
    stream = agent.stream(
        {"messages": [HumanMessage(content="纳米硅是什么？")]}, config=config(), stream_mode="updates", durability="exit"
    )
    next(stream)

    saver.delete_thread("t1")
    list(stream)

    assert rows(db, "checkpoints", "t1") == 0 and rows(db, "writes", "t1") == 0


def test_an_id_used_again_after_deleting_starts_afresh(tmp_path):
    db = tmp_path / "c.sqlite3"
    saver = ConversationCheckpointer(db)
    agent = graph(saver)
    agent.invoke({"messages": [HumanMessage(content="纳米硅是什么？")]}, config=config(), durability="exit")
    saver.delete_thread("t1")

    saver.begin_turn("t1")
    agent.invoke({"messages": [HumanMessage(content="纳米硅是什么？")]}, config=config(), durability="exit")

    assert len(agent.get_state(config()).values["messages"]) == 2


def test_conversations_answer_at_the_same_time(tmp_path):
    db = tmp_path / "c.sqlite3"
    agent = graph(ConversationCheckpointer(db))

    def turn(n):
        thread = config(f"t{n}")
        agent.invoke({"messages": [HumanMessage(content="纳米硅是什么？")]}, config=thread, durability="exit")
        return agent.invoke({"messages": [HumanMessage(content="那它的粒径有多大？")]}, config=thread, durability="exit")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(turn, range(8)))

    assert all(r["rewritten_query"] == "纳米硅的粒径" for r in results)
    assert all(rows(db, "checkpoints", f"t{n}") == 1 for n in range(8))
    assert all(len(agent.get_state(config(f"t{n}")).values["messages"]) == 4 for n in range(8))


def test_a_stopped_answer_does_not_break_the_conversation(tmp_path):
    db = tmp_path / "c.sqlite3"
    agent = graph(ConversationCheckpointer(db))
    agent.invoke({"messages": [HumanMessage(content="纳米硅是什么？")]}, config=config(), durability="exit")
    stream = agent.stream(
        {"messages": [HumanMessage(content="它还有什么特点？")]}, config=config(), stream_mode="updates", durability="exit"
    )
    next(stream)
    stream.close()

    # The app restarts and the user asks again.
    result = graph(ConversationCheckpointer(db)).invoke(
        {"messages": [HumanMessage(content="那它的粒径有多大？")]}, config=config(), durability="exit"
    )

    assert result["rewritten_query"] == "纳米硅的粒径" and "0.8 nm" in result["answer"]
    assert rows(db, "checkpoints", "t1") == 1


def test_a_turn_that_never_finished_is_not_part_of_the_context(tmp_path):
    db = tmp_path / "c.sqlite3"
    saver = ConversationCheckpointer(db)
    agent = graph(saver)
    agent.invoke({"messages": [HumanMessage(content="纳米硅是什么？")]}, config=config(), durability="exit")
    stored = saver.get_tuple(config()).checkpoint["id"]
    stream = agent.stream(
        {"messages": [HumanMessage(content="写到一半进程被杀")]}, config=config(), stream_mode="updates", durability="exit"
    )
    next(stream)

    # Killed here: nothing about this turn has been written yet.
    assert saver.get_tuple(config()).checkpoint["id"] == stored
    stream.close()


# ---------------- a conversation older than the stored state ----------------


def saved_conversation():
    return {
        "id": "old",
        "title": "纳米硅是什么？",
        "updated": 5,
        "scope": None,
        "messages": [
            {"role": "user", "text": "人事部门的联系电话是多少？"},
            {"role": "bot", "id": "b0", "question": "人事部门的联系电话是多少？", "intent": "structured_data", "done": True,
             "answer": "人力资源部联系人为周老师，联系电话是 400-800-1001 [来源：结构化数据库]。",
             "citations": [{"source": "结构化数据库", "locator": ""}]},
            {"role": "user", "text": "纳米硅是什么？"},
            {"role": "bot", "id": "b1", "question": "纳米硅是什么？", "intent": "knowledge", "done": True, "rewritten": "纳米硅",
             "answer": "纳米硅是零维纳米材料 [来源：产品规格.docx｜位置：section:1]", "grounded": True,
             "citations": [{"source": "产品规格.docx", "locator": "section:1", "doc_id": "d" * 24, "quote": "纳米硅是零维纳米材料"}]},
            {"role": "user", "text": "写到一半就关了"},
            {"role": "bot", "id": "b2", "question": "写到一半就关了", "done": False, "answer": "半截"},
        ],
    }


def test_the_state_of_an_old_conversation_comes_from_its_text():
    values = state_from_text(saved_conversation())

    assert [m.content for m in values["messages"] if m.type == "human"] == ["纳米硅是什么？"]
    assert values["question"] == "纳米硅是什么？" and values["intent"] == "knowledge"
    assert values["context_docs"][0]["doc_id"] == "d" * 24
    assert values["context_docs"][0]["text"] == "纳米硅是零维纳米材料"
    # The demo answer is not knowledge to continue from.
    assert not any("400-800-1001" in m.content for m in values["messages"])


def test_a_conversation_with_nothing_answered_has_no_state():
    assert state_from_text({"id": "x", "messages": [{"role": "user", "text": "问"}]}) is None
    assert state_from_text({"id": "x", "messages": []}) is None


def system_with_graph(tmp_path, llm=None):
    settings = Settings(_env_file=None, data_dir=tmp_path / "kb", siliconflow_api_key="sk-test")
    system = RAGSystem(settings)
    system.initialized = True
    system.agent_graph = graph(system.checkpointer, llm)
    return system


def test_an_old_conversation_continues_with_its_context(tmp_path):
    llm = HistoryAwareLLM()
    system = system_with_graph(tmp_path, llm)
    system.conversations.save(saved_conversation())

    result = system.chat("那它的粒径有多大？", "old")

    assert result["rewritten_query"] == "纳米硅的粒径"
    assert "0.8 nm" in result["answer"]
    assert "400-800-1001" not in llm.plans[0]


def test_followups_of_an_old_conversation_come_from_its_text(tmp_path):
    system = system_with_graph(tmp_path)
    system.conversations.save(saved_conversation())
    seen = {}
    system.followup_service = SimpleNamespace(
        suggest=lambda values, n: seen.update(values) or SimpleNamespace(to_dict=lambda: {"scenario": "answer", "items": [{"text": "下一问？"}]})
    )

    assert system.followups("old", "纳米硅是什么？", n=4)["items"] == [{"text": "下一问？"}]
    assert seen["answer"].startswith("纳米硅是零维纳米材料")
    # Another question than the last one answered: nothing to follow.
    assert system.followups("old", "别的问题", n=4)["items"] == []


def test_followups_survive_a_restart(tmp_path):
    system = system_with_graph(tmp_path)
    system.chat("纳米硅是什么？", "t1")
    system.checkpointer.close()

    restarted = system_with_graph(tmp_path)
    seen = {}
    restarted.followup_service = SimpleNamespace(
        suggest=lambda values, n: seen.update(values) or SimpleNamespace(to_dict=lambda: {"scenario": "answer", "items": [{"text": "下一问？"}]})
    )

    assert restarted.followups("t1", "纳米硅是什么？")["items"] == [{"text": "下一问？"}]
    assert seen["context_docs"][0]["source"] == "产品规格.docx"


def test_followups_do_not_undo_a_delete(tmp_path):
    system = system_with_graph(tmp_path)
    system.followup_service = SimpleNamespace(suggest=lambda values, n: pytest.fail("nothing to follow"))
    system.chat("纳米硅是什么？", "t1")
    system.delete_conversation("t1")

    assert system.followups("t1", "纳米硅是什么？")["items"] == []
    # A run still going when it was deleted stays dropped.
    assert "t1" in system.checkpointer._deleted


def test_deleting_a_conversation_removes_its_history_and_state(tmp_path):
    system = system_with_graph(tmp_path)
    system.conversations.save({"id": "t1", "title": "问", "updated": 1, "messages": []})
    system.chat("纳米硅是什么？", "t1")
    db = system.settings.conversations_db_path
    assert rows(db, "checkpoints", "t1") == 1

    assert system.delete_conversation("t1") is True

    assert system.conversations.get("t1") is None
    assert rows(db, "checkpoints", "t1") == 0 and rows(db, "writes", "t1") == 0


# ---------------- the history ----------------


def conversation(identifier, updated=1.0, **extra):
    return {"id": identifier, "title": identifier, "updated": updated, "messages": [{"role": "user", "text": identifier}], **extra}


def test_the_history_has_no_cap_and_lists_newest_first(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    for n in range(120):
        store.save(conversation(f"c{n}", updated=float(n)))

    listed = store.list()

    assert len(listed) == 120
    assert [c["id"] for c in listed[:2]] == ["c119", "c118"]
    assert "messages" not in listed[0]
    assert store.get("c0")["messages"] == [{"role": "user", "text": "c0"}]


def test_saving_replaces_a_conversation_and_keeps_its_json(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    store.save(conversation("a", scope={"doc_ids": ["d1"], "label": "甲.pdf"}))
    bot = {"role": "bot", "id": "b", "answer": "回答", "citations": [{"source": "甲.pdf"}], "followups": {"items": [{"text": "下一问"}]}}
    store.save({**conversation("a", updated=2.0, scope=None), "messages": [bot]})

    saved = store.get("a")

    assert saved["messages"] == [bot] and saved["updated"] == 2.0 and saved["scope"] is None
    assert len(store.list()) == 1


def test_importing_keeps_order_and_what_is_already_here(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")
    store.save(conversation("newer", updated=9.0, title="后来存的"))

    result = store.import_many([conversation("x"), conversation("newer", title="旧副本"), conversation("y")])

    assert result["imported"] == 2 and result["existing"] == 1
    assert result["ids"] == ["x", "newer", "y"]
    assert store.get("newer")["title"] == "后来存的"
    assert [c["id"] for c in store.list()] == ["newer", "x", "y"]
    # Importing again changes nothing.
    assert store.import_many([conversation("x"), conversation("y")])["imported"] == 0


def test_an_import_that_fails_stores_nothing(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite3")

    with pytest.raises(TypeError):
        store.import_many([conversation("x"), {"id": "bad", "messages": [object()]}])

    assert store.list() == []


def test_saved_conversations_survive_reopening_the_store(tmp_path):
    db = tmp_path / "c.sqlite3"
    first = ConversationStore(db)
    first.save(conversation("a"))
    first.close()

    assert ConversationStore(db).get("a")["title"] == "a"


# ---------------- the API ----------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    from api import main

    settings = Settings(_env_file=None, data_dir=tmp_path / "kb", siliconflow_api_key="")
    settings.app_token = ""
    system = RAGSystem(settings)
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "system", system)
    return TestClient(main.app)


def test_the_history_works_without_an_api_key(client):
    assert client.get("/api/v1/conversations").json() == {"conversations": []}
    assert client.put("/api/v1/conversations/a", json={"title": "问", "updated": 2, "messages": [{"role": "user", "text": "问"}]}).status_code == 200
    assert client.get("/api/v1/conversations").json()["conversations"] == [{"id": "a", "title": "问", "updated": 2.0, "scope": None}]
    assert client.get("/api/v1/conversations/a").json()["messages"] == [{"role": "user", "text": "问"}]
    assert client.delete("/api/v1/conversations/a").json() == {"id": "a", "deleted": True}
    assert client.get("/api/v1/conversations/a").status_code == 404
    assert client.delete("/api/v1/conversations/a").json()["deleted"] is False


def test_the_import_endpoint_says_what_is_stored(client):
    payload = {"conversations": [conversation(f"c{n}", updated=float(n)) for n in range(60)]}

    first = client.post("/api/v1/conversations/import", json=payload).json()
    second = client.post("/api/v1/conversations/import", json=payload).json()

    assert first["imported"] == 60 and len(first["ids"]) == 60
    assert second["imported"] == 0 and len(second["ids"]) == 60
    assert len(client.get("/api/v1/conversations").json()["conversations"]) == 60


def test_followups_without_a_key_are_simply_empty(client):
    body = client.post("/api/v1/followups", json={"thread_id": "t1", "question": "问"}).json()

    assert body["items"] == [] and body["enabled"] is True


def test_the_demo_routes_and_stores_are_gone(client, tmp_path):
    assert client.get("/mock-api/v1/inventory/NX-MEET-PRO").status_code == 404
    assert client.get("/mock-api/v1/services/meeting-cloud").status_code == 404
    assert not (tmp_path / "kb" / "enterprise.db").exists()
    assert not (tmp_path / "kb" / "business_api.json").exists()


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_moving_and_saving_in_the_ui():
    result = subprocess.run([NODE, "--test", str(JS_TESTS)], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
