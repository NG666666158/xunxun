"""Per-stage answer timings (core/timings.py) and where they are kept."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import zipfile
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from core import diagnostics, timings
from rag_agent.graph import create_agent_graph
from tests.test_graph_integration import FakeLLM, FakeParentStore, FakeRetriever, GraphSettings


def test_stages_add_up_only_inside_an_active_turn(monkeypatch):
    clock = iter((10.0, 10.01, 10.02, 10.04, 10.05))
    monkeypatch.setattr(timings, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    with timings.stage("embed"):
        pass  # no turn: nothing to record, nothing raised
    turn = timings.TurnTimings(started=10.0)
    with timings.activate(turn):
        with timings.stage("search"):
            pass
        with timings.stage("search"):
            pass
    record = turn.finish(intent="knowledge")
    assert record["stages"]["search"] == 30
    assert record["total_ms"] == 50
    assert record["counts"] == {"search": 2}
    assert record["intent"] == "knowledge"
    assert timings.current() is None


def test_iterate_keeps_the_turn_current_in_whichever_thread_resumes_the_generator():
    turn = timings.TurnTimings()

    def steps():
        for name in ("a", "b"):
            with timings.stage(name):
                where = threading.current_thread().name
            yield where

    generator = timings.iterate(steps(), turn)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="other") as pool:
        first = next(generator)
        second = pool.submit(next, generator).result()
    assert first != second
    assert set(turn.stages) == {"a", "b"}


def test_bind_context_carries_the_turn_into_a_thread_pool():
    turn = timings.TurnTimings()

    def work(_=None):
        with timings.stage("notes"):
            return 1

    with timings.activate(turn), ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(timings.bind_context(work), range(3))) == 3
        pool.submit(work).result()  # not bound: not counted
    assert turn.counts.get("notes") == 3


def test_graph_nodes_are_timed_and_model_calls_named_by_node():
    turn = timings.TurnTimings()

    class TimedLLM:
        def __init__(self, inner):
            self.inner = inner

        def invoke(self, messages):
            timer = timings.ModelCallTimer()
            reply = self.inner.invoke(messages)
            timer.token()
            timer.usage = {"input_tokens": 10, "output_tokens": 2}
            timer.done()
            return reply

    graph = create_agent_graph(
        llm=TimedLLM(FakeLLM()), retriever=FakeRetriever(), settings=GraphSettings(), parent_store=FakeParentStore()
    )
    with timings.activate(turn):
        graph.invoke(
            {"messages": [HumanMessage(content="住宿费超标怎么办？")]},
            config={"configurable": {"thread_id": "t"}},
        )
    record = turn.finish()
    assert {"analyze_query", "retrieve", "build_context", "generate_answer", "fact_check"} <= set(record["stages"])
    nodes = [call["node"] for call in record["calls"]]
    assert "generate_answer" in nodes and "fact_check" in nodes
    assert record["first_token_ms"] is not None


def test_saved_records_feed_the_diagnostics_export(tmp_path):
    class Settings:
        data_dir = tmp_path
        log_dir = tmp_path / "logs"
        embedded_db_path = tmp_path / "knowledge.sqlite3"
        app_version = "test"
        llm_model = embedding_model = rerank_model = "m"
        answer_mode = "strict"
        image_understanding = "local"
        siliconflow_api_key = "sk-secret-value"
        app_token = ""

    for total in (1000, 3000, 9000):
        turn = timings.TurnTimings()
        turn.add("generate_answer", total / 2)
        turn.total_ms = float(total)
        timings.save(turn.finish(intent="knowledge", profile="fact"), Settings.log_dir)
    path = diagnostics.export(Settings, {"state": "ok"})
    with zipfile.ZipFile(path) as archive:
        report = json.loads(archive.read("answer_timings.json"))
    assert report["overview"]["answers"] == 3
    assert report["overview"]["total_ms"]["p50"] == 3000
    assert report["recent"][0]["profile"] == "fact"
    assert "question" not in json.dumps(report)


def test_summary_line_names_the_slow_stages():
    record = {"kind": "answer", "intent": "knowledge", "profile": "explain", "total_ms": 12500,
              "first_token_ms": 4200, "stages": {"generate_answer": 8000, "embed": 20}, "calls": [{}, {}]}
    line = timings.summary_line(record)
    assert "total=12.5s" in line and "first_token=4.2s" in line and "generate_answer=8.0s" in line
    assert "embed" not in line
