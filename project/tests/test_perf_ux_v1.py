"""Tests for feat/perf-ux-v1: survey rejection guard, hedge backoff, batched per-doc retrieval,
SQLite doc_type pushdown, early SSE preview sources, and API deduplication.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from api import main as api_main
from config import Settings
from core.rag_system import RAGSystem
from library.registry import FileRegistry, SEARCHABLE_STATUSES
from library.service import LibraryService
from rag_agent.nodes import AgentNodes
from retrieval.doc_cards import OTHER_TYPE
from retrieval.embedded_store import EmbeddedStore
from retrieval.types import ChunkRecord
from services.model_clients import SiliconFlowChat, chat_busy


def test_survey_answer_skips_rejection_and_anchor_notes(tmp_path):
    """Survey answers synthesize across multiple files and must not trigger single-doc rejection/anchor notes."""
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        rejection_note="all",
        rejection_refusal=True,
        anchor_note=True,
        conflict_finder=True,
        survey_route=True,
    )

    class StubLLM:
        def invoke(self, _messages, **_kwargs):
            return SimpleNamespace(content="- A指南最高资助50万元[来源：A指南.docx | 位置：p1]。")

    nodes = AgentNodes(StubLLM(), retriever=None, settings=settings)
    state = {
        "question": "一般的创新项目申报能给多少补贴？",
        "intent": "knowledge",
        "answer_profile": "survey",
        "messages": [],
        "context_docs": [
            {
                "id": "d1-0",
                "doc_id": "d1",
                "source": "A指南.docx",
                "locator": "p1",
                "text": "单个项目资助资金最高不超过50万元。",
            }
        ],
    }
    out = nodes.generate_answer(state)
    assert out["refused"] is not True if "refused" in out else True
    assert out["rejection_notes"] == []
    assert out["anchor_notes"] == []
    assert "没有找到" not in out["answer"]


def test_hedge_skipped_when_busy_or_nostream(monkeypatch):
    """First-token hedging must not fire when the model is already noted busy or tagged nostream."""
    chat_busy.clear()
    chat = SiliconFlowChat(model="Qwen/Qwen3-8B", api_key="test-key", base_url="http://127.0.0.1:1/v1", hedge_after=8.0)

    hedged_calls = []
    direct_calls = []

    def fake_hedge(self, messages, stop, kwargs):
        hedged_calls.append(True)
        if False:
            yield

    def fake_super_stream(self, messages, stop=None, run_manager=None, **kwargs):
        direct_calls.append(True)
        if False:
            yield

    monkeypatch.setattr(SiliconFlowChat, "_first_to_answer", fake_hedge)
    monkeypatch.setattr("langchain_openai.ChatOpenAI._stream", fake_super_stream)

    # 1. Normal interactive call uses hedge
    list(chat._stream([]))
    assert len(hedged_calls) == 1 and len(direct_calls) == 0

    # 2. nostream tag skips hedge
    list(chat._stream([], run_manager=SimpleNamespace(tags=["nostream"], on_llm_new_token=lambda *a, **k: None)))
    assert len(hedged_calls) == 1 and len(direct_calls) == 1

    # 3. Busy model skips hedge
    chat_busy.note("Qwen/Qwen3-8B", seconds=30.0)
    try:
        list(chat._stream([]))
        assert len(hedged_calls) == 1 and len(direct_calls) == 2
    finally:
        chat_busy.clear()


def test_search_hybrid_per_doc_matches_individual_calls(tmp_path):
    """Batched single-snapshot per-doc hybrid search must match individual scoped calls."""
    settings = Settings(_env_file=None, data_dir=tmp_path, embedding_dimension=64)
    store = EmbeddedStore(settings)
    store.ensure_collection()
    rng = np.random.default_rng(9)
    try:
        doc_ids = [f"{i + 1:032x}" for i in range(6)]
        for i, doc_id in enumerate(doc_ids):
            batch = []
            for c in range(6):
                v = rng.normal(size=64).astype(np.float32)
                v /= np.linalg.norm(v)
                text = f"创新项目第{i}号指南第{c}节：资助标准最高{10 + i + c}万元。"
                batch.append(ChunkRecord(f"{doc_id}-{c}", doc_id, f"{doc_id}-p{c}", f"指南_{i}.docx", "", f"p{c}", text, v.tolist()))
            store.replace_document(doc_id, batch)

        qv = rng.normal(size=64).astype(np.float32)
        qv /= np.linalg.norm(qv)
        query = "创新项目资助标准"
        batched = store.search_hybrid_per_doc(query, qv.tolist(), doc_ids, per_doc=3)
        for doc_id in doc_ids:
            single = store.search_hybrid(query, qv.tolist(), limit=3, doc_ids={doc_id})
            assert [h.id for h in batched[doc_id]] == [h.id for h in single]
            assert [pytest.approx(h.score, rel=1e-6) for h in batched[doc_id]] == [h.score for h in single]
    finally:
        store.close()


def test_sqlite_doc_type_filter_and_summary_match_python_semantics(tmp_path):
    """SQL pushdown for doc_type filtering and library_types must match name-over-card rules exactly."""
    settings = Settings(_env_file=None, data_dir=tmp_path, doc_cards=True)
    db_file = tmp_path / "knowledge.sqlite3"
    store = EmbeddedStore(settings, db_path=db_file)
    store.ensure_collection()
    reg = FileRegistry(db_file)
    lib = LibraryService(settings, reg, store, parent_store=None, indexer=None)
    try:
        root = reg.add_root(str(tmp_path / "docs"), volume_id=None)
        f_named = reg.insert_file(str(tmp_path / "docs" / "2026_工作报告.docx"), root_id=root["root_id"], status="indexed")
        f_carded = reg.insert_file(str(tmp_path / "docs" / "alpha.pdf"), root_id=root["root_id"], status="indexed")
        f_override = reg.insert_file(str(tmp_path / "docs" / "采购合同.pdf"), root_id=root["root_id"], status="indexed")
        f_other_card = reg.insert_file(str(tmp_path / "docs" / "misc.pdf"), root_id=root["root_id"], status="indexed")
        f_pending = reg.insert_file(str(tmp_path / "docs" / "wait.pdf"), root_id=root["root_id"], status="indexed")
        f_name_only = reg.insert_file(
            str(tmp_path / "docs" / "secret.pdf"), root_id=root["root_id"], status="indexed", content_mode="name"
        )

        # Card on f_carded -> 简历; card on f_override says 报告, but filename 合同 must win!
        store.upsert_doc_card(f_carded["file_id"], {"doc_type": "简历", "label": "简历", "subject": "", "period": "", "summary": "", "input_hash": "h1", "version": 3})
        store.upsert_doc_card(f_override["file_id"], {"doc_type": "报告", "label": "报告", "subject": "", "period": "", "summary": "", "input_hash": "h2", "version": 3})
        store.upsert_doc_card(f_other_card["file_id"], {"doc_type": "", "label": "其他", "subject": "", "period": "", "summary": "", "input_hash": "h3", "version": 3})

        # Filter by 合同 -> f_override only
        rows, total = reg.list_files_by_doc_type("合同")
        assert total == 1 and [r["file_id"] for r in rows] == [f_override["file_id"]]

        # Filter by 报告 -> 0 (f_named is 工作报告, f_override is overridden by 合同)
        rows, total = reg.list_files_by_doc_type("报告")
        assert total == 0

        # Filter by 工作报告 -> f_named
        rows, total = reg.list_files_by_doc_type("工作报告")
        assert total == 1 and [r["file_id"] for r in rows] == [f_named["file_id"]]

        # Filter by 简历 -> f_carded
        rows, total = reg.list_files_by_doc_type("简历")
        assert total == 1 and [r["file_id"] for r in rows] == [f_carded["file_id"]]

        # Filter by 其他 -> only f_other_card (has card with empty doc_type), NOT f_pending/f_name_only
        rows, total = reg.list_files_by_doc_type(OTHER_TYPE)
        assert total == 1 and [r["file_id"] for r in rows] == [f_other_card["file_id"]]

        # Summary counts & pending (only f_pending waits for a card; f_name_only & f_named do not)
        counts, pending = reg.doc_type_summary(statuses=SEARCHABLE_STATUSES, other_type=OTHER_TYPE)
        assert counts == {"工作报告": 1, "简历": 1, "合同": 1, OTHER_TYPE: 3}
        assert pending == 1
        assert api_main._shared_doc_cards_sql(lib) is True
    finally:
        reg.close()
        store.close()


def test_stream_emits_preview_sources_on_retrieve(monkeypatch, tmp_path):
    """The retrieve SSE node event should include deduplicated preview_sources before answer generation."""
    settings = Settings(_env_file=None, data_dir=tmp_path)
    rag = RAGSystem(settings)
    rag.agent_graph = SimpleNamespace(
        get_state=lambda _cfg: SimpleNamespace(
            values={"answer": "答案", "citations": [], "grounded": True, "intent": "knowledge"}
        )
    )
    monkeypatch.setattr(rag, "_begin_turn", lambda _tid: None)
    monkeypatch.setattr(rag, "_followups_wanted", lambda: False)
    monkeypatch.setattr(
        rag,
        "_graph_stream",
        lambda *_a, **_k: iter([
            ("updates", {
                "retrieve": {
                    "retrieval_attempts": 1,
                    "documents": [
                        {"doc_id": "d1", "source": "a.pdf", "locator": "p1"},
                        {"doc_id": "d1", "source": "a.pdf", "locator": "p2"},
                        {"doc_id": "d2", "source": "b.docx", "locator": "s1"},
                    ],
                }
            })
        ]),
    )
    events = list(rag._stream("问题", "t1", scope=None))
    retrieve_ev = next(ev for ev in events if ev.get("event") == "node" and ev.get("node") == "retrieve")
    assert retrieve_ev["retrieved"] == 3
    assert [s["doc_id"] for s in retrieve_ev["preview_sources"]] == ["d1", "d2"]
