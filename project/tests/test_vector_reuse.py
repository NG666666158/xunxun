"""Re-indexing an edited file embeds only the chunks whose text changed."""

import hashlib

import pytest

pytest.importorskip("sqlite_vec")
pytest.importorskip("jieba")

from config import Settings  # noqa: E402
from db.parent_store_manager import ParentStoreManager  # noqa: E402
from ingestion.pipeline import IngestionPipeline  # noqa: E402
from retrieval.embedded_store import EmbeddedStore  # noqa: E402
from retrieval.types import ChunkRecord, text_key  # noqa: E402

DOC = "c" * 32


class CountingEmbeddings:
    """Deterministic 8-d vectors from the text; records every request."""

    def __init__(self):
        self.requests: list[int] = []

    def embed_documents(self, texts):
        self.requests.append(len(texts))
        return [self.vector(text) for text in texts]

    @staticmethod
    def vector(text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:8]]


def _settings(tmp_path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        embedding_dimension=8,
        parent_chunk_size=400,
        child_chunk_size=120,
        child_chunk_overlap=20,
    )
    settings.ensure_directories()
    return settings


def _document(paragraphs):
    return "# 制度汇编\n\n" + "\n\n".join(paragraphs)


PARAGRAPHS = [f"第{i}条 员工出差应提前提交申请，住宿标准按城市等级执行，超出部分自理，编号 R-{i:03d}。" for i in range(1, 41)]


def test_store_returns_vectors_by_text(tmp_path):
    store = EmbeddedStore(_settings(tmp_path))
    store.ensure_collection()
    vector = [0.5] * 8
    store.insert([ChunkRecord("x1", DOC, f"{DOC}-p0", "a.md", "", "", "同一段文字", vector)])
    known = store.vectors_by_text(DOC)
    assert list(known) == [text_key("同一段文字")]
    assert known[text_key("同一段文字")] == pytest.approx(vector)
    assert store.vectors_by_text("d" * 32) == {} and store.vectors_by_text("bad id") == {}
    store.close()


def test_one_paragraph_edit_embeds_only_the_changed_chunks(tmp_path):
    settings = _settings(tmp_path)
    store = EmbeddedStore(settings)
    store.ensure_collection()
    embeddings = CountingEmbeddings()
    pipeline = IngestionPipeline(store, embeddings, settings, parent_store=ParentStoreManager(settings))
    path = tmp_path / "制度汇编.md"
    path.write_text(_document(PARAGRAPHS), encoding="utf-8")

    first: dict = {}
    chunks, _ = pipeline.index_document(path, DOC, "制度汇编.md", report=first)
    assert first["reused"] == 0 and first["embedded"] == chunks >= 15

    edited = list(PARAGRAPHS)
    edited[20] = "第21条 出差住宿标准调整为：一线城市每晚不超过 600 元，其他城市不超过 450 元。"
    path.write_text(_document(edited), encoding="utf-8")
    embeddings.requests.clear()
    pauses = []
    second: dict = {}
    chunks_after, _ = pipeline.index_document(path, DOC, "制度汇编.md", pause=lambda: pauses.append(1), report=second)

    assert second["embedded"] + second["reused"] == chunks_after
    assert 1 <= second["embedded"] <= 3  # the edited paragraph's chunk(s), not the whole file
    assert sum(embeddings.requests) == second["embedded"] and second["embedding_requests"] == 1
    assert pauses == [1, 1, 1]  # before parsing, chunking and writing; no batch boundary was crossed
    # Every stored vector is the one its text would get: reused vectors belong to their text.
    rows = store._conn.execute("SELECT text FROM chunks WHERE doc_id = ?", (DOC,)).fetchall()
    known = store.vectors_by_text(DOC)
    for (text,) in rows:
        assert known[text_key(text)] == pytest.approx(CountingEmbeddings.vector(text), abs=1e-6)
    store.close()


def test_pause_runs_between_embedding_requests(tmp_path):
    settings = _settings(tmp_path)
    store = EmbeddedStore(settings)
    store.ensure_collection()
    embeddings = CountingEmbeddings()
    pipeline = IngestionPipeline(store, embeddings, settings, parent_store=ParentStoreManager(settings))
    pipeline.embed_batch = 8
    path = tmp_path / "长文.md"
    path.write_text(_document(PARAGRAPHS), encoding="utf-8")
    pauses = []
    report: dict = {}
    chunks, _ = pipeline.index_document(path, DOC, pause=lambda: pauses.append(len(embeddings.requests)), report=report)
    batches = -(-chunks // 8)
    assert report["embedding_requests"] == batches == len(embeddings.requests)
    # Before parsing and chunking, before every request but the first, then before writing.
    assert pauses == [0, *range(0, batches), batches]
    store.close()


def test_stores_without_lookup_embed_everything(tmp_path):
    settings = _settings(tmp_path)

    class PlainStore:
        rows = []

        def delete_document(self, doc_id):
            self.rows = []

        def insert(self, records):
            self.rows = list(records)

    embeddings = CountingEmbeddings()
    pipeline = IngestionPipeline(PlainStore(), embeddings, settings, parent_store=ParentStoreManager(settings))
    path = tmp_path / "a.md"
    path.write_text(_document(PARAGRAPHS[:5]), encoding="utf-8")
    report: dict = {}
    chunks, _ = pipeline.index_document(path, DOC, report=report)
    pipeline.index_document(path, DOC, report=report)
    assert report["reused"] == 0 and report["embedded"] == chunks  # e.g. Milvus: re-embeds as before
