"""Backend selection, pymilvus isolation and the Milvus -> embedded migration."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from config import Settings
from retrieval import factory
from retrieval.types import ChunkRecord


REPO_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_DIR / "project"
DOC_A = "a" * 24
DOC_B = "b" * 24


def _settings(tmp_path, **overrides):
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        parsed_dir=tmp_path / "parsed",
        parent_store_dir=tmp_path / "parents",
        cache_dir=tmp_path / "cache",
        embedded_db_path=tmp_path / "store.sqlite3",
        embedding_dimension=4,
        **overrides,
    )


def _run_python(code: str, **env) -> subprocess.CompletedProcess:
    base = {key: value for key, value in os.environ.items() if key.upper() != "VECTOR_BACKEND"}
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_DIR,
        env={**base, **env},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_embedded_is_the_default_backend(monkeypatch):
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    settings = Settings(_env_file=None)
    assert settings.vector_backend == "embedded"
    assert settings.embedded_db_path.parent == settings.data_dir


def test_backend_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "milvus")
    assert Settings(_env_file=None).vector_backend == "milvus"


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError):
        Settings(_env_file=None, vector_backend="faiss")


def test_factory_builds_embedded_store_at_configured_path(tmp_path):
    pytest.importorskip("sqlite_vec")
    store = factory.create_store(_settings(tmp_path))
    try:
        assert store.backend == "embedded"
        assert store.db_path == tmp_path / "store.sqlite3"
    finally:
        store.close()


def test_factory_builds_milvus_store(tmp_path, monkeypatch):
    milvus_store = pytest.importorskip("retrieval.milvus_store")
    monkeypatch.setattr(milvus_store, "MilvusClient", lambda **kwargs: kwargs)
    store = factory.create_store(_settings(tmp_path, vector_backend="milvus"))
    assert store.backend == "milvus"
    assert store.client["uri"] == "http://127.0.0.1:19530"


def test_embedded_startup_does_not_import_pymilvus(tmp_path):
    # Fresh interpreter: import the API app and build its store; nothing on that
    # path may pull in pymilvus or grpc.
    code = "\n".join(
        [
            "import sys",
            "from pathlib import Path",
            "from api import main",
            "from retrieval.factory import create_store",
            "assert main.settings.vector_backend == 'embedded'",
            f"probe = main.settings.model_copy(update={{'embedded_db_path': Path({str(tmp_path)!r}) / 'probe.sqlite3'}})",
            "create_store(probe).close()",
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in {'pymilvus', 'grpc'})",
            "assert not bad, bad",
        ]
    )
    result = _run_python(code, VECTOR_BACKEND="embedded")
    assert result.returncode == 0, result.stderr


def test_jieba_cache_goes_to_app_data_dir(tmp_path):
    pytest.importorskip("jieba")
    cache = str(tmp_path / "cache")
    code = "\n".join(
        [
            "import pathlib, jieba",
            "from retrieval.embedded_store import set_jieba_cache_dir, tokenize",
            f"set_jieba_cache_dir({cache!r})",
            "tokenize('住宿费超标')",
            f"assert jieba.dt.tmp_dir == {cache!r}, jieba.dt.tmp_dir",
            f"assert any(pathlib.Path({cache!r}).glob('jieba*.cache'))",
        ]
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr


# --- migration ----------------------------------------------------------------


def _migration_module():
    path = REPO_DIR / "scripts" / "migrate_milvus_to_embedded.py"
    spec = importlib.util.spec_from_file_location("migrate_milvus_to_embedded", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExportClient:
    """Mimics MilvusClient.query_iterator over full chunk rows, vectors included."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query_iterator(self, collection_name, batch_size, filter, output_fields):
        self.calls.append({"filter": filter, "output_fields": output_fields, "batch_size": batch_size})
        rows = iter(self.rows)

        class Iterator:
            def next(self):
                return [{key: row[key] for key in output_fields} for _, row in zip(range(batch_size), rows)]

            def close(self):
                pass

        return Iterator()


def _rows():
    rows = []
    for doc_index, (doc_id, source) in enumerate(((DOC_A, "差旅制度.md"), (DOC_B, "报销流程.pdf"))):
        for chunk in range(3):
            vector = [0.0] * 4
            vector[(doc_index + chunk) % 4] = 1.0
            rows.append(
                {
                    "id": f"{doc_id}-c{chunk}",
                    "doc_id": doc_id,
                    "parent_id": f"{doc_id}-p0",
                    "source": source,
                    "heading": "住宿标准",
                    "locator": f"第{chunk + 1}段",
                    "text": f"住宿费超标需要事前审批，编号 SEC-{doc_index}{chunk}",
                    "dense": vector,
                }
            )
    return rows


def test_migration_copies_chunks_and_vectors_without_embedding(tmp_path, monkeypatch):
    pytest.importorskip("sqlite_vec")
    milvus_store = pytest.importorskip("retrieval.milvus_store")
    from retrieval.embedded_store import EmbeddedStore
    from services import model_clients

    def no_embedding(*args, **kwargs):
        raise AssertionError("migration must not call the embedding API")

    for name in ("embed_documents", "embed_query"):
        monkeypatch.setattr(model_clients.SiliconFlowEmbeddings, name, no_embedding, raising=False)

    settings = _settings(tmp_path)
    client = ExportClient(_rows())
    source = milvus_store.MilvusStore(settings, client=client)
    target = EmbeddedStore(settings, db_path=settings.embedded_db_path)
    migration = _migration_module()
    try:
        report = migration.migrate(source, target, batch_size=2)

        assert report["chunks"] == 6
        assert report["mismatched"] == {}
        assert [doc["chunks"] for doc in report["documents"]] == [3, 3]
        assert "dense" in client.calls[0]["output_fields"]
        assert client.calls[0]["batch_size"] == 2
        assert target.list_documents() == [
            {"doc_id": DOC_A, "source": "差旅制度.md"},
            {"doc_id": DOC_B, "source": "报销流程.pdf"},
        ]
        # Vectors arrive unchanged: an exact match scores cosine similarity 1.
        dense = target.search_dense([0.0, 0.0, 1.0, 0.0], limit=1)[0]
        assert dense.score == pytest.approx(1.0)
        assert dense.id in {f"{DOC_A}-c2", f"{DOC_B}-c1"}
        assert target.search_sparse("SEC-12", limit=1)[0].id == f"{DOC_B}-c2"

        # Re-running is an idempotent upsert; clear=True rebuilds from scratch.
        migration.migrate(source, target, batch_size=4)
        assert target.count() == 6
        partial = milvus_store.MilvusStore(settings, client=ExportClient(_rows()[:3]))
        migration.migrate(partial, target, clear=True)
        assert target.count() == 3
    finally:
        target.close()


def test_migration_reports_dropped_chunks(tmp_path):
    pytest.importorskip("sqlite_vec")
    from retrieval.embedded_store import EmbeddedStore

    class Source:
        def iter_chunks(self, batch_size):
            yield from (ChunkRecord(**row) for row in _rows())

    class LossyTarget(EmbeddedStore):
        def insert(self, records):
            return super().insert([record for record in records if not record.id.endswith("c2")])

    settings = _settings(tmp_path)
    target = LossyTarget(settings, db_path=settings.embedded_db_path)
    try:
        report = _migration_module().migrate(Source(), target)
    finally:
        target.close()
    assert report["mismatched"] == {
        DOC_A: {"milvus": 3, "embedded": 2},
        DOC_B: {"milvus": 3, "embedded": 2},
    }
