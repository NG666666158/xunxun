"""Same-named uploads must coexist and document listing must not cap out."""

import io
from types import SimpleNamespace

from fastapi import UploadFile

from api import main
from config import Settings
from core.rag_system import RAGSystem
from db.parent_store_manager import ParentStoreManager
from ingestion.parsers import stable_doc_id
from ingestion.pipeline import IngestionPipeline
from retrieval.milvus_store import MilvusStore


class FakeVectorStore:
    """In-memory stand-in for MilvusStore keyed by chunk rows."""

    def __init__(self):
        self.rows = []

    def has_document(self, doc_id):
        return any(row.doc_id == doc_id for row in self.rows)

    def document_ids_for_source(self, source):
        return {row.doc_id for row in self.rows if row.source == source}

    def list_documents(self):
        return sorted({(row.doc_id, row.source) for row in self.rows})

    def delete_document(self, doc_id):
        self.rows = [row for row in self.rows if row.doc_id != doc_id]

    def insert(self, records):
        self.rows.extend(records)
        return len(records)


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]


def _settings(tmp_path):
    settings = Settings(
        _env_file=None,
        upload_dir=tmp_path / "uploads",
        parsed_dir=tmp_path / "parsed",
        parent_store_dir=tmp_path / "parents",
    )
    for path in (settings.upload_dir, settings.parsed_dir, settings.parent_store_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings


def _pipeline(settings):
    return IngestionPipeline(FakeVectorStore(), FakeEmbeddings(), settings)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_same_name_different_content_keeps_both(tmp_path):
    settings = _settings(tmp_path)
    pipeline = _pipeline(settings)

    first = pipeline.ingest(_write(tmp_path / "a" / "合同.md", "# 甲\n第一份合同"))
    second = pipeline.ingest(_write(tmp_path / "b" / "合同.md", "# 乙\n第二份合同"))

    assert first.doc_id != second.doc_id
    assert {first.status, second.status} == {"indexed"}
    assert {doc_id for doc_id, _ in pipeline.store.list_documents()} == {first.doc_id, second.doc_id}
    assert (settings.parsed_dir / f"{first.doc_id}.json").is_file()


def test_identical_upload_is_skipped(tmp_path):
    pipeline = _pipeline(_settings(tmp_path))
    first = pipeline.ingest(_write(tmp_path / "a" / "合同.md", "同一份"))
    again = pipeline.ingest(_write(tmp_path / "b" / "合同.md", "同一份"))
    assert again.doc_id == first.doc_id
    assert again.status == "skipped"


def test_explicit_replace_removes_same_named_versions(tmp_path):
    settings = _settings(tmp_path)
    pipeline = _pipeline(settings)
    old = pipeline.ingest(_write(tmp_path / "a" / "合同.md", "旧版本"))
    _write(settings.upload_dir / old.doc_id / "合同.md", "旧版本")

    new = pipeline.ingest(_write(tmp_path / "b" / "合同.md", "新版本"), replace=True)

    assert [doc_id for doc_id, _ in pipeline.store.list_documents()] == [new.doc_id]
    assert not (settings.parsed_dir / f"{old.doc_id}.json").exists()
    assert not (settings.upload_dir / old.doc_id).exists()


class PagingClient:
    """Mimics MilvusClient.query_iterator returning rows in batches."""

    def __init__(self, rows):
        self.rows = rows
        self.filters = []

    def query(self, **kwargs):
        raise AssertionError("list_documents must not use a capped query")

    def query_iterator(self, collection_name, batch_size, filter, output_fields):
        self.filters.append(filter)
        rows = iter(self.rows)

        class Iterator:
            def next(self):
                return [row for _, row in zip(range(batch_size), rows)]

            def close(self):
                pass

        return Iterator()


def test_list_documents_pages_past_query_limit(tmp_path):
    # 20k chunks of one document followed by a second document: a single
    # query with limit=16384 would never see the second one.
    rows = [{"doc_id": "a" * 24, "source": "big.pdf"}] * 20000
    rows.append({"doc_id": "b" * 24, "source": "late.md"})
    store = MilvusStore(_settings(tmp_path), client=PagingClient(rows))

    assert store.list_documents() == [
        {"doc_id": "a" * 24, "source": "big.pdf"},
        {"doc_id": "b" * 24, "source": "late.md"},
    ]


def test_document_ids_for_source_escapes_filter(tmp_path):
    client = PagingClient([{"doc_id": "a" * 24, "source": 'x"y\\z.md'}])
    store = MilvusStore(_settings(tmp_path), client=client)
    assert store.document_ids_for_source('x"y\\z.md') == {"a" * 24}
    assert client.filters == ['source == "x\\"y\\\\z.md"']


def test_original_file_prefers_doc_dir_and_checks_legacy_hash(tmp_path):
    settings = _settings(tmp_path)
    rag = RAGSystem(settings)
    legacy = _write(settings.upload_dir / "合同.md", "旧上传")
    legacy_id = stable_doc_id(legacy)

    # Legacy flat file is only served for the document it actually belongs to.
    assert rag.original_file(legacy_id, "合同.md") == legacy
    assert rag.original_file("c" * 24, "合同.md") is None

    current = _write(settings.upload_dir / ("c" * 24) / "合同.md", "新上传")
    assert rag.original_file("c" * 24, "合同.md") == current
    assert rag.original_file("../x", "合同.md") is None


def test_upload_endpoint_stores_same_named_files_separately(monkeypatch, tmp_path):
    from library import FileRegistry, LibraryService

    settings = _settings(tmp_path)
    store = FakeVectorStore()
    indexed = []
    library = LibraryService(
        settings,
        FileRegistry(tmp_path / "registry.sqlite3"),
        store,
        ParentStoreManager(settings),
        indexer=lambda path, file_id, source: indexed.append((file_id, source)) or (1, 3),
    )
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "_library", lambda: library)

    def upload(text):
        data = text.encode("utf-8")
        return UploadFile(io.BytesIO(data), size=len(data), filename="合同.md")

    first = main.upload_documents([upload("第一份")])["results"][0]
    second = main.upload_documents([upload("第二份")])["results"][0]

    # Upload returns immediately; indexing happens in the background queue.
    assert first["status"] == second["status"] == "pending"
    assert first["doc_id"] != second["doc_id"]
    for result, text in ((first, "第一份"), (second, "第二份")):
        path = settings.upload_dir / result["doc_id"] / "合同.md"
        assert path.read_text(encoding="utf-8") == text
    assert not any((settings.upload_dir / ".staging").iterdir())

    assert library.run_until_idle() == ["indexed", "indexed"]
    assert {file_id for file_id, _ in indexed} == {first["doc_id"], second["doc_id"]}
    assert library.registry.get_file(first["doc_id"])["status"] == "indexed"
