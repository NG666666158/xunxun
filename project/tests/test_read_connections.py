"""Reads never wait for a write: searches and registry lookups use connections of their own.

While the library re-indexes a large file, its write transaction (and a registry update
waiting for it) used to hold the one connection that answers read from as well.
"""

import sqlite3
import threading
import time

import pytest

pytest.importorskip("sqlite_vec")
pytest.importorskip("jieba")

from db.read_pool import ReaderPool  # noqa: E402
from library.registry import FileRegistry  # noqa: E402
from retrieval.embedded_store import EmbeddedStore  # noqa: E402
from retrieval.types import ChunkRecord  # noqa: E402

DOC_A = "a" * 32
DOC_B = "b" * 32
DOC_C = "c" * 32


class StoreSettings:
    milvus_collection = "unit"
    embedding_dimension = 4
    dense_top_k = 3
    sparse_top_k = 3
    fusion_top_k = 3
    rrf_k = 60

    def __init__(self, data_dir):
        self.data_dir = data_dir


def record(identifier, doc_id, text, dense):
    return ChunkRecord(identifier, doc_id, f"{doc_id}-p0", "policy.txt", "", "", text, dense)


@pytest.fixture
def store(tmp_path):
    store = EmbeddedStore(StoreSettings(tmp_path))
    store.ensure_collection()
    store.insert([
        record("c1", DOC_A, "差旅报销需要在30天内提交发票", [1, 0, 0, 0]),
        record("c2", DOC_A, "业务标识 SEC-05-ZX11 用于特权账号开通", [0, 1, 0, 0]),
        record("c3", DOC_B, "年假申请由直属负责人审批", [0, 0, 1, 0]),
    ])
    yield store
    store.close()


def _ids(hits):
    return {hit.id for hit in hits}


def test_a_search_does_not_wait_for_a_write_in_progress(store):
    written, release = threading.Event(), threading.Event()

    def long_write():  # an indexer in the middle of a large document's transaction
        with store._lock, store._transaction():
            store._insert_rows(store._prepare([record("new", DOC_C, "新增的差旅说明", [1, 0, 0, 0])]))
            written.set()
            release.wait(10)

    writer = threading.Thread(target=long_write)
    writer.start()
    assert written.wait(5)
    try:
        started = time.perf_counter()
        hits = store.search_hybrid("差旅", [1, 0, 0, 0], limit=5)
        assert time.perf_counter() - started < 2
        # The last committed state: the write is not visible yet.
        assert "c1" in _ids(hits) and "new" not in _ids(hits)
        assert not store.has_document(DOC_C) and store.count() == 3
    finally:
        release.set()
        writer.join()
    assert "new" in _ids(store.search_hybrid("差旅", [1, 0, 0, 0], limit=5))


def test_a_document_being_replaced_stays_searchable(store, monkeypatch):
    deleted, release = threading.Event(), threading.Event()
    insert_rows = store._insert_rows

    def held_insert(rows):  # old chunks deleted, new ones not written yet
        deleted.set()
        release.wait(10)
        insert_rows(rows)

    monkeypatch.setattr(store, "_insert_rows", held_insert)
    worker = threading.Thread(
        target=store.replace_document, args=(DOC_A, [record("c9", DOC_A, "新版差旅报销规定", [1, 0, 0, 0])])
    )
    worker.start()
    assert deleted.wait(5)
    try:
        assert store.chunk_counts()[DOC_A] == 2  # the old version, whole
        assert store.search_dense([1, 0, 0, 0], limit=1)[0].id == "c1"
    finally:
        release.set()
        worker.join()
    assert store.chunk_counts() == {DOC_A: 1, DOC_B: 1}
    assert store.search_dense([1, 0, 0, 0], limit=1)[0].id == "c9"


def test_replace_document_checks_before_writing(store):
    with pytest.raises(ValueError):
        store.replace_document(DOC_A, [record("bad", DOC_A, "维度不对", [1, 0, 0])])
    with pytest.raises(ValueError):
        store.replace_document("not an id", [])
    assert store.chunk_counts()[DOC_A] == 2  # nothing was deleted


def test_reads_use_read_only_connections(store):
    with store._reading() as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM chunks")
    assert store.count() == 3


def test_read_connections_follow_a_cleared_store(store):
    assert store.search_dense([1, 0, 0, 0], limit=1)[0].id == "c1"
    store.clear()  # tables dropped and created again under the read connections
    assert store.count() == 0 and store.search_dense([1, 0, 0, 0], limit=1) == []
    store.insert([record("d1", DOC_C, "重建后的内容", [1, 0, 0, 0])])
    assert store.search_hybrid("重建", [1, 0, 0, 0], limit=1)[0].id == "d1"


def _held_elsewhere(lock) -> bool:
    if lock.acquire(blocking=False):
        lock.release()
        return False
    return True


def test_registry_lookups_do_not_queue_behind_a_waiting_update(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    registry = FileRegistry(path)
    row = registry.insert_file(str(tmp_path / "a.txt"), root_id=None)
    # The chunk store's write transaction: the registry's update waits for it (busy timeout).
    store = sqlite3.connect(path, isolation_level=None)
    store.execute("BEGIN IMMEDIATE")
    update = threading.Thread(target=registry.update_file, args=(row["file_id"],), kwargs={"status": "indexing"})
    update.start()
    deadline = time.monotonic() + 5
    while not _held_elsewhere(registry._lock) and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert _held_elsewhere(registry._lock)
        started = time.perf_counter()
        assert registry.get_file(row["file_id"])["status"] == "pending"
        assert [r["file_id"] for r in registry.list_files()] == [row["file_id"]]
        assert registry.status_counts()["pending"] == 1
        assert time.perf_counter() - started < 2
    finally:
        store.execute("COMMIT")
        store.close()
        update.join()
    assert registry.get_file(row["file_id"])["status"] == "indexing"
    registry.close()


def test_reader_pool_opens_connections_only_as_reads_overlap(tmp_path):
    opened = []

    def connect():
        conn = sqlite3.connect(tmp_path / "x.db", check_same_thread=False)
        opened.append(conn)
        return conn

    pool = ReaderPool(connect, size=2)
    with pool.connection() as first:
        pass
    with pool.connection() as again:
        assert again is first
    entered = threading.Event()

    def third():
        with pool.connection():
            entered.set()

    with pool.connection() as one, pool.connection() as two:
        assert one is not two
        waiting = threading.Thread(target=third)
        waiting.start()
        assert not entered.wait(0.2)  # both in use: the third read waits
    assert entered.wait(5)
    waiting.join()
    assert len(opened) == 2
    pool.close()
    with pytest.raises(sqlite3.ProgrammingError):
        with pool.connection():
            pass
