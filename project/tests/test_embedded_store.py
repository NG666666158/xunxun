import pytest

pytest.importorskip("sqlite_vec")
pytest.importorskip("jieba")

from retrieval.embedded_store import EmbeddedStore, build_match_query  # noqa: E402
from retrieval.types import ChunkRecord  # noqa: E402


DOC_A = "a" * 24
DOC_B = "b" * 24


class StoreSettings:
    milvus_collection = "unit"
    embedding_dimension = 4
    dense_top_k = 3
    sparse_top_k = 3
    fusion_top_k = 3
    rrf_k = 60

    def __init__(self, data_dir):
        self.data_dir = data_dir


def record(identifier, doc_id, text, dense, source="policy.txt"):
    return ChunkRecord(
        id=identifier,
        doc_id=doc_id,
        parent_id=f"{doc_id}-p0",
        source=source,
        heading="",
        locator="",
        text=text,
        dense=dense,
    )


@pytest.fixture
def store(tmp_path):
    store = EmbeddedStore(StoreSettings(tmp_path))
    store.ensure_collection()
    store.insert(
        [
            record("c1", DOC_A, "差旅报销需要在30天内提交发票", [1, 0, 0, 0]),
            record("c2", DOC_A, "业务标识 SEC-05-ZX11 用于特权账号开通", [0, 1, 0, 0]),
            record("c3", DOC_B, "年假申请由直属负责人审批", [0, 0, 1, 0], source="hr.md"),
        ]
    )
    yield store
    store.close()


def test_dense_search_returns_cosine_similarity_order(store):
    hits = store.search_dense([0.9, 0.1, 0, 0], limit=2)

    assert [hit.id for hit in hits] == ["c1", "c2"]
    assert hits[0].score > hits[1].score
    assert hits[0].retrieval_method == "dense"


def test_sparse_search_matches_jieba_tokens_and_codes(store):
    assert store.search_sparse("报销发票怎么提交")[0].id == "c1"
    assert store.search_sparse("SEC-05-ZX11 是什么")[0].id == "c2"
    assert store.search_sparse("！？") == []
    assert build_match_query('"quoted"') == '"quoted"'


def test_hybrid_fuses_dense_and_sparse(store):
    hits = store.search_hybrid("年假审批", [0, 0, 1, 0], limit=2)

    assert hits[0].id == "c3"
    assert all(hit.retrieval_method == "hybrid_rrf" for hit in hits)


def test_document_listing_and_delete(store):
    assert store.list_documents() == [
        {"doc_id": DOC_A, "source": "policy.txt"},
        {"doc_id": DOC_B, "source": "hr.md"},
    ]
    assert store.document_ids_for_source("hr.md") == {DOC_B}
    assert store.has_document(DOC_A)

    store.delete_document(DOC_A)

    assert not store.has_document(DOC_A)
    assert store.count() == 1
    assert store.search_sparse("报销发票") == []
    assert [hit.id for hit in store.search_dense([1, 0, 0, 0])] == ["c3"]
    with pytest.raises(ValueError):
        store.delete_document("not-a-doc-id")


def test_reinserting_chunk_id_replaces_it(store):
    store.insert([record("c1", DOC_A, "会议室预订规则", [1, 0, 0, 0])])

    assert store.count() == 3
    assert store.search_sparse("报销")[:1] == []
    assert store.search_sparse("会议室")[0].id == "c1"


def test_dimension_mismatch_is_rejected(store, tmp_path):
    with pytest.raises(ValueError):
        store.insert([record("bad", DOC_A, "x", [1, 0])])

    other = StoreSettings(tmp_path)
    other.embedding_dimension = 8
    reopened = EmbeddedStore(other, db_path=store.db_path)
    with pytest.raises(ValueError):
        reopened.ensure_collection()
    reopened.close()
