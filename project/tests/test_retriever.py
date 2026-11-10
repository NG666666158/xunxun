from retrieval.hybrid_retriever import HybridRetriever
from retrieval.types import SearchHit


class RetrieverSettings:
    rerank_top_k = 4
    fusion_top_k = 6
    candidate_auto_scale = True
    candidate_pool_max = 80
    rerank_min_keep = 3
    rerank_min_score = 0.05
    rerank_relative_score = 0.1
    rerank_per_document = 0
    rrf_k = 60


def hit(identifier, method):
    return SearchHit(
        id=identifier,
        score=0.5,
        text=f"text-{identifier}",
        doc_id="a" * 24,
        parent_id=f"{'a' * 24}-p0",
        source="policy.txt",
        heading="",
        locator="",
        retrieval_method=method,
    )


class FailingHybridStore:
    def search_hybrid(self, query, vector, limit):
        raise ConnectionError("hybrid unavailable")

    def search_dense(self, vector, limit):
        return [hit("shared", "dense"), hit("dense", "dense")]

    def search_sparse(self, query, limit):
        return [hit("shared", "bm25"), hit("sparse", "bm25")]


class Embeddings:
    def embed_query(self, query):
        return [0.1, 0.2]


def test_hybrid_failure_degrades_to_local_rrf():
    retriever = HybridRetriever(
        FailingHybridStore(), Embeddings(), object(), RetrieverSettings()
    )
    results = retriever.retrieve("报销制度", mode="hybrid", top_k=3)

    assert results[0].id == "shared"
    assert all(item.retrieval_method == "fallback_rrf" for item in results)


class BrokenEmbeddings:
    def embed_query(self, query):
        raise TimeoutError("embedding unavailable")


def test_embedding_failure_degrades_to_bm25():
    retriever = HybridRetriever(
        FailingHybridStore(), BrokenEmbeddings(), object(), RetrieverSettings()
    )
    results = retriever.retrieve("NX-MEET-PRO", mode="hybrid", top_k=2)

    assert [item.id for item in results] == ["shared", "sparse"]
    assert all(item.retrieval_method == "bm25_degraded" for item in results)


def test_comparison_questions_search_the_topic_without_document_names():
    from retrieval.hybrid_retriever import comparison_topic

    assert comparison_topic("《AI Agent 实战》和《图解大模型》对RAG技术的描述侧重点有何不同？") == "RAG技术"
    assert comparison_topic("《AI Agent 实战》和《图解大模型》在强化学习训练方法上有何不同侧重点？") == "强化学习训练方法"
    assert comparison_topic("ReAct 循环包含哪三个环节？") is None


def test_named_sources_and_diversify_keep_every_named_document():
    from retrieval.hybrid_retriever import diversify, named_sources
    from retrieval.types import SearchHit

    sources = {"AI-Agents-in-Depth-zh-CN.pdf", "图解大模型：生成式AI原理与实战.pdf", "【配套资料】大模型面试题200问.pdf"}
    named = named_sources("《AI Agent 实战》和《图解大模型》有何不同？", sources)
    assert set(named) == {"AI-Agents-in-Depth-zh-CN.pdf", "图解大模型：生成式AI原理与实战.pdf"}

    def hit(n, source, score):
        return SearchHit(id=str(n), score=score, text="", doc_id="d", parent_id="p", source=source, heading="",
                         locator="", retrieval_method="test", rerank_score=score)

    ranked = [hit(i, "图解大模型：生成式AI原理与实战.pdf", 0.9 - i * 0.01) for i in range(6)]
    ranked += [hit(9, "【配套资料】大模型面试题200问.pdf", 0.5), hit(10, "AI-Agents-in-Depth-zh-CN.pdf", 0.02)]
    top = diversify(ranked, 5, named)
    assert len(top) == 5 and {h.source for h in top} == set(named)  # the weak but named book is kept
    assert [h.rerank_score for h in top] == sorted((h.rerank_score for h in top), reverse=True)


class ScoredReranker:
    """Scores each document by the number in its text ("text-<id>" -> scores[id])."""

    def __init__(self, scores):
        self.scores = scores
        self.top_n = None

    def rerank(self, query, documents, top_n):
        from services.model_clients import RerankItem

        self.top_n = top_n
        items = [RerankItem(index=i, score=self.scores[text.removeprefix("text-")]) for i, text in enumerate(documents)]
        return sorted(items, key=lambda item: item.score, reverse=True)[:top_n]


class PoolStore:
    def __init__(self, hits):
        self.hits = hits
        self.limits = []

    def search_hybrid(self, query, vector, limit):
        self.limits.append(limit)
        return self.hits[:limit]

    def count(self):
        return 1000


def scored_hits(scores, docs=None):
    hits = []
    for name in scores:
        item = hit(name, "hybrid_rrf")
        item.doc_id = (docs or {}).get(name, "a" * 24)
        hits.append(item)
    return hits


def test_rerank_keeps_hits_above_the_cutoffs_up_to_the_question_types_maximum():
    scores = {"1": 0.95, "2": 0.6, "3": 0.2, "4": 0.04, "5": 0.01, "6": 0.005}
    reranker = ScoredReranker(scores)
    store = PoolStore(scored_hits(scores))
    retriever = HybridRetriever(store, Embeddings(), reranker, RetrieverSettings())
    # cutoff = max(0.05, 0.1 * 0.95): "4" and below drop out; every candidate is scored.
    assert [h.id for h in retriever.retrieve("报销")] == ["1", "2", "3"]
    assert reranker.top_n == 6 and store.limits[-1] == 6
    assert [h.id for h in retriever.retrieve("报销", max_keep=2)] == ["1", "2"]
    # An explicit top_k keeps exactly that many (benchmarks and callers that ask for k).
    assert [h.id for h in retriever.retrieve("报销", top_k=5)] == ["1", "2", "3", "4", "5"]


def test_weak_results_still_keep_min_keep_for_the_relevance_gate():
    scores = {"1": 0.03, "2": 0.02, "3": 0.01, "4": 0.005}
    retriever = HybridRetriever(PoolStore(scored_hits(scores)), Embeddings(), ScoredReranker(scores), RetrieverSettings())
    assert [h.id for h in retriever.retrieve("报销")] == ["1", "2", "3"]


def test_pool_grows_with_the_library_but_not_past_the_cap():
    class Settings(RetrieverSettings):
        fusion_top_k = 40
        candidate_pool_max = 60

    store = PoolStore([])
    retriever = HybridRetriever(store, Embeddings(), object(), Settings())
    assert retriever.candidate_pool() == 40
    store.count = lambda: 40000
    retriever._count = None
    assert retriever.candidate_pool() == 50
    store.count = lambda: 10_000_000
    retriever._count = None
    assert retriever.candidate_pool() == 60
    Settings.candidate_auto_scale = False
    assert retriever.candidate_pool() == 40


def test_older_versions_are_dropped_when_the_newest_is_retrieved():
    old, new = "b" * 24, "c" * 24
    scores = {"1": 0.9, "2": 0.85, "3": 0.8}
    files = {old: ("报销制度_v1.docx", 1.0), new: ("报销制度_v2.docx", 2.0)}
    store = PoolStore(scored_hits(scores, {"1": old, "2": new, "3": "a" * 24}))
    retriever = HybridRetriever(store, Embeddings(), ScoredReranker(scores), RetrieverSettings(), files.get)
    assert [h.id for h in retriever.retrieve("报销")] == ["2", "3"]


def test_comparison_keeps_each_named_document_under_dynamic_truncation():
    scores = {"1": 0.95, "2": 0.9, "3": 0.85, "4": 0.03}
    hits = scored_hits(scores, {"4": "b" * 24})
    for item in hits:
        item.source = "图解大模型.pdf" if item.doc_id == "a" * 24 else "AI-Agents-实战.pdf"

    class Store(PoolStore):
        def list_documents(self):
            return [{"source": "图解大模型.pdf"}, {"source": "AI-Agents-实战.pdf"}]

    retriever = HybridRetriever(Store(hits), Embeddings(), ScoredReranker(scores), RetrieverSettings())
    kept = retriever.retrieve("《图解大模型》和《AI-Agents 实战》对 RAG 的描述有何不同？")
    # "4" scores below the floor but is the only passage of a named book.
    assert {h.source for h in kept} == {"图解大模型.pdf", "AI-Agents-实战.pdf"}
    assert "4" in [h.id for h in kept] and len(kept) <= RetrieverSettings.rerank_top_k


class FloorSettings(RetrieverSettings):
    rerank_no_evidence_below = 0.1


def test_no_evidence_when_even_the_best_passage_scores_below_the_floor():
    weak = {"1": 0.09, "2": 0.05, "3": 0.01}
    retriever = HybridRetriever(PoolStore(scored_hits(weak)), Embeddings(), ScoredReranker(weak), FloorSettings())
    # Nothing, rather than the rerank_min_keep weakest-of-all passages.
    assert retriever.retrieve("报销") == []
    # An explicit top_k still returns the best k (callers that rank rather than answer).
    assert [h.id for h in retriever.retrieve("报销", top_k=2)] == ["1", "2"]
    # A best passage at the floor is evidence, and rerank_min_keep applies as before.
    some = {"1": 0.1, "2": 0.05, "3": 0.01}
    retriever = HybridRetriever(PoolStore(scored_hits(some)), Embeddings(), ScoredReranker(some), FloorSettings())
    assert [h.id for h in retriever.retrieve("报销")] == ["1", "2", "3"]


def test_a_comparison_without_evidence_returns_nothing():
    scores = {"1": 0.08, "2": 0.03, "3": 0.01}
    retriever = HybridRetriever(PoolStore(scored_hits(scores)), Embeddings(), ScoredReranker(scores), FloorSettings())
    assert retriever.retrieve("报销和差旅有什么区别") == []
    strong = {"1": 0.8, "2": 0.03, "3": 0.01}
    retriever = HybridRetriever(PoolStore(scored_hits(strong)), Embeddings(), ScoredReranker(strong), FloorSettings())
    assert [h.id for h in retriever.retrieve("报销和差旅有什么区别")][0] == "1"


def test_a_question_type_can_set_its_own_per_document_cap():
    scores = {"1": 0.9, "2": 0.85, "3": 0.8, "4": 0.7}
    store = PoolStore(scored_hits(scores, {"4": "b" * 24}))
    retriever = HybridRetriever(store, Embeddings(), ScoredReranker(scores), RetrieverSettings())
    assert [h.id for h in retriever.retrieve("报销", max_keep=3)] == ["1", "2", "3"]  # the setting: no cap
    assert [h.id for h in retriever.retrieve("报销", max_keep=3, per_document=2)] == ["1", "2", "4"]
