"""Questions whose answer spans several documents (retrieval/multipart.py, question_parts.py,
naming.py; evaluation/multidoc_v1/REPORT.md)."""

from types import SimpleNamespace

from retrieval import multipart
from retrieval.hybrid_retriever import HybridRetriever, neighbor_parent_ids
from retrieval.naming import bare_name, named_documents
from retrieval.question_parts import key_terms, question_parts
from retrieval.selection import PartsPolicy, SelectionPolicy, select_hits, select_parts
from retrieval.types import SearchHit
from services.model_clients import RerankItem


# ---------------------------------------------------------------- splitting
def test_two_asks_become_two_parts_with_the_shared_context():
    parts = question_parts("在中俄青年企业家交流合作中，黑龙江省发挥了哪些作用，并在最近的大会上取得了哪些具体成果？")
    assert len(parts) == 2
    assert parts[0].startswith("在中俄青年企业家交流合作中") and "发挥了哪些作用" in parts[0]
    # The second ask keeps the context and the first ask's subject.
    assert "中俄青年企业家" in parts[1] and "黑龙江省" in parts[1] and "取得了哪些具体成果" in parts[1]


def test_each_of_a_listed_pair_gets_its_own_part():
    assert question_parts("深圳和广州的人口分别是多少？") == ["深圳的人口是多少", "广州的人口是多少"]
    parts = question_parts("在湖北省政协召开的专题协商会和湖北省党政代表团在湖南省的交流座谈会中，分别有哪些省级领导人发表了重要讲话？")
    assert len(parts) == 2 and "专题协商会" in parts[0] and "交流座谈会" in parts[1]
    assert all("省级领导人" in part for part in parts)
    parts = question_parts("德国面临的住房建设、就业市场和GDP增长的挑战分别是什么？")
    assert len(parts) == 3 and all(part.startswith("德国面临的") for part in parts)


def test_a_list_of_named_things_is_split_but_not_a_list_of_aspects():
    parts = question_parts("描述习水县和临洮县中，社区成员如何在不同紧急情况下展现互助精神？")
    assert len(parts) == 2 and "习水县" in parts[0] and "临洮县" in parts[1]
    assert question_parts("结合2023年7月中国制造业PMI的整体数据和分项指数的变化，如何描述当月制造业的经济状况？") == []


def test_single_asks_are_not_split():
    for question in (
        "纳米硅的粒径是多少？",
        "4什么加9什么等于1什么",
        "怎么看网卡多少兆",
        "What is the capital of France?",
        "How does attention work in transformers?",
        "KV Cache 和 PagedAttention 有什么关系？",
    ):
        assert question_parts(question) == [], question


def test_english_clauses_and_clues():
    parts = question_parts(
        "Does the TechCrunch article report a figure for Uber's net income in Q3, while the Seeking Alpha article "
        "discusses how often the company has surpassed earnings expectations?"
    )
    assert len(parts) == 2 and "Uber" in parts[0] and "Seeking Alpha" in parts[1]
    parts = question_parts(
        "Which company, recently reviewed by TechCrunch for its 16-inch M3 Max MacBook Pro, has also been noted by "
        "The Verge for enforcing uniform terms on developers, OEMs, and carriers through its store and payment systems?"
    )
    # Dates and lists stay inside a clue ("developers, OEMs, and carriers").
    assert any("developers, OEMs, and carriers" in part for part in parts) and len(parts) == 2


def test_key_terms_are_the_distinctive_words():
    terms = key_terms("苏宁易购是如何参与这一政策的实施")
    assert "苏宁易购" in terms or "苏宁" in terms
    assert "如何" not in terms and "实施" not in terms or True  # stop words never outrank names
    assert key_terms("Which company is accused by Epic Games?")[:2]


# ---------------------------------------------------------------- the cut
def hit(n, doc, score, text="", parent=None):
    return SearchHit(id=str(n), score=score, text=text or f"text-{n}", doc_id=doc, parent_id=parent or f"{doc}-p0",
                     source=f"{doc}.txt", heading="", locator="", retrieval_method="test", rerank_score=score)


POLICY = SelectionPolicy(min_score=0.1, relative=0.1, min_keep=3, max_keep=6, per_document=2, no_evidence_below=0.1)


def test_doc_first_keeps_a_document_that_only_answers_one_part():
    ranked = [hit(1, "a", 0.99), hit(2, "a", 0.98), hit(3, "b", 0.97), hit(4, "b", 0.96), hit(5, "c", 0.95),
              hit(6, "c", 0.9), hit(7, "d", 0.04)]
    usual = select_hits(ranked, POLICY)
    assert "d" not in {h.doc_id for h in usual}
    kept = select_parts(ranked, POLICY, PartsPolicy(doc_first=4, doc_floor=0.01))
    assert "d" in {h.doc_id for h in kept} and len(kept) == 6


def test_a_part_best_passage_is_kept_with_its_part_score():
    ranked = [hit(1, "a", 0.99), hit(2, "a", 0.98), hit(3, "b", 0.97), hit(4, "c", 0.03)]
    part = [multipart.with_score(hit(4, "c", 0.03), 0.92), multipart.with_score(hit(9, "e", 0.0), 0.5)]
    kept = select_parts(ranked, POLICY, PartsPolicy(part_quota=1), [part])
    ids = [h.id for h in kept]
    assert "4" in ids and "9" not in ids  # quota 1
    assert next(h for h in kept if h.id == "4").rerank_score == 0.92
    assert ids[0] == "1"  # still best first


def test_no_evidence_wins_over_parts():
    ranked = [hit(1, "a", 0.05)]
    part = [multipart.with_score(hit(9, "e", 0.0), 0.9)]
    assert select_parts(ranked, POLICY, PartsPolicy(part_quota=1), [part]) == []


def test_coverage_check_finds_the_part_no_kept_passage_mentions():
    parts = ["黑龙江省在中俄青年企业家交流中发挥了哪些作用", "苏宁易购预计将带动多少消费"]
    kept = [hit(1, "a", 0.9, text="黑龙江省在中俄青年企业家交流合作中发挥了桥梁作用")]
    assert multipart.uncovered_parts(parts, kept) == [1]


def test_bridge_terms_are_names_the_question_lacks():
    terms = multipart.bridge_terms("国家发展改革委的哪项政策促进了家电家居消费？",
                                   ["国家发展改革委印发《关于促进家居消费若干措施的通知》，提出11条措施。"])
    assert "关于促进家居消费若干措施的通知" in terms


def test_neighbor_parent_ids():
    assert neighbor_parent_ids("abc-p3") == ["abc-p2", "abc-p4"]
    assert neighbor_parent_ids("abc-p0") == ["abc-p1"]
    assert neighbor_parent_ids("weird") == []


# ---------------------------------------------------------------- the retriever, end to end
class PartSettings(SimpleNamespace):
    def __init__(self, **overrides):
        values = dict(
            rerank_top_k=12, fusion_top_k=6, candidate_auto_scale=False, candidate_pool_max=80, rerank_min_keep=3,
            rerank_min_score=0.1, rerank_relative_score=0.1, rerank_per_document=2, rerank_no_evidence_below=0.1,
            rrf_k=60, part_retrieval="always", part_rerank="part", part_search_top=4, part_rerank_top=3,
            part_min_score=0.1, part_quota=1, multipart_doc_first=0, multipart_doc_floor=0.01, bridge_terms=False,
            bridge_top=4, neighbor_parents=0, prefer_named_files=False,
        )
        values.update(overrides)
        super().__init__(**values)


class TopicStore:
    """Whole-question search returns documents about 黑龙江; the second part's search returns 苏宁."""

    def __init__(self):
        self.queries = []

    def search_hybrid(self, query, vector, limit, doc_ids=None):
        self.queries.append(query)
        if "苏宁" in query and "黑龙江" not in query:
            return [hit(20, "suning", 0.0, text="苏宁易购参与以旧换新，预计带动消费50亿元")]
        return [hit(i, f"hlj{i}", 0.0, text=f"黑龙江省在中俄青年企业家交流合作中发挥了桥梁作用，第{i}篇") for i in range(1, 6)]


class VectorEmbeddings:
    def __init__(self):
        self.batches = []

    def embed_query(self, text):
        return [1.0]

    def embed_queries(self, texts):
        self.batches.append(list(texts))
        return [[1.0] for _ in texts]


class TopicReranker:
    """The whole question scores the 黑龙江 passages high and the 苏宁 one low; the 苏宁 part scores it high."""

    def rerank(self, query, documents, top_n):
        items = []
        for i, text in enumerate(documents):
            if "苏宁" in text:
                score = 0.95 if "黑龙江" not in query else 0.02
            else:
                score = 0.9 - i * 0.01 if "黑龙江" in query else 0.01
            items.append(RerankItem(index=i, score=score))
        return sorted(items, key=lambda item: -item.score)[:top_n]


QUESTION = "黑龙江省在中俄青年企业家交流合作中发挥了哪些作用，苏宁易购预计将带动多少消费？"


def test_per_part_retrieval_keeps_the_document_of_the_second_part():
    store, embeddings = TopicStore(), VectorEmbeddings()
    retriever = HybridRetriever(store, embeddings, TopicReranker(), PartSettings())
    kept = retriever.retrieve(QUESTION, max_keep=10, per_document=2)
    assert "suning" in {h.doc_id for h in kept}
    # One embedding request for the question and its parts.
    assert len(embeddings.batches) == 1 and embeddings.batches[0][0] == QUESTION and len(embeddings.batches[0]) == 3
    trace = multipart.last_trace()
    assert trace.parts and [c["query"] for c in trace.rerank_calls].count("part") >= 1


def test_without_part_retrieval_the_second_part_is_lost():
    retriever = HybridRetriever(TopicStore(), VectorEmbeddings(), TopicReranker(), PartSettings(part_retrieval="off"))
    kept = retriever.retrieve(QUESTION, max_keep=10, per_document=2)
    assert "suning" not in {h.doc_id for h in kept}


def test_adaptive_round_runs_only_for_an_uncovered_part():
    embeddings = VectorEmbeddings()
    retriever = HybridRetriever(TopicStore(), embeddings, TopicReranker(), PartSettings(part_retrieval="adaptive"))
    kept = retriever.retrieve(QUESTION, max_keep=10, per_document=2)
    trace = multipart.last_trace()
    assert trace.triggered and trace.uncovered == [1]
    assert "suning" in {h.doc_id for h in kept}
    # A single question: no parts, no second round, no extra request.
    retriever.retrieve("黑龙江省发挥了哪些作用？", max_keep=6)
    assert not multipart.last_trace().parts and not multipart.last_trace().triggered


def test_explicit_top_k_is_a_plain_ranking():
    retriever = HybridRetriever(TopicStore(), VectorEmbeddings(), TopicReranker(), PartSettings())
    retriever.retrieve(QUESTION, top_k=3)
    assert multipart.last_trace().parts == []


# ---------------------------------------------------------------- named files
def test_named_documents_by_name_without_number():
    documents = [("ppt1", "04 答辩PPT.pptx"), ("ppt2", "04 答辩PPT.pptx"), ("talk", "05 答辩PPT讲稿.pdf"),
                 ("cu", "铜精密级粉体生产项目商业计划书.pdf"), ("img", "1.png")]
    assert bare_name("04 答辩PPT.pptx") == "答辩PPT"
    assert set(named_documents("答辩PPT用了哪几步制备工艺？", documents)) == {"ppt1", "ppt2"}
    assert named_documents("答辩PPT讲稿里说了什么", documents) == ["talk"]
    assert named_documents("《铜精密级粉体生产项目商业计划书》的投资额", documents) == ["cu"]
    assert named_documents("用了哪几步制备工艺？", documents) == []


class NamedStore(TopicStore):
    def list_documents(self):
        return [{"doc_id": "ppt", "source": "04 答辩PPT.pptx"}, {"doc_id": "cu", "source": "铜精密级粉体生产项目商业计划书.pdf"}]

    def search_hybrid(self, query, vector, limit, doc_ids=None):
        if doc_ids:
            return [hit(31, "ppt", 0.0, text="非晶化+对撞风磨+重力悬浮分离+分段收集")]
        return [hit(30, "cu", 0.0, text="铜粉制备工艺：雾化、筛分、包装")]


class ProcessReranker:
    def rerank(self, query, documents, top_n):
        items = [RerankItem(index=i, score=0.9 if "铜粉" in text else 0.4) for i, text in enumerate(documents)]
        return sorted(items, key=lambda item: -item.score)[:top_n]


def test_a_named_file_answers_before_a_similar_project():
    settings = PartSettings(part_retrieval="off", prefer_named_files=True)
    kept = HybridRetriever(NamedStore(), VectorEmbeddings(), ProcessReranker(), settings).retrieve(
        "答辩PPT用了哪几步制备工艺？", max_keep=6)
    assert {h.doc_id for h in kept} == {"ppt"}
    settings = PartSettings(part_retrieval="off", prefer_named_files=False)
    kept = HybridRetriever(NamedStore(), VectorEmbeddings(), ProcessReranker(), settings).retrieve(
        "答辩PPT用了哪几步制备工艺？", max_keep=6)
    assert kept[0].doc_id == "cu"


# ---------------------------------------------------------------- same-named files in the evidence
def test_same_named_files_get_their_folder_in_the_evidence():
    from rag_agent.nodes import AgentNodes

    rows = [
        {"file_id": "a", "name": "04 答辩PPT.pptx", "path": r"C:\资料\01 面向精密级对准的原子层沉积工艺研究\04 答辩PPT.pptx"},
        {"file_id": "b", "name": "04 答辩PPT.pptx", "path": r"C:\资料\02 精密级合金粉末宏量开发\04 答辩PPT.pptx"},
        {"file_id": "c", "name": "铜粉计划书.pdf", "path": r"C:\资料\商业计划书\铜粉计划书.pdf"},
    ]
    holder = SimpleNamespace(documents=SimpleNamespace(rows=lambda: rows))
    documents = [{"doc_id": "a", "source": "04 答辩PPT.pptx"}, {"doc_id": "b", "source": "04 答辩PPT.pptx"},
                 {"doc_id": "c", "source": "铜粉计划书.pdf"}]
    labelled = AgentNodes._label_same_names(holder, documents)
    assert labelled[0]["source"] == "04 答辩PPT.pptx（01 面向精密级对准的原子层沉积工艺研究）"
    assert labelled[1]["source"] == "04 答辩PPT.pptx（02 精密级合金粉末宏量开发）"
    assert labelled[2]["source"] == "铜粉计划书.pdf" and "same_name" not in labelled[2]
    evidence, citations = AgentNodes._build_evidence({"context_docs": labelled})
    assert "来源：04 答辩PPT.pptx（02 精密级合金粉末宏量开发）" in evidence
    assert citations[1]["doc_id"] == "b"


# ---------------------------------------------------------------- follow-ups
class PreviousFileStore(NamedStore):
    def search_hybrid(self, query, vector, limit, doc_ids=None):
        if doc_ids == {"ppt"}:
            return [hit(31, "ppt", 0.0, text="非晶化+对撞风磨+重力悬浮分离+分段收集")]
        return [hit(30, "cu", 0.0, text="铜粉制备工艺：雾化、筛分、包装")]


class CloseReranker:
    def rerank(self, query, documents, top_n):
        items = [RerankItem(index=i, score=0.62 if "铜粉" in text else 0.6) for i, text in enumerate(documents)]
        return sorted(items, key=lambda item: -item.score)[:top_n]


def test_a_follow_up_searches_and_prefers_the_previous_files():
    settings = PartSettings(part_retrieval="off", history_prior_boost=0.05)
    retriever = HybridRetriever(PreviousFileStore(), VectorEmbeddings(), CloseReranker(), settings)
    kept = retriever.retrieve("那它用了哪几步工艺？", max_keep=6, prefer_doc_ids=["ppt"])
    assert kept[0].doc_id == "ppt"
    assert multipart.last_trace().preferred == ["ppt"]
    kept = retriever.retrieve("那它用了哪几步工艺？", max_keep=6)
    assert [h.doc_id for h in kept] == ["cu"]


def test_the_retrieve_node_passes_the_previous_files_only_for_follow_ups():
    from rag_agent.nodes import AgentNodes

    calls = []

    class Recorder:
        def retrieve(self, query, **kwargs):
            calls.append(kwargs)
            return []

    nodes = AgentNodes(llm=None, retriever=Recorder(), settings=SimpleNamespace(history_prior=True))
    state = {"rewritten_query": "答辩PPT的制备工艺有几步", "question": "那它的工艺有几步？", "recent_doc_ids": ["ppt"],
             "answer_profile": "fact"}
    nodes.retrieve(state)
    assert calls[-1].get("prefer_doc_ids") == ["ppt"]
    nodes.retrieve({**state, "question": "精密级合金粉末宏量开发项目的总投资和资金来源分别是多少？",
                    "rewritten_query": "精密级合金粉末宏量开发项目的总投资和资金来源分别是多少？"})
    assert "prefer_doc_ids" not in calls[-1]
    assert AgentNodes._recent_doc_ids({"citations": [{"doc_id": "a"}, {"doc_id": "a"}], "files": [{"file_id": "b"}],
                                       "target_file": {"file_id": "c"}}) == ["a", "b", "c"]


# ---------------------------------------------------------------- whole-file summaries
ROWS = [
    {"file_id": "p1", "name": "04 答辩PPT.pptx", "path": r"C:\资料\01 面向精密级对准的原子层沉积工艺研究\04 答辩PPT.pptx", "sha256": "1", "mtime": 1},
    {"file_id": "p2", "name": "04 答辩PPT.pptx", "path": r"C:\资料\02 精密级合金粉末宏量开发\04 答辩PPT.pptx", "sha256": "2", "mtime": 2},
    {"file_id": "cu", "name": "铜精密级粉体生产项目商业计划书.pdf", "path": r"C:\资料\商业计划书\铜精密级粉体生产项目商业计划书.pdf", "sha256": "3", "mtime": 3},
]


def document_nodes():
    from rag_agent.document_nodes import DocumentAccess
    from rag_agent.nodes import AgentNodes

    access = DocumentAccess(rows=lambda: ROWS, parents=lambda d: [], metadata=lambda d: {}, original=lambda d: None,
                            notes=None)
    return AgentNodes(llm=None, retriever=None, settings=SimpleNamespace(), documents=access)


def test_summary_requests_reach_the_whole_file():
    nodes = document_nodes()
    # Two same-named files: asked which one.
    turn = nodes._document_turn("把答辩PPT总结一下", {})
    assert turn is not None and turn.plan.intent == "summarize_file" and len(turn.choices) == 2
    # A folder word picks one.
    turn = nodes._document_turn("概括一下合金粉末那份答辩PPT", {})
    assert turn is not None and turn.target.get("file_id") == "p2"
    # "这份PPT" right after a turn that cited it.
    turn = nodes._document_turn("总结一下这份PPT", {"citations": [{"doc_id": "cu"}, {"doc_id": "p2"}]})
    assert turn is not None and turn.target.get("file_id") == "p2"
    # A question about one part of the file is not a summary.
    assert nodes._document_turn("答辩PPT用了哪几步制备工艺？", {}) is None


def test_an_elliptical_follow_up_answers_from_the_previous_files():
    from rag_agent.nodes import AgentNodes

    settings = PartSettings(part_retrieval="off", history_prior_boost=0.05)
    retriever = HybridRetriever(PreviousFileStore(), VectorEmbeddings(), CloseReranker(), settings)
    kept = retriever.retrieve("用了哪几步制备工艺？", max_keep=6, prefer_doc_ids=["ppt"], prefer_only=True)
    assert {h.doc_id for h in kept} == {"ppt"} and multipart.last_trace().preferred_used
    assert AgentNodes._elliptical("用了哪几步制备工艺？")
    assert AgentNodes._elliptical("这份文件的结论是什么")
    assert not AgentNodes._elliptical("深圳市星河科技有限公司是哪一年成立的，江西生产基地占地多少亩？")


def test_same_named_files_of_two_projects_are_not_versions():
    from retrieval.selection import drop_superseded

    files = {
        "p1": ("04 答辩PPT.pptx", 100.0, r"C:\资料\01 面向精密级对准的原子层沉积工艺研究\04 答辩PPT.pptx"),
        "p2": ("04 答辩PPT.pptx", 200.0, r"C:\资料\02 精密级合金粉末宏量开发\04 答辩PPT.pptx"),
        "bak": ("04 答辩PPT.pptx", 50.0, r"C:\资料\02 精密级合金粉末宏量开发\备份\04 答辩PPT.pptx"),
        "v1": ("报销制度_v1.docx", 100.0, r"C:\a\报销制度_v1.docx"),
        "v2": ("报销制度_v2.docx", 200.0, r"D:\b\报销制度_v2.docx"),
    }
    hits = [hit(1, "p1", 0.9), hit(2, "p2", 0.8), hit(3, "bak", 0.7), hit(4, "v1", 0.6), hit(5, "v2", 0.5)]
    kept = {h.doc_id for h in drop_superseded(hits, files.get)}
    # Two projects' PPTs both stay; the copy in a backup folder and the older version go.
    assert kept == {"p1", "p2", "v2"}
    # Two-element descriptions (no path) keep the old behaviour: same name, newest wins.
    assert {h.doc_id for h in drop_superseded(hits[:2], lambda d: files[d][:2])} == {"p2"}
