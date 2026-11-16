"""Questions in everyday words, documents in their own (retrieval/wording.py and the steps of
evaluation/recall_wording_v1/REPORT.md)."""

from types import SimpleNamespace

from retrieval import multipart, wording
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.types import SearchHit
from services.model_clients import RerankItem


# ---------------------------------------------------------------- the table
def test_a_money_question_gets_the_papers_words_for_money():
    expansion = wording.expand("一般的创新项目申报能给多少补贴")
    assert expansion is not None and expansion.groups == ("funding", "applying")
    for word in ("资助", "奖励", "扶持", "经费", "支持强度", "申请", "立项"):
        assert word in expansion.terms
    # Words the question already has are not repeated; the text starts with the question as asked.
    assert "补贴" not in expansion.terms and "申报" not in expansion.terms
    assert expansion.text.startswith("一般的创新项目申报能给多少补贴 ")
    # "多少" asks for an amount: how papers state one.
    assert "万元" in expansion.terms and "不超过" in expansion.terms


def test_everyday_ways_of_asking_for_money_count():
    for question in ("望海建一个技能大师工作室政府给多少钱？", "出国参展的摊位费能报销吗", "在北渚创业个人最多能贷多少"):
        expansion = wording.expand(question)
        assert expansion is not None and "funding" in expansion.groups, question


def test_deadline_eligibility_and_materials_only_beside_money_or_applying():
    deadline = wording.expand("澜州青年创新项目几号截止申报？")
    assert deadline.groups == ("applying", "deadline") and "期限" in deadline.terms and "受理时间" in deadline.terms
    who = wording.expand("这个补贴谁可以申请，有什么条件")
    assert "eligible" in who.groups and "对象" in who.terms and "资格" in who.terms
    assert "materials" in wording.expand("申报要交哪些材料").groups
    # The same asks about something else are left alone.
    for question in ("唐朝是什么时候建立的", "这个反应需要什么条件", "实验要准备哪些材料", "KV Cache 的原理是什么"):
        assert wording.expand(question) is None, question


def test_questions_outside_the_table_are_not_expanded():
    for question in ("", "What is the subsidy for startups?", "纳米硅的粒径是多少？", "x" * 300 + "补贴"):
        assert wording.expand(question) is None


def test_the_added_words_are_limited_and_shared_between_groups():
    expansion = wording.expand("谁能申请这个补贴，什么时候截止，要交什么材料，能拿多少钱", limit=10)
    assert len(expansion.terms) == 10
    # One word of each group before the second of any.
    assert set(expansion.terms[:6]) == {"资助", "申报", "期限", "条件", "对象", "资料"}


# ---------------------------------------------------------------- the retriever
def hit(n, doc, text):
    return SearchHit(id=str(n), score=0.0, text=text, doc_id=doc, parent_id=f"{doc}-p0", source=f"{doc}.txt",
                     heading="", locator="", retrieval_method="test")


class Settings(SimpleNamespace):
    def __init__(self, **overrides):
        values = dict(
            rerank_top_k=12, fusion_top_k=4, candidate_auto_scale=False, candidate_pool_max=80, rerank_min_keep=3,
            rerank_min_score=0.1, rerank_relative_score=0.1, rerank_per_document=4, rerank_no_evidence_below=0.1,
            rrf_k=60, part_retrieval="off", multipart_doc_first=0, bridge_terms=False, neighbor_parents=0,
            prefer_named_files=False, wording_expansion=True, wording_search_top=10, wording_top=2,
        )
        values.update(overrides)
        super().__init__(**values)


class PolicyStore:
    """The question as asked finds the collection that says 补贴; the expanded question finds the guide."""

    def __init__(self):
        self.queries = []

    def search_hybrid(self, query, vector, limit, doc_ids=None):
        self.queries.append(query)
        if "资助" in query:
            return [hit(1, "collection", "稳岗补贴申报"), hit(10, "guide", "单个项目资助资金不超过6万元"),
                    hit(11, "form", "申请资助金额"), hit(12, "other", "经费管理")][:limit]
        return [hit(i, "collection", f"第{i}项补贴申报") for i in range(1, 5)][:limit]


class Embeddings:
    def __init__(self):
        self.batches = []

    def embed_query(self, text):
        self.batches.append([text])
        return [1.0]

    def embed_queries(self, texts):
        self.batches.append(list(texts))
        return [[1.0] for _ in texts]


class Reranker:
    def __init__(self):
        self.queries = []

    def rerank(self, query, documents, top_n):
        self.queries.append(query)
        scores = [0.9 if "资助资金" in text else 0.3 for text in documents]
        return sorted((RerankItem(index=i, score=s) for i, s in enumerate(scores)), key=lambda item: -item.score)[:top_n]


QUESTION = "创新项目申报能给多少补贴"


def test_the_expanded_question_adds_candidates_scored_against_the_question_as_asked():
    store, embeddings, reranker = PolicyStore(), Embeddings(), Reranker()
    kept = HybridRetriever(store, embeddings, reranker, Settings()).retrieve(QUESTION, max_keep=6)
    assert kept[0].doc_id == "guide"
    # One embedding request for the question and its expansion; both searched; one rerank
    # request, with the question as the user asked it.
    assert len(embeddings.batches) == 1 and embeddings.batches[0][0] == QUESTION and len(embeddings.batches[0]) == 2
    assert sorted(store.queries, key=len)[0] == QUESTION and "资助" in sorted(store.queries, key=len)[1]
    assert reranker.queries == [QUESTION]
    trace = multipart.last_trace()
    assert trace.wording["groups"] == ["funding", "applying"]
    # wording_top new candidates, the first ones of the expanded search not already in the pool.
    assert trace.extra == ["10", "11"] and set(trace.extra_kinds.values()) == {"wording"}


def test_switched_off_or_outside_the_table_nothing_changes():
    for settings, question in ((Settings(wording_expansion=False), QUESTION), (Settings(), "纳米硅的粒径是多少")):
        store, embeddings = PolicyStore(), Embeddings()
        kept = HybridRetriever(store, embeddings, Reranker(), settings).retrieve(question, max_keep=6)
        assert store.queries == [question] and embeddings.batches == [[question]]
        assert {h.doc_id for h in kept} == {"collection"}
        assert not multipart.last_trace().wording


def test_an_explicit_top_k_or_a_plain_hybrid_search_is_not_expanded():
    store = PolicyStore()
    retriever = HybridRetriever(store, Embeddings(), Reranker(), Settings())
    retriever.retrieve(QUESTION, top_k=3)
    retriever.retrieve(QUESTION, mode="hybrid")
    assert store.queries == [QUESTION, QUESTION]


def test_a_failed_expanded_search_leaves_the_question_s_own_results():
    class Failing(PolicyStore):
        def search_hybrid(self, query, vector, limit, doc_ids=None):
            if "资助" in query:
                raise RuntimeError("boom")
            return super().search_hybrid(query, vector, limit, doc_ids)

    kept = HybridRetriever(Failing(), Embeddings(), Reranker(), Settings()).retrieve(QUESTION, max_keep=6)
    assert {h.doc_id for h in kept} == {"collection"}


# ---------------------------------------------------------------- a pool that a few files fill
from retrieval.selection import crowded, spread_documents  # noqa: E402


def test_spread_documents_caps_each_file_and_keeps_the_order():
    ranked = [hit(i, "big", f"t{i}") for i in range(6)] + [hit(10, "a", "ta"), hit(11, "b", "tb"), hit(12, "big", "t12")]
    assert crowded(ranked, 2) and not crowded(ranked, 7)
    kept = spread_documents(ranked, per_document=2, limit=5)
    # Two of the big file, the two other files, and the place left over goes back to the big file.
    assert [h.id for h in kept] == ["0", "1", "2", "10", "11"]
    assert spread_documents(ranked, 0, 3) == ranked[:3]


class CrowdedStore:
    """The first 4 candidates all come from one collection; a deeper search reaches the guide."""

    def __init__(self):
        self.limits = []

    def search_hybrid(self, query, vector, limit, doc_ids=None):
        self.limits.append(limit)
        ranked = [hit(i, "collection", f"第{i}项补贴申报") for i in range(1, 9)] + [hit(20, "guide", "单个项目资助资金不超过6万元")]
        ranked += [hit(30, "form", "申请资助金额")]
        return ranked[:limit]


def test_a_crowded_pool_is_searched_deeper_and_cut_per_file():
    store = CrowdedStore()
    settings = Settings(wording_expansion=False, candidate_per_document=2)
    kept = HybridRetriever(store, Embeddings(), Reranker(), settings).retrieve(QUESTION, max_keep=6)
    assert store.limits == [4, 16]
    assert kept[0].doc_id == "guide"
    assert multipart.last_trace().crowded == {"files": 1, "files_after": 3}


def test_a_pool_no_file_crowds_is_left_alone():
    store = CrowdedStore()
    settings = Settings(wording_expansion=False, candidate_per_document=4)
    kept = HybridRetriever(store, Embeddings(), Reranker(), settings).retrieve(QUESTION, max_keep=6)
    assert store.limits == [4] and {h.doc_id for h in kept} == {"collection"}
    # Off by default; never for an explicit top_k or inside one file.
    for options, arguments in ((dict(candidate_per_document=0), {}), (dict(candidate_per_document=2), {"top_k": 3}),
                               (dict(candidate_per_document=2), {"doc_ids": {"collection"}})):
        store = CrowdedStore()
        HybridRetriever(store, Embeddings(), Reranker(), Settings(wording_expansion=False, **options)).retrieve(QUESTION, **arguments)
        assert len(store.limits) == 1, (options, arguments)


def test_a_library_of_one_file_keeps_its_pool():
    class OneFile(CrowdedStore):
        def search_hybrid(self, query, vector, limit, doc_ids=None):
            self.limits.append(limit)
            return [hit(i, "book", f"第{i}段补贴") for i in range(1, 9)][:limit]

    store = OneFile()
    settings = Settings(wording_expansion=False, candidate_per_document=2)
    kept = HybridRetriever(store, Embeddings(), Reranker(), settings).retrieve(QUESTION, max_keep=6)
    assert store.limits == [4, 16] and [h.id for h in kept] == ["1", "2", "3", "4"]
    assert not multipart.last_trace().crowded


# ---------------------------------------------------------------- headings written as paragraphs
from ingestion.sections import Paragraph, heading_level, sectioned  # noqa: E402


def test_numbered_and_bold_headings_are_recognised():
    assert heading_level("三、支持强度与方式") == (1, "三、支持强度与方式")
    assert heading_level("第二章 申报条件") == (1, "第二章 申报条件")
    assert heading_level("（一）优质档") == (2, "（一）优质档")
    assert heading_level("联系方式", bold=True) == (3, "联系方式")
    # A heading that opens its own paragraph.
    assert heading_level("三、支持强度与方式。项目分三档给予支持，优质档不超过6万元。") == (1, "三、支持强度与方式")
    # Sentences, list items and provisions are not headings.
    for text in ("（一）申请人信用状况良好，未被列入严重失信主体名单。", "一、申请人应当在规定时间内提交材料，逾期不予受理。",
                 "第三条 本办法所称专项资金，是指市财政安排的资金。", "项目分三档给予支持。", "联系方式", "3.14"):
        assert heading_level(text) is None, text


GUIDE = [
    Paragraph("2026年青年创新项目申请指南", bold=True),
    Paragraph("现将指南印发给你们，请遵照执行。"),
    Paragraph("一、设定依据", bold=True),
    Paragraph("为贯彻落实高质量发展的工作部署，规范项目的组织实施，提高财政资金使用效益，依据有关规定，结合工作实际，制定本文件。"),
    Paragraph("二、申请内容", bold=True),
    Paragraph("（一）新一代信息技术领域的技术研发和应用示范。"),
    Paragraph("（二）面向社会治理和公共服务的创新性解决方案。"),
    Paragraph("三、支持强度与方式", bold=True),
    Paragraph("项目分三档给予支持。优质档：单个项目资助资金不超过6万元；潜力档：不超过3万元；起步档：不超过0.8万元。"),
]


def test_a_guide_is_cut_at_its_numbered_headings():
    units = sectioned(GUIDE)
    assert [unit.heading for unit in units] == ["", "一、设定依据", "二、申请内容", "三、支持强度与方式"]
    money = units[-1]
    # The heading line stays in the text; the amounts are the section's own unit.
    assert money.text.startswith("三、支持强度与方式\n项目分三档") and "申请内容" not in money.text
    assert "".join(unit.text.replace("\n", "") for unit in units) == "".join(p.text for p in GUIDE)


def test_files_without_such_headings_keep_their_usual_units():
    assert sectioned([Paragraph("第一段。" * 30), Paragraph("第二段。" * 30)]) is None
    # A numbered list is not a document's sections.
    assert sectioned([Paragraph("一、张三"), Paragraph("二、李四"), Paragraph("三、王五")]) is None
    # A single heading is not enough either.
    assert sectioned([Paragraph("一、总则"), Paragraph("正文。" * 50)]) is None


def test_a_long_section_is_cut_at_its_sub_headings_and_a_short_one_is_not():
    short = [Paragraph("一、总则"), Paragraph("总则内容。" * 20), Paragraph("二、支持标准"), Paragraph("（一）优质档"), Paragraph("不超过6万元。"),
             Paragraph("（二）潜力档"), Paragraph("不超过3万元。")]
    assert [unit.heading for unit in sectioned(short)] == ["一、总则", "二、支持标准"]
    long = short[:3] + [Paragraph("（一）优质档"), Paragraph("优质档的说明。" * 100), Paragraph("（二）潜力档"), Paragraph("潜力档的说明。" * 100)]
    assert [unit.heading for unit in sectioned(long)] == ["一、总则", "二、支持标准", "二、支持标准 > （一）优质档", "二、支持标准 > （二）潜力档"]


def test_word_heading_styles_still_lead():
    units = sectioned([Paragraph("第一部分", styled=1), Paragraph("一、目的"), Paragraph("目的内容。" * 20), Paragraph("二、范围"),
                       Paragraph("范围内容。" * 20), Paragraph("第二部分", styled=1), Paragraph("没有小标题的正文。" * 10)])
    assert [unit.heading for unit in units] == ["第一部分 > 一、目的", "第一部分 > 二、范围", "第二部分"]


def test_docx_and_txt_parsers_follow_the_switch(tmp_path):
    from docx import Document

    from ingestion.parsers import MultiFormatParser, parser_version

    path = tmp_path / "指南.docx"
    document = Document()
    for paragraph in GUIDE:
        document.add_paragraph().add_run(paragraph.text).bold = paragraph.bold
    document.save(path)
    text = tmp_path / "指南.txt"
    text.write_text("\n".join(paragraph.text for paragraph in GUIDE), encoding="utf-8")
    off = SimpleNamespace(max_local_file_mb=100, max_local_text_mb=50, section_headings=False, image_understanding="off")
    on = SimpleNamespace(max_local_file_mb=100, max_local_text_mb=50, section_headings=True, image_understanding="off")
    assert len(MultiFormatParser(off).parse(path).units) == 1 and len(MultiFormatParser(off).parse(text).units) == 1
    assert [unit.heading for unit in MultiFormatParser(on).parse(path).units][-1] == "三、支持强度与方式"
    assert [unit.heading for unit in MultiFormatParser(on).parse(text).units] == ["指南", "一、设定依据", "二、申请内容", "三、支持强度与方式"]
    # Switching it changes the index version of these types only, so they are read again.
    assert parser_version(path, on) == parser_version(path, off) + 1 and parser_version(text, on) == parser_version(text, off) + 1
    assert parser_version("a.md", on) == parser_version("a.md", off)
