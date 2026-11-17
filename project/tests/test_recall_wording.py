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


# ---------------------------------------------------------------- survey questions
from rag_agent.answering import PROFILES, classify_question, retrieval_limits  # noqa: E402
from retrieval.selection import SelectionPolicy  # noqa: E402
from retrieval.survey import Survey, has_shape, select_survey, survey_question  # noqa: E402


def test_survey_questions_are_recognised_by_their_wording():
    assert survey_question("一般的创新项目申报能给多少补贴？") == Survey("amount", "一般")
    assert survey_question("创新项目通常什么时候截止申报").ask == "time"
    assert survey_question("各个创新项目的补贴额度分别是多少？").ask == "amount"
    assert survey_question("人才方面的政策通常有哪些门槛？").ask == "who"
    assert survey_question("有哪些创业扶持，分别给多少钱？") is not None
    for question in (
        "澜州的青年创新项目最多能拿多少补贴？",  # one programme
        "广东省和河南省分别实施了哪些措施？",  # named things: the multi-part path
        "《差旅制度》里一般怎么报销？",  # a named file
        "一般纳税人的税率是多少？",  # a term, not a generality
        "一般来说应该注意身体。",  # not a question
        "KV Cache 的原理是什么？",
        "纪念币有哪些",  # a plain list
        "面试题里有哪些关于模型思考能力的题目？",  # a list from one file
        "产品线编号 P-039 的年度收入和毛利率分别是多少？",  # one identified thing
        "在2023年7月10日的检查中，核查组在兰州新区一般化工园区的4家企业中共发现了多少项问题隐患？",  # long and specific
    ):
        assert survey_question(question) is None, question


def test_answer_shapes():
    assert has_shape("单个项目资助资金不超过6万元", "amount") and has_shape("按实际投入的30%给予扶持", "amount")
    assert not has_shape("补贴金额按照正文的规定执行，以最终批复文件为准", "amount")
    assert has_shape("须于5月31日前提交", "time") and has_shape("自受理之日起15个工作日内", "time")
    assert has_shape("一般停牌5个交易日", "time") and has_shape("保质期半年", "time")
    assert not has_shape("结束时间以正式文件为准", "time")
    assert has_shape("任何文字", "who") and has_shape("任何文字", "")


def scored(n, doc, score, text):
    return SearchHit(id=str(n), score=score, text=text, doc_id=doc, parent_id=f"{doc}-p0", source=f"{doc}.txt",
                     heading="", locator="", retrieval_method="test", rerank_score=score)


SURVEY_POLICY = SelectionPolicy(min_score=0.1, relative=0.1, min_keep=3, max_keep=10, per_document=2, no_evidence_below=0.1)
RANKED = [
    scored(1, "faq", 0.9, "问：能拿多少补贴？答：按正文规定执行"),  # echoes the question, states nothing
    scored(2, "collection", 0.8, "稳岗补贴每人每月500元"),
    scored(3, "collection", 0.7, "见习补贴每人每月1000元"),
    scored(4, "collection", 0.6, "租房补贴每户最高3000元"),
    scored(5, "form", 0.3, "申请财政补贴6.0万元"),
    scored(6, "report", 0.2, "平均每个项目补贴约3.4万元"),
    scored(7, "guide_a", 0.08, "单个项目资助资金不超过6万元"),
    scored(8, "guide_b", 0.05, "每项安排财政科研经费20万元"),
    scored(9, "guide_c", 0.02, "按实际投入的30%给予扶持，最高200万元"),
    scored(10, "guide_d", 0.005, "每项1.5万元"),
]
TYPES = {"guide_a": "说明书", "guide_b": "制度规范", "guide_c": "说明书", "guide_d": "通知公告", "form": "申报书", "report": "工作报告"}


def test_survey_selection_keeps_one_answer_per_file_rule_files_first():
    kept = select_survey(RANKED, SURVEY_POLICY, Survey("amount", "一般"), files=5, floor=0.01, types=TYPES)
    docs = [h.doc_id for h in kept]
    # The FAQ has no amount; the two best files stay; then the guides come before the form and the report.
    assert "faq" not in docs and docs.count("collection") == 2  # its second passage fills a free place
    assert {"collection", "form", "guide_a", "guide_b", "guide_c"} == set(docs)
    assert "report" not in docs and "guide_d" not in docs  # past the five files / below the floor
    assert [h.rerank_score for h in kept] == sorted((h.rerank_score for h in kept), reverse=True)
    # Without types the order is the reranker's.
    untyped = select_survey(RANKED, SURVEY_POLICY, Survey("amount", "一般"), files=5, floor=0.01)
    assert {"collection", "form", "report", "guide_a", "guide_b"} == {h.doc_id for h in untyped}


def test_survey_selection_without_evidence_or_shape():
    assert select_survey([scored(1, "a", 0.05, "5万元")], SURVEY_POLICY, Survey("amount", "一般")) == []
    # No passage looks like an answer: the reranker's order is all there is.
    kept = select_survey([scored(1, "a", 0.9, "没有数字"), scored(2, "b", 0.5, "也没有")], SURVEY_POLICY, Survey("amount", "一般"))
    assert [h.id for h in kept] == ["1", "2"]


class SurveyStore:
    def __init__(self):
        self.limits = []

    def search_hybrid(self, query, vector, limit, doc_ids=None):
        self.limits.append(limit)
        return [SearchHit(id=h.id, score=0.0, text=h.text, doc_id=h.doc_id, parent_id=h.parent_id, source=h.source,
                          heading="", locator="", retrieval_method="test") for h in RANKED][:limit]

    def doc_cards(self, doc_ids):
        return {doc_id: {"doc_type": TYPES.get(doc_id, "")} for doc_id in doc_ids}


class SurveyReranker:
    def rerank(self, query, documents, top_n):
        by_text = {h.text: h.rerank_score for h in RANKED}
        return sorted((RerankItem(index=i, score=by_text[t]) for i, t in enumerate(documents)), key=lambda item: -item.score)[:top_n]


def test_the_retriever_takes_the_survey_route_only_when_switched_on():
    question = "一般的创新项目申报能给多少补贴？"
    on = Settings(wording_expansion=False, survey_route=True, survey_pool=10, survey_candidates_per_document=3, survey_files=5)
    store = SurveyStore()
    kept = HybridRetriever(store, Embeddings(), SurveyReranker(), on).retrieve(question, **retrieval_limits("survey"))
    assert {"guide_a", "guide_b", "guide_c"} <= {h.doc_id for h in kept} and "faq" not in {h.doc_id for h in kept}
    assert store.limits[0] == 10  # the survey pool, not fusion_top_k
    assert multipart.last_trace().survey["ask"] == "amount"
    off = HybridRetriever(SurveyStore(), Embeddings(), SurveyReranker(), Settings(wording_expansion=False))
    usual = off.retrieve(question, **retrieval_limits("fact"))
    assert usual[0].doc_id == "faq" and not multipart.last_trace().survey


def test_the_survey_profile_is_chosen_only_with_the_switch():
    question = "各个创新项目的补贴额度分别是多少？"
    assert classify_question(question) == "compare" and classify_question(question, survey=True) == "survey"
    assert classify_question("纳米硅的粒径是多少？", survey=True) == "fact"
    profile = PROFILES["survey"]
    assert profile.max_passages == 10 and profile.per_document == 2 and profile.parent_context is False
    assert all(PROFILES[name].parent_context for name in ("fact", "multi", "explain", "compare"))


def test_rule_file_names():
    from retrieval.survey import rule_name

    for name in ("青屿市博士后科研资助工作指引.docx", "望海市外贸企业开拓国际市场项目申报须知.docx", "1、政策集合.pdf",
                 "青屿市科技创新专项项目管理办法（2025年修订）.docx", "临溪市首台（套）重大技术装备推广应用指导意见.docx"):
        assert rule_name(name), name
    for name in ("2026年澜州青年创新项目申请指南常见问题解答.txt", "管理办法年度工作总结.txt", "01 项目申报书.docx", "周报-01.txt"):
        assert not rule_name(name), name


# ---------------------------------------------------------------- scored in the documents' words too
class TwoWayReranker:
    """Against the question as asked the guide scores low; against the expanded question, high."""

    def __init__(self):
        self.queries = []

    def rerank(self, query, documents, top_n):
        self.queries.append(query)
        expanded = "资助" in query
        scores = [(0.9 if expanded else 0.05) if "资助资金" in text else 0.3 for text in documents]
        return sorted((RerankItem(index=i, score=s) for i, s in enumerate(scores)), key=lambda item: -item.score)[:top_n]


def test_rescoring_keeps_the_better_of_the_two_scores():
    reranker = TwoWayReranker()
    kept = HybridRetriever(PolicyStore(), Embeddings(), reranker, Settings(wording_rescore=True)).retrieve(QUESTION, max_keep=6)
    assert kept[0].doc_id == "guide" and kept[0].rerank_score == 0.9
    assert sorted(reranker.queries, key=len)[0] == QUESTION and len(reranker.queries) == 2
    assert [call["query"] for call in multipart.last_trace().rerank_calls].count("wording") == 1
    # Off: one request, and the guide stays below the collection.
    reranker = TwoWayReranker()
    kept = HybridRetriever(PolicyStore(), Embeddings(), reranker, Settings()).retrieve(QUESTION, max_keep=6)
    assert reranker.queries == [QUESTION] and kept[0].doc_id == "collection"


def test_a_failed_second_scoring_leaves_the_first():
    class Failing(TwoWayReranker):
        def rerank(self, query, documents, top_n):
            if "资助" in query:
                raise RuntimeError("boom")
            return super().rerank(query, documents, top_n)

    kept = HybridRetriever(PolicyStore(), Embeddings(), Failing(), Settings(wording_rescore=True)).retrieve(QUESTION, max_keep=6)
    assert kept[0].doc_id == "collection"


# ---------------------------------------------------------------- the survey answer reads the passages themselves
def test_survey_context_uses_the_passages_not_their_parents():
    from rag_agent.nodes import AgentNodes

    class Parents:
        def get(self, parent_id):
            return {"text": "整份文件的一大段" * 50, "parent_id": parent_id}

    nodes = AgentNodes(llm=None, retriever=None, settings=SimpleNamespace(context_token_budget=9000, dated_evidence=False),
                       parent_store=Parents())
    chunks = [{"id": "1", "parent_id": "a-p0", "doc_id": "a", "source": "a.docx", "text": "单个项目资助资金不超过6万元"},
              {"id": "2", "parent_id": "a-p0", "doc_id": "a", "source": "a.docx", "text": "受理时间为3月2日至4月17日"}]
    survey = nodes.build_context({"retrieve_chunks": chunks, "answer_profile": "survey", "question": "一般能给多少补贴"})["context_docs"]
    assert [d["text"] for d in survey] == [c["text"] for c in chunks]  # both passages of one parent, as they are
    fact = nodes.build_context({"retrieve_chunks": chunks, "answer_profile": "fact", "question": "能给多少补贴"})["context_docs"]
    assert len(fact) == 1 and fact[0]["text"].startswith("整份文件")


def test_rescoring_does_not_turn_no_evidence_into_evidence():
    class Nothing(TwoWayReranker):
        def rerank(self, query, documents, top_n):
            self.queries.append(query)
            score = 0.9 if "资助" in query else 0.03
            return [RerankItem(index=i, score=score) for i in range(len(documents))][:top_n]

    kept = HybridRetriever(PolicyStore(), Embeddings(), Nothing(), Settings(wording_rescore=True)).retrieve(QUESTION, max_keep=6)
    assert kept == []
