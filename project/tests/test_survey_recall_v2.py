"""Survey questions, round two (evaluation/survey_recall_v2/REPORT.md): files first."""

from config import Settings
from rag_agent.answering import retrieval_limits
from retrieval import multipart
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.selection import SelectionPolicy
from retrieval.survey import Survey, select_by_files, survey_topics, topic_files
from retrieval.types import SearchHit
from services.model_clients import RerankItem


def hit(n, doc, text, score=0.0):
    return SearchHit(id=str(n), score=score, text=text, doc_id=doc, parent_id=f"{doc}-p0", source=f"{doc}.docx",
                     heading="", locator="", retrieval_method="test")


# ---------------------------------------------------------------- the topic
def test_the_topic_is_the_question_without_its_survey_wording():
    assert survey_topics("一般的创新项目申报能给多少补贴？") == ["创新"]
    assert survey_topics("创业政策通常要满足哪些要求？") == ["创业"]
    assert survey_topics("考技能证书一般能领多少补贴？") == ["考 技能 证书"]
    # Listed topics stay apart: a passage about one of them is about the topic.
    assert survey_topics("参展、办展、跨境电商一般各补贴多少？") == ["参展", "办展", "跨境 电商"]
    assert survey_topics("技术改造和绿色工厂这类项目都有什么门槛？") == ["技术改造", "绿色 工厂"]
    assert len(survey_topics("甲、乙、丙、丁一般各补贴多少？")) == 3
    assert survey_topics("一般能给多少补贴？") == []


def test_topic_files_weigh_a_files_two_best_passages():
    hits = [hit(1, "a", "", 0.05), hit(2, "b", "", 0.04), hit(3, "b", "", 0.03), hit(4, "c", "", 0.02), hit(5, "b", "", 0.9)]
    # b: 0.9 + 0.04 (its third passage does not count); a: 0.05.
    assert topic_files(hits, 2) == ["b", "a"]


# ---------------------------------------------------------------- choosing files, then passages
POLICY = SelectionPolicy(min_score=0.1, relative=0.1, min_keep=3, max_keep=10, per_document=2, no_evidence_below=0.1)
POOL = [
    hit(1, "guide_a", "一、设定依据……二、申请主体：18至35周岁的青年"),
    hit(2, "guide_a", "四、支持强度：单个项目资助资金不超过6万元"),
    hit(3, "guide_b", "经费安排：每项安排80万元"),
    hit(4, "faq_a", "问：能拿多少补贴？答：按正文执行"),
    hit(5, "collection", "稳岗补贴每人每月500元"),
    hit(6, "other", "季度销售报告，收入120万元"),
]
TOPIC = {"1": 0.4, "2": 0.3, "3": 0.2, "4": 0.5, "5": 0.02, "6": 0.001}
ASKED = {"1": 0.6, "2": 0.05, "3": 0.04, "4": 0.9, "5": 0.3, "6": 0.01}
TYPES = {"guide_a": "说明书", "guide_b": "制度规范", "collection": "制度规范"}


def test_files_about_the_topic_rule_files_first_and_each_ones_answer():
    kept = select_by_files(POOL, TOPIC, ASKED, Survey("amount", "一般"), POLICY, files=3, types=TYPES)
    # The rule-stating files come first (the FAQ scores higher on both, but is not one). The
    # collection is far below the best file (0.02 < 0.05 x 0.5) and "other" is not about the
    # topic at all.
    assert [h.doc_id for h in kept] == ["guide_a", "guide_b", "faq_a"]
    # guide_a's answer is the passage with an amount, although the other scores better against the question.
    assert kept[0].id == "2"
    assert kept[0].rerank_score == 0.3  # the better of its two scores
    # With a looser relative floor the collection is about the topic too, and a rule file.
    looser = select_by_files(POOL, TOPIC, ASKED, Survey("amount", "一般"), POLICY, files=3, types=TYPES, topic_relative=0.01)
    assert [h.doc_id for h in looser] == ["guide_a", "guide_b", "collection"]


def test_no_file_about_the_topic_is_no_evidence():
    low = {key: 0.01 for key in TOPIC}
    assert select_by_files(POOL, low, ASKED, Survey("amount", "一般"), POLICY, topic_floor=0.05) == []


# ---------------------------------------------------------------- in the retriever
class FilesStore:
    def __init__(self):
        self.searches = []

    def search_hybrid(self, query, vector, limit, doc_ids=None):
        self.searches.append((query, sorted(doc_ids) if doc_ids else None))
        hits = [hit(h.id, h.doc_id, h.text, 0.1 / int(h.id)) for h in POOL]
        if doc_ids:
            hits = [h for h in hits if h.doc_id in doc_ids]
        return hits[:limit]

    def doc_cards(self, doc_ids):
        return {doc_id: {"doc_type": TYPES.get(doc_id, "")} for doc_id in doc_ids}


class FilesReranker:
    def __init__(self):
        self.queries = []

    def rerank(self, query, documents, top_n):
        self.queries.append(query)
        table = TOPIC if query == "创新" else ASKED
        by_text = {h.text: table[h.id] for h in POOL}
        return sorted((RerankItem(index=i, score=by_text[t]) for i, t in enumerate(documents)), key=lambda item: -item.score)


class Embeddings:
    def embed_query(self, text):
        return [1.0]

    def embed_queries(self, texts):
        return [[1.0] for _ in texts]


def test_the_retriever_goes_files_first_only_when_switched_on():
    question = "一般的创新项目申报能给多少补贴？"
    settings = Settings(wording_expansion=False, survey_file_first=True, survey_files=3, survey_file_candidates=6,
                        survey_passages_per_file=2, survey_question_files=0, survey_excerpt_scoring=False,
                        survey_answering_files=False, survey_excerpts=False)
    store, reranker = FilesStore(), FilesReranker()
    kept = HybridRetriever(store, Embeddings(), reranker, settings).retrieve(question, **retrieval_limits("survey"))
    assert [h.doc_id for h in kept][:2] == ["guide_a", "guide_b"] and kept[0].id == "2"
    assert store.searches[0] == ("创新", None)  # the topic alone finds the files
    assert all(doc_ids and len(doc_ids) == 1 for _, doc_ids in store.searches[1:])  # then each file on its own
    assert sorted(reranker.queries) == sorted([question, "创新"])
    trace = multipart.last_trace().survey
    assert trace["route"] == "files" and trace["topics"] == ["创新"]
    # Off: the round-one survey path (one pool for the whole question).
    store = FilesStore()
    HybridRetriever(store, Embeddings(), FilesReranker(), Settings(wording_expansion=False, survey_file_first=False)).retrieve(
        question, **retrieval_limits("survey"))
    assert store.searches[0][0] == question and multipart.last_trace().survey.get("route") is None


def test_a_failed_topic_scoring_falls_back_to_the_usual_survey_path():
    class Failing(FilesReranker):
        def rerank(self, query, documents, top_n):
            if query == "创新":
                raise RuntimeError("boom")
            return super().rerank(query, documents, top_n)

    settings = Settings(wording_expansion=False, survey_file_first=True)
    kept = HybridRetriever(FilesStore(), Embeddings(), Failing(), settings).retrieve(
        "一般的创新项目申报能给多少补贴？", **retrieval_limits("survey"))
    assert kept and multipart.last_trace().survey.get("route") is None


# ---------------------------------------------------------------- excerpts (survey_excerpts)
from retrieval.survey import excerpt  # noqa: E402

GUIDE = "\n".join([
    "【2026年澜州青年创新项目申请指南】",
    "一、总体要求",
    "为进一步优化营商环境，按照“公开透明、突出重点”的原则，制定本指南。",
    "二、申请主体",
    "年龄在18至35周岁之间的青年个人或青年团队均可提出申请。",
    "三、支持强度与方式",
    "单个项目资助资金不超过6万元；起步档不超过0.8万元。资助资金分两期拨付。",
    "四、受理时间",
    "本年度项目4月17日18时系统关闭，逾期不再受理。",
    "五、绩效管理",
    "对弄虚作假的，五年内不再受理其申请。",
    "六、联系方式",
    "咨询电话：0561-5550568，受理地点：市民服务中心3楼26号窗口。",
])


def test_an_excerpt_is_the_sentences_that_answer_the_ask_under_their_headings():
    money = excerpt(GUIDE, "amount")
    assert money.splitlines()[0] == "【2026年澜州青年创新项目申请指南】"
    assert "三、支持强度与方式 单个项目资助资金不超过6万元；" in money and "0.8万元" in money
    assert "申请主体" not in money and "联系方式" not in money
    time = excerpt(GUIDE, "time")
    assert "4月17日18时系统关闭" in time
    assert "五年内" not in time and "26号窗口" not in time  # a period of a penalty, a room number
    who = excerpt(GUIDE, "who")
    assert "二、申请主体 年龄在18至35周岁" in who.splitlines()[1]
    assert "总体要求" not in who.splitlines()[1]


def test_no_answering_sentence_no_excerpt():
    assert excerpt("【周报】\n本周完成了接口联调。", "amount") is None
    assert excerpt("", "who") is None


def test_the_retriever_cuts_survey_passages_only_with_the_switch():
    question = "一般的创新项目申报能给多少补贴？"
    on = Settings(wording_expansion=False, survey_file_first=True, survey_excerpts=True, survey_files=3,
                  survey_excerpt_scoring=False, survey_question_files=0)
    kept = HybridRetriever(FilesStore(), Embeddings(), FilesReranker(), on).retrieve(question, **retrieval_limits("survey"))
    assert kept[0].text == "四、支持强度 单个项目资助资金不超过6万元"  # a heading and its text on one line
    off = Settings(wording_expansion=False, survey_file_first=True, survey_excerpts=False, survey_files=3,
                   survey_excerpt_scoring=False, survey_question_files=0)
    assert HybridRetriever(FilesStore(), Embeddings(), FilesReranker(), off).retrieve(
        question, **retrieval_limits("survey"))[0].text == POOL[1].text


def test_files_with_an_answering_sentence_come_first():
    from retrieval.survey import answers_in

    pool = POOL + [hit(7, "mail", "答辩通知：请于下周到会议室参加创新项目答辩，携带PPT，共计10分钟。")]
    topic = dict(TOPIC, **{"7": 0.45})
    asked = dict(ASKED, **{"7": 0.2})
    plain = select_by_files(pool, topic, asked, Survey("amount", "一般"), POLICY, files=3, types=TYPES)
    assert "mail" not in {h.doc_id for h in plain}  # past three files already: rule files first
    untyped = select_by_files(pool, topic, asked, Survey("amount", "一般"), POLICY, files=2)
    assert [h.doc_id for h in untyped] == ["faq_a", "mail"]  # by topic score alone
    answering = select_by_files(pool, topic, asked, Survey("amount", "一般"), POLICY, files=2, answers=answers_in)
    # The mail states no amount next to a word for money; the FAQ neither: both go last.
    assert [h.doc_id for h in answering] == ["guide_a", "guide_b"]


def test_a_file_can_be_about_the_topic_by_its_score_against_the_question():
    topic = dict(TOPIC, **{"5": 0.001})  # the collection never names the topic
    without = select_by_files(POOL, topic, ASKED, Survey("amount", "一般"), POLICY, files=8, types=TYPES)
    assert "collection" not in {h.doc_id for h in without}
    with_asked = select_by_files(POOL, topic, ASKED, Survey("amount", "一般"), POLICY, files=8, types=TYPES, about_asked=True)
    assert "collection" in {h.doc_id for h in with_asked}


# ---------------------------------------------------------------- the graph keeps the survey route
def test_a_survey_answer_is_not_sent_to_a_search_rewrite_for_a_low_first_score():
    from rag_agent.edges import route_after_retrieval

    state = {"answer_profile": "survey", "documents": [{"rerank_score": 0.05}], "retrieval_attempts": 1}
    assert route_after_retrieval(state, Settings(survey_keep_route=True)) == "build_context"
    assert route_after_retrieval(state, Settings(survey_keep_route=False)) == "rewrite_search"
    # Nothing kept is still nothing; other question types keep the threshold.
    assert route_after_retrieval(dict(state, documents=[]), Settings(survey_keep_route=True)) == "rewrite_search"
    fact = dict(state, answer_profile="fact")
    assert route_after_retrieval(fact, Settings(survey_keep_route=True)) == "rewrite_search"


def test_a_second_survey_search_uses_the_question_as_asked():
    from types import SimpleNamespace

    from rag_agent.nodes import AgentNodes

    class Recorder:
        def __init__(self):
            self.queries = []

        def retrieve(self, query, **kwargs):
            self.queries.append(query)
            return []

    recorder = Recorder()
    nodes = AgentNodes(llm=None, retriever=recorder, settings=Settings(survey_keep_route=True))
    state = {"question": "一般的创新项目申报能给多少补贴？", "rewritten_query": "创新项目补贴金额", "answer_profile": "survey",
             "retrieval_attempts": 1}
    nodes.retrieve(state)
    nodes.retrieve(dict(state, answer_profile="fact"))
    assert recorder.queries == ["一般的创新项目申报能给多少补贴？", "创新项目补贴金额"]
    off = Recorder()
    AgentNodes(llm=None, retriever=off, settings=Settings(survey_keep_route=False)).retrieve(state)
    assert off.queries == ["创新项目补贴金额"]


# ---------------------------------------------------------------- one bad line costs that line (survey_answer_repair)
from rag_agent.survey_answer import closest_source, drop_lines, repair_names  # noqa: E402

BLOCKS = [{"source": "1-未来产业方向创新任务申报指南.pdf", "locator": "page:2", "text": "……"},
          {"source": "2026年澜州青年创新项目申请指南.docx", "locator": "", "text": "……"},
          {"source": "青屿市科技创新专项项目管理办法（2025年修订）.docx", "locator": "", "text": "……"}]


def test_a_mistyped_file_name_cites_the_closest_evidence_file():
    answer = ("1. 联合体至少有1家用户企业 [来源：1-未来产业方向创新产业任务申报指南.pdf｜位置：page:2]。\n"
              "2. 单个项目不超过6万元 [来源：2026年澜州青年创新项目申请指南.docx]。\n"
              "3. 重点项目80万元 [来源：青屿市科技创新专项项目管理办法.docx]。")
    fixed, changed = repair_names(answer, BLOCKS)
    assert changed == 1
    assert "[来源：1-未来产业方向创新任务申报指南.pdf｜位置：page:2]" in fixed
    assert "青屿市科技创新专项项目管理办法.docx]" in fixed  # shortened: already resolves, left alone
    assert closest_source("周报.txt", [b["source"] for b in BLOCKS]) is None
    # Two files as close: no guess.
    assert closest_source("A项目指南.docx", ["A项目指南一.docx", "A项目指南二.docx"]) is None


def test_unconfirmed_lines_are_dropped_and_the_rest_numbered_again():
    answer = ("各文件的规定如下：\n\n"
              "1. 澜州：单个项目不超过6万元 [来源：a.docx]。\n"
              "2. 青屿：重点项目每项80万元 [来源：b.docx]。\n"
              "3. 栖川：最高200万元 [来源：c.docx]。")
    trimmed, dropped, left = drop_lines(answer, ["青屿：重点项目每项80万元"])
    assert (dropped, left) == (1, 2)
    assert "青屿" not in trimmed and "1. 澜州" in trimmed and "2. 栖川" in trimmed
    assert drop_lines(answer, ["别的内容"]) == (answer, 0, 3)


def test_the_fact_check_keeps_the_confirmed_lines_of_a_survey_answer():
    from types import SimpleNamespace

    from rag_agent.nodes import AgentNodes
    from rag_agent.schemas import FaithfulnessCheck

    class Checker(AgentNodes):
        def _invoke_structured(self, schema, messages, patience=0.0):
            return FaithfulnessCheck(grounded=False, unsupported_claims=["青屿：重点项目每项80万元"])

    answer = ("1. 澜州：单个项目资助资金不超过6万元 [来源：a.docx]。\n"
              "2. 青屿：重点项目每项80万元 [来源：b.docx]。")
    docs = [{"id": "1", "source": "a.docx", "locator": "", "text": "单个项目资助资金不超过6万元", "doc_id": "a"},
            {"id": "2", "source": "b.docx", "locator": "", "text": "面上项目每项20万元", "doc_id": "b"}]
    state = {"answer": answer, "question": "一般能给多少补贴？", "answer_profile": "survey", "context_docs": docs}
    on = Checker(llm=None, retriever=None, settings=Settings(survey_answer_repair=True)).fact_check(state)
    assert on["grounded"] is True and "青屿" not in on["answer"] and on["check_mode"].endswith("dropped:1")
    off = Checker(llm=None, retriever=None, settings=Settings(survey_answer_repair=False)).fact_check(state)
    assert off["grounded"] is False and "answer" not in off
    fact = Checker(llm=None, retriever=None, settings=Settings(survey_answer_repair=True)).fact_check(
        dict(state, answer_profile="fact"))
    assert fact["grounded"] is False
