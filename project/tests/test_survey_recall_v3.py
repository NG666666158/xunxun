"""Survey questions, round three (evaluation/survey_recall_v3/REPORT.md)."""

from config import Settings
from rag_agent import survey_answer
from rag_agent.answering import classify_question
from retrieval.survey import answers_in, place_names, rule_name, survey_question, survey_topics


# ---------------------------------------------------------------- the detector without 一般 (survey_wide)
SURVEYS = ["哪些企业能申请创新补贴？", "创新项目补贴有多少？", "有什么补贴可以申请？", "什么企业能申请技术改造补贴？",
           "创业补贴什么时候申请？", "外贸企业有哪些补贴？", "哪些单位可以申报科技项目？", "技改项目最多补多少？"]
FACTS = ["澜州青年创新项目最多能拿多少补贴？", "《临溪市众创空间备案及绩效考核办法》的补贴是多少？", "这个创新项目补贴有多少？",
         "文件里说的补贴多少钱？", "第三条规定的补贴标准是多少？", "P-039项目的补贴是多少？", "张伟的创业补贴批下来了吗？",
         "我们公司去年拿了多少补贴？", "什么是创新券？", "申报材料需要盖章吗？", "苹果手机多少钱？",
         "博士后科研资助工作指引里在站补贴多少？", "总部转岗申请免于人力资源经理追加审批的单笔金额上限是多少？"]


def test_wide_survey_questions_need_the_switch():
    for question in SURVEYS:
        assert survey_question(question) is None, question
        assert survey_question(question, wide=True) is not None, question


def test_look_alike_fact_questions_stay_fact_questions():
    for question in FACTS:
        assert survey_question(question, wide=True) is None, question


def test_the_wide_ask():
    assert survey_question("什么企业能申请技术改造补贴？", wide=True).ask == "who"
    assert survey_question("创新项目补贴有多少？", wide=True).ask == "amount"
    assert survey_question("创业补贴什么时候申请？", wide=True).ask == "time"
    assert survey_question("有什么补贴可以申请？", wide=True).ask == ""


def test_a_place_of_the_library_makes_it_one_places_question():
    places = place_names(["青屿市专利资助操作规程.docx", "关于岚江新区会展资金的通知.txt", "人才市场年报.docx", "高新区通知.pdf"])
    assert {"青屿", "青屿市", "岚江", "岚江新区"} <= places and "人才" not in places and "高新" not in places
    question = "青屿的专利补贴能拿多少？"
    assert survey_question(question, wide=True) is not None  # jieba does not know the place
    assert survey_question(question, wide=True, places=places) is None
    assert classify_question(question, True, True, places) != "survey"
    assert classify_question("专利补贴能拿多少？", True, True, places) == "survey"
    assert classify_question("专利补贴能拿多少？", True, False, places) != "survey"


def test_round_three_is_on_by_default():
    settings = Settings()
    assert settings.survey_wide and settings.survey_topic_related and settings.survey_missing_files
    assert (settings.survey_file_candidates, settings.survey_topic_passages) == (24, 2)


# ---------------------------------------------------------------- files the answer left out (survey_missing_files)
BLOCKS = [
    {"source": "望海市初创企业扶持办法.docx", "locator": "四、扶持标准",
     "text": "【望海市初创企业扶持办法 > 四、扶持标准】\n符合条件的初创企业可获得一次性创业资助1万元。"},
    {"source": "栖川市创业孵化基地认定和运营评价细则.md", "locator": "五、奖补",
     "text": "【栖川市创业孵化基地认定和运营评价细则 > 五、奖补标准】\n新认定为市级创业孵化基地的，给予30万元一次性奖补；\n"
             "年度运营评价为优秀的，给予20万元运营经费。"},
    {"source": "临溪市青年人才生活津贴发放规程.txt", "locator": "三、津贴标准",
     "text": "【临溪市青年人才生活津贴发放规程 > 三、津贴标准】\n生活津贴按学历一次性发放：本科每人1.5万元。"},
    {"source": "1、惠企补贴政策集合.pdf", "locator": "page:3",
     "text": "【1、惠企补贴政策集合】\n补贴标准：每家企业每年最高8万元给予创业场地补贴。"},
]
ANSWER = "创业补贴如下：\n\n1. 望海市初创企业扶持办法：一次性创业资助1万元 [来源：望海市初创企业扶持办法.docx｜位置：四、扶持标准]。\n\n以上供参考。"


def on_topic(question):
    topics = survey_topics(question)
    return lambda source: rule_name(source) and survey_answer.about_topic(source, topics)


def test_a_left_out_guide_about_the_topic_gets_a_line():
    text, added = survey_answer.missing_lines(ANSWER, BLOCKS, "amount", answers_in, on_topic("创业一般能拿到多少补贴？"))
    # The incubator rules (创业 in the name) are added; a talent rule (another topic) and a
    # collection are not.
    assert added == ["栖川市创业孵化基地认定和运营评价细则.md"]
    assert ("2. **栖川市创业孵化基地认定和运营评价细则**：新认定为市级创业孵化基地的，给予30万元一次性奖补；"
            "年度运营评价为优秀的，给予20万元运营经费。[来源：栖川市创业孵化基地认定和运营评价细则.md｜位置：五、奖补]") in text
    assert text.index("2. **栖川") < text.index("以上供参考")


def test_nothing_is_added_without_an_ask_or_an_answering_sentence():
    assert survey_answer.missing_lines(ANSWER, BLOCKS, "", answers_in)[1] == []
    assert survey_answer.missing_lines(ANSWER, BLOCKS, "time", answers_in)[1] == []


def test_a_cited_file_is_not_added_again():
    cited = ANSWER.replace("以上供参考。", "2. 孵化基地：20万元 [来源：栖川市创业孵化基地认定和运营评价细则.md]\n\n以上供参考。")
    assert survey_answer.missing_lines(cited, BLOCKS, "amount", answers_in, on_topic("创业一般能拿到多少补贴？"))[1] == []


def test_an_added_line_drops_section_headings_and_repeated_clauses():
    lines = ["二、申报条件 领办人应具有技师以上职业资格；申请单位应当财务管理制度健全；",
             "三、基本条件 （一）申请单位应当财务管理制度健全；（二）在研项目不超过两项"]
    assert survey_answer._clauses(lines) == "领办人应具有技师以上职业资格；申请单位应当财务管理制度健全；在研项目不超过两项"


# ---------------------------------------------------------------- related programme words (survey_topic_related)
from rag_agent.answering import retrieval_limits  # noqa: E402
from retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from retrieval.survey import related_topics  # noqa: E402
from retrieval.types import SearchHit  # noqa: E402
from services.model_clients import RerankItem  # noqa: E402


def test_related_words_are_the_programmes_under_a_topic():
    assert related_topics(["创新"]) == ["技术 攻关", "成果 转化", "研发"]
    assert related_topics(["开拓市场"]) == ["参展", "会展", "跨境 电商"]  # a table word inside the topic word
    assert related_topics(["技改", "绿色 制造"]) == ["技术改造", "智能化 改造", "设备 更新"]  # what is said is left out
    assert related_topics(["宠物 医院"]) == []


GUIDE, TACKLE, PLAN = "创新项目申请指南", "技术攻关项目实施细则", "研发商业计划书"
PASSAGES = {
    "g1": (GUIDE, "支持强度：单个项目资助资金不超过6万元"),
    "t1": (TACKLE, "支持额度：单个项目最高500万元"),
    "p1": (PLAN, "融资计划：申请政府资助100万元"),
}
FOUND = {"创新": ["g1"], "技术 攻关": ["t1", "p1"], "成果 转化": ["p1"], "研发": ["p1"]}


def passage(key, score=0.1):
    doc, text = PASSAGES[key]
    return SearchHit(id=key, score=score, text=text, doc_id=doc, parent_id=f"{doc}-p0", source=f"{doc}.docx",
                     heading="", locator="", retrieval_method="test")


class RelatedStore:
    def search_hybrid(self, query, vector, limit, doc_ids=None):
        if doc_ids:
            return [passage(key) for key in PASSAGES if PASSAGES[key][0] in doc_ids]
        return [passage(key) for key in FOUND.get(query, list(PASSAGES))]


class RelatedReranker:
    def __init__(self, own=None):
        self.own = own or {"g1": 0.6, "t1": 0.002, "p1": 0.02}
        self.sent = {}

    def rerank(self, query, documents, top_n):
        self.sent[query] = len(documents)
        tables = {"创新": self.own, "技术 攻关": {"g1": 0.01, "t1": 0.7, "p1": 0.3}, "成果 转化": {"p1": 0.5}, "研发": {"p1": 0.5}}
        table = tables.get(query, {key: 0.5 for key in PASSAGES})
        keys = {text: key for key, (_, text) in PASSAGES.items()}
        return sorted((RerankItem(index=i, score=table.get(keys[text], 0.01)) for i, text in enumerate(documents)),
                      key=lambda item: -item.score)


class OneVector:
    def embed_query(self, text):
        return [1.0]

    def embed_queries(self, texts):
        return [[1.0] for _ in texts]


def related_settings(**overrides):
    base = dict(wording_expansion=False, survey_file_first=True, survey_question_files=0, survey_excerpt_scoring=False,
                survey_answering_files=False, survey_excerpts=False, survey_file_candidates=6)
    return Settings(**{**base, **overrides})


def kept_files(settings, reranker=None):
    hits = HybridRetriever(RelatedStore(), OneVector(), reranker or RelatedReranker(), settings).retrieve(
        "一般的创新项目申报能给多少补贴？", **retrieval_limits("survey"))
    return {hit.doc_id for hit in hits}


def test_a_guide_that_never_says_the_topic_is_found_through_a_related_word():
    assert kept_files(related_settings(survey_topic_related=False)) == {GUIDE}
    # The tackling rules are kept; the business plan, found only through 技术攻关 / 研发 and not
    # named like a paper that states rules, is not even a candidate.
    assert kept_files(related_settings(survey_topic_related=True)) == {GUIDE, TACKLE}


def test_the_no_evidence_decision_is_the_questions_own_topic():
    low = RelatedReranker(own={"g1": 0.01, "t1": 0.002, "p1": 0.02})
    assert kept_files(related_settings(survey_topic_related=True), low) == set()


def test_a_topic_judges_each_file_on_its_first_passages():
    reranker = RelatedReranker()
    kept_files(related_settings(survey_topic_related=True, survey_topic_passages=1), reranker)
    assert reranker.sent["创新"] == 2  # one passage of each of the two candidate files
