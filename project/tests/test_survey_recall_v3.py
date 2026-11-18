"""Survey questions, round three (evaluation/survey_recall_v3/REPORT.md)."""

from config import Settings
from rag_agent.answering import classify_question
from retrieval.survey import place_names, survey_question


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


def test_the_switch_is_off_by_default():
    assert Settings().survey_wide is False
