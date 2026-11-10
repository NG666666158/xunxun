"""The answer levers for weak answer models (config; evaluation/generation_v1/REPORT.md)."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from rag_agent.graph import create_agent_graph
from rag_agent.guards import near_term_notes, note_present, rule_check
from rag_agent.nodes import AgentNodes
from rag_agent.prompts import (
    ANSWER_PROMPT,
    CONFLICT_RULE,
    EXTRACT_FIRST_GUIDANCE,
    FACT_CHECK_CITED_PROMPT,
    PLAIN_REFUSAL,
    SUBJECT_CHECK_RULE,
    SUBJECT_RULE,
    answer_prompt,
    fact_check_prompt,
)
from retrieval.types import SearchHit
from tests.test_graph_integration import FakeLLM, FakeParentStore, FakeRetriever, GraphSettings
from tests.test_model_failures import FailingLLM, rate_limit_error


def lines(question, passages):
    return [note.line() for note in near_term_notes(question, passages)]


# ---------------------------------------------------------------- near terms
def test_near_term_gets_one_line_naming_the_word_the_documents_use():
    assert lines("你们产品的质保期是多久？", ["规格书：产品保质期：未开封12个月，开封后3个月内用完。"]) == [
        "资料里写的是“保质期：未开封12个月”，没有提到“质保期”。"
    ]
    assert lines("产品的保修期是多久", ["本产品质保期为两年，自购买之日起计算。"]) == [
        "资料里写的是“质保期为两年”，没有提到“保修期”。"
    ]
    assert lines("进口额是多少", ["今年1-9月出口额为1.2万亿元。"]) == ["资料里写的是“出口额为1.2万亿元”，没有提到“进口额”。"]
    assert lines("神舟十四号的指令长是谁", ["神舟十五号指令长为费俊龙。"]) == [
        "资料里写的是“神舟十五号指令长为费俊龙”，没有提到“神舟十四号”。"
    ]


def test_no_note_on_an_exact_match():
    assert lines("你们产品的保质期是多久？", ["规格书：产品保质期：未开封12个月。"]) == []
    # The question's word anywhere in the evidence is enough.
    assert lines("质保期多久", ["保质期：12个月。质保金在质保期结束后支付。"]) == []


def test_no_note_for_windows_across_words_or_question_furniture():
    assert lines("美国中部的天气如何", ["中国的天气晴朗。"]) == []  # "国中" is not a word here
    assert lines("香港第七届立法会选举有多少个议席", ["香港第七届立法会选举共有90个议席。"]) == []
    assert lines("电影《流浪地球2》什么时候上映", ["《流浪地球2》将于2023年1月22日在国内上映。"]) == []
    assert lines("哪些国家参加了会议", ["共有5个国家参加。"]) == []  # not "些国家" / "个国家"
    # A count is not a term: "两个项目" asked, "单个项目" written (seen on the user's data).
    assert lines("这两个项目分别申请多少资金？", ["单个项目最高资助1000万元。"]) == []
    assert lines("三家公司的注册资本是多少", ["两家公司注册资本均为500万元。"]) == []


def test_no_note_when_the_evidence_names_the_difference_or_only_reorders_words():
    # "亚军" is written, only not as "女单亚军".
    assert lines("2023年德国乒乓球公开赛女单亚军是谁", ["陈梦夺得女单冠军，孙颖莎获得亚军。"]) == []
    assert lines("2022年冬奥会男子单板冠军是谁", ["男子单板滑雪比赛在冬奥会上举行，苏翊鸣夺冠。"]) == []


def test_note_present_recognizes_an_answer_that_already_says_it():
    note = near_term_notes("你们产品的质保期是多久？", ["产品保质期：未开封12个月。"])[0]
    assert note_present("资料里写的是保质期12个月，没有提到质保期 [来源：a｜位置：b]。", note)
    assert not note_present("质保期为12个月 [来源：a｜位置：b]。", note)


# ---------------------------------------------------------------- rule check
def test_rule_check_flags_a_model_name_or_number_found_in_no_passage():
    problems = rule_check("KV Cache 在 A100 上性能最优 [来源：书.pdf｜位置：page:3]。", ["KV Cache 可以减少重复计算。"])
    assert len(problems) == 1 and "a100" in problems[0].lower()
    assert rule_check("发行规模为3.5亿元 [来源：a｜位置：b]。", ["发行规模为2.36亿元。"])


def test_rule_check_accepts_rewritten_numbers_and_ignores_words_that_look_like_numbers():
    assert rule_check("保质期为十二个月 [来源：规格书｜位置：table:3]。", ["保质期：未开封12个月"]) == []
    assert rule_check("这一点十分重要，规模为2,360万元 [来源：a｜位置：b]", ["规模为2360万元"]) == []
    # A statement citing the wrong passage is not an invented fact: all evidence counts.
    assert rule_check("规模为2.36亿元 [来源：a｜位置：1]", ["无关", "规模为2.36亿元"]) == []


def test_rule_check_takes_what_the_question_says_as_given():
    answer = "《踏冰逐梦》2022年的收视率为1.2% [来源：a｜位置：b]。"
    assert rule_check(answer, ["收视率为1.2%"])
    assert rule_check(answer, ["收视率为1.2%"], question="《踏冰逐梦》2022年的收视率是多少") == []


# ---------------------------------------------------------------- prompts
def test_answer_prompt_switches():
    assert answer_prompt() == ANSWER_PROMPT
    assert PLAIN_REFUSAL in answer_prompt(plain_refusal=True) and "当前知识库没有足够依据" not in answer_prompt(plain_refusal=True)
    assert answer_prompt(conflict_note=True).endswith("8. " + CONFLICT_RULE)
    assert answer_prompt(subject_rule=True).endswith("8. " + SUBJECT_RULE)
    both = answer_prompt(conflict_note=True, subject_rule=True)
    assert both.endswith("8. " + CONFLICT_RULE + "9. " + SUBJECT_RULE)


def test_fact_check_prompt_switch():
    assert fact_check_prompt(True) == FACT_CHECK_CITED_PROMPT
    assert fact_check_prompt(True, subject_rule=True) == FACT_CHECK_CITED_PROMPT + SUBJECT_CHECK_RULE
    assert fact_check_prompt(False, subject_rule=True).endswith(SUBJECT_CHECK_RULE)


class LeverSettings(GraphSettings):
    answer_mode = "strict"
    fact_check_patience_seconds = 0.0
    rule_precheck = False
    term_note = False
    fact_max_passages = 6
    fact_relative_score = 0.0
    extract_first = False
    conflict_note = False
    plain_refusal = False
    subject_rule = False
    anchor_note = False
    conflict_finder = False


class RecordingLLM(FakeLLM):
    def __init__(self, answer=None):
        super().__init__(grounded=True)
        self.answer, self.prompts = answer, []

    def invoke(self, messages):
        self.prompts.append((str(messages[0].content), str(messages[-1].content)))
        if self.answer and "基于用户知识库文档回答问题" in str(messages[0].content):
            return AIMessage(content=self.answer)
        return super().invoke(messages)


def run(llm, settings, question="住宿费超标怎么办？", retriever=None):
    graph = create_agent_graph(llm=llm, retriever=retriever or FakeRetriever(), settings=settings,
                               parent_store=FakeParentStore())
    return graph.invoke({"messages": [HumanMessage(content=question)]}, config={"configurable": {"thread_id": "t"}})


def test_switches_off_send_the_old_prompt():
    llm = RecordingLLM()
    run(llm, LeverSettings())
    answer_calls = [p for p in llm.prompts if "基于用户知识库文档回答问题" in p[0]]
    assert answer_calls[0][0] == ANSWER_PROMPT


def test_subject_rule_reaches_the_answer_and_the_check():
    llm = RecordingLLM()
    settings = LeverSettings()
    settings.subject_rule = True
    run(llm, settings)
    systems = [system for system, _ in llm.prompts]
    assert any(system.endswith(SUBJECT_RULE) for system in systems if "基于用户知识库文档回答问题" in system)
    assert any(system.endswith(SUBJECT_CHECK_RULE) for system in systems if "事实一致性检查器" in system)
    llm = RecordingLLM()
    run(llm, LeverSettings())
    assert not any(SUBJECT_CHECK_RULE in system or SUBJECT_RULE in system for system, _ in llm.prompts)


def test_extract_first_changes_only_fact_questions():
    llm = RecordingLLM()
    settings = LeverSettings()
    settings.extract_first = True
    run(llm, settings, question="住宿费超标需要谁审批？")
    assert any(EXTRACT_FIRST_GUIDANCE in user for system, user in llm.prompts)
    llm = RecordingLLM()
    run(llm, settings, question="为什么住宿费超标要审批？")  # an explanation
    assert not any(EXTRACT_FIRST_GUIDANCE in user for system, user in llm.prompts)


class ShelfLifeParents(FakeParentStore):
    def get(self, parent_id):
        return {**super().get(parent_id), "source": "规格书.pdf", "locator": "table:3",
                "text": "产品保质期：未开封12个月，开封后3个月内用完。"}


def test_term_note_opens_the_answer_and_the_prompt_warns_the_model():
    llm = RecordingLLM(answer="产品的质保期为未开封12个月 [来源：规格书.pdf｜位置：table:3]。")
    settings = LeverSettings()
    settings.term_note = True
    graph = create_agent_graph(llm=llm, retriever=FakeRetriever(), settings=settings, parent_store=ShelfLifeParents())
    result = graph.invoke({"messages": [HumanMessage(content="你们产品的质保期是多久？")]},
                          config={"configurable": {"thread_id": "t"}})
    assert result["answer"].startswith("资料里写的是“保质期：未开封12个月”，没有提到“质保期”。")
    assert result["term_notes"]
    assert any("用词提醒" in user for system, user in llm.prompts)


def test_term_note_is_not_repeated_when_the_answer_says_it():
    answer = "资料里写的是“保质期：未开封12个月”，没有提到“质保期”。保质期为未开封12个月 [来源：规格书.pdf｜位置：table:3]。"
    llm = RecordingLLM(answer=answer)
    settings = LeverSettings()
    settings.term_note = True
    graph = create_agent_graph(llm=llm, retriever=FakeRetriever(), settings=settings, parent_store=ShelfLifeParents())
    result = graph.invoke({"messages": [HumanMessage(content="你们产品的质保期是多久？")]},
                          config={"configurable": {"thread_id": "t"}})
    assert result["answer"] == answer and result["term_notes"] == []


# ---------------------------------------------------------------- waiting out rate limits
def test_backoff_with_patience_adds_up_to_the_patience():
    nodes = AgentNodes(None, None, LeverSettings())
    assert nodes._backoff(0) == [4.0, 8.0, 16.0]
    assert sum(nodes._backoff(75)) == 75 and max(nodes._backoff(75)) <= 20


def test_patient_fact_check_waits_instead_of_skipping(monkeypatch):
    monkeypatch.setattr(AgentNodes, "rate_limit_backoff", (0.0,))
    monkeypatch.setattr(AgentNodes, "patience_step", 0.001)
    settings = LeverSettings()
    settings.fact_check_patience_seconds = 0.005  # five more tries than the usual one
    llm = FailingLLM("事实一致性检查器", rate_limit_error, times=4)
    result = run(llm, settings)
    assert result["grounded"] is True and not result["check_skipped"]
    # Without patience the same failures skip the check.
    result = run(FailingLLM("事实一致性检查器", rate_limit_error, times=4), LeverSettings())
    assert result["check_skipped"] is True


def timeout_error():
    import httpx
    import openai

    return openai.APITimeoutError(request=httpx.Request("POST", "https://api.siliconflow.cn/v1/chat/completions"))


def test_patient_fact_check_also_retries_a_timed_out_request(monkeypatch):
    monkeypatch.setattr(AgentNodes, "rate_limit_backoff", (0.0,))
    monkeypatch.setattr(AgentNodes, "patience_step", 0.001)
    settings = LeverSettings()
    settings.fact_check_patience_seconds = 0.005
    result = run(FailingLLM("事实一致性检查器", timeout_error, times=2), settings)
    assert result["grounded"] is True and not result["check_skipped"]
    # Without patience a timeout is not retried here (the client retried it already).
    result = run(FailingLLM("事实一致性检查器", timeout_error, times=1), LeverSettings())
    assert result["check_skipped"] is True


def test_patience_gives_up_once_the_wall_time_is_used(monkeypatch):
    monkeypatch.setattr(AgentNodes, "rate_limit_backoff", (0.0,))
    monkeypatch.setattr(AgentNodes, "patience_step", 0.001)
    monkeypatch.setattr(AgentNodes, "patience_slack", -1.0)  # the cap is already passed
    settings = LeverSettings()
    settings.fact_check_patience_seconds = 0.005
    llm = FailingLLM("事实一致性检查器", timeout_error, times=99)
    result = run(llm, settings)
    assert result["check_mode"] == "rules"
    assert sum(call.startswith("你是事实一致性") for call in llm.calls) <= 2


class A100LLM(FailingLLM):
    def invoke(self, messages):
        if "基于用户知识库文档回答问题" in str(messages[0].content):
            return AIMessage(content="住宿费超标时需要在 A100 系统登记 [来源：差旅制度.docx，section:2]。")
        return super().invoke(messages)


def test_check_that_cannot_run_falls_back_to_the_rule_check(monkeypatch):
    monkeypatch.setattr(AgentNodes, "rate_limit_backoff", (0.0,))
    monkeypatch.setattr(AgentNodes, "patience_step", 0.001)
    settings = LeverSettings()
    settings.fact_check_patience_seconds = 0.002
    # A name in no passage: not shown unchecked; searched again, then refused.
    result = run(A100LLM("事实一致性检查器", rate_limit_error), settings)
    assert result["check_mode"] == "rules" and result["refused"] is True
    # Nothing the rules can catch: shown, marked as checked by the rules only.
    result = run(FailingLLM("事实一致性检查器", rate_limit_error), settings)
    assert result["check_skipped"] is True and result["check_mode"] == "rules"
    assert "部门负责人审批" in result["answer"]


def test_rule_precheck_fails_an_invented_name_without_a_model_call(monkeypatch):
    settings = LeverSettings()
    settings.rule_precheck = True
    llm = A100LLM("事实一致性检查器", rate_limit_error, times=0)
    result = run(llm, settings)
    assert result["check_mode"] == "rules" and result["refused"] is True
    assert not any("事实一致性" in call for call in llm.calls)


def test_planner_failure_searches_the_question_as_asked(monkeypatch):
    monkeypatch.setattr(AgentNodes, "rate_limit_backoff", (0.0,))
    llm = FailingLLM("查询规划器", rate_limit_error)
    plan = AgentNodes(llm, None)._plan("北方区考勤办法", "")
    assert plan.intent == "knowledge" and plan.rewritten_query == "北方区考勤办法"


# ---------------------------------------------------------------- fewer passages
class LimitRetriever:
    def __init__(self):
        self.kwargs = None

    def retrieve(self, query, mode, **kwargs):
        self.kwargs = kwargs
        return []


def test_fact_passage_limits_only_for_fact_questions():
    settings = SimpleNamespace(fact_max_passages=4, fact_relative_score=0.3)
    retriever = LimitRetriever()
    nodes = AgentNodes(None, retriever, settings)
    nodes.retrieve({"rewritten_query": "q", "answer_profile": "fact"})
    assert retriever.kwargs == {"max_keep": 4, "relative": 0.3}
    nodes.retrieve({"rewritten_query": "q", "answer_profile": "explain"})
    assert retriever.kwargs == {"max_keep": 10}
    # The defaults pass nothing new (the retrievers of older tests take no "relative").
    nodes = AgentNodes(None, retriever, SimpleNamespace(fact_max_passages=6, fact_relative_score=0.0))
    nodes.retrieve({"rewritten_query": "q", "answer_profile": "fact"})
    assert retriever.kwargs == {"max_keep": 6}


def test_relative_cutoff_reaches_the_selection_policy():
    from retrieval.hybrid_retriever import HybridRetriever

    settings = SimpleNamespace(rerank_top_k=12, rerank_min_score=0.1, rerank_relative_score=0.1, rerank_min_keep=3,
                               rerank_per_document=4, rerank_no_evidence_below=0.1)
    retriever = HybridRetriever(None, None, None, settings)
    assert retriever.policy(6).relative == 0.1
    assert retriever.policy(6, relative=0.3).relative == 0.3



# ---------------------------------------------------------------- anchors (lever 7)
def test_question_anchors_are_years_titles_and_named_places():
    from rag_agent.guards import question_anchors

    assert question_anchors("印控克什米尔地区寺庙踩踏事故死亡人数") == ["克什米尔地区"]
    assert question_anchors("2021年香港GDP增长了多少？") == ["2021", "香港"]
    assert question_anchors("《踏冰逐梦》的主演是谁") == ["踏冰逐梦"]
    assert question_anchors("2020-2021赛季欧冠冠军") == ["2020-2021"]
    assert "2023-12" not in question_anchors("Was the report published on 2023-12-01 consistent?")


def test_missing_anchor_found_only_when_no_cited_passage_mentions_it():
    from rag_agent.guards import missing_anchors

    blocks = [{"source": "文档5", "locator": "段5", "text": "印度拉贾斯坦邦一座寺庙入口发生踩踏，造成3人死亡。"},
              {"source": "文档1", "locator": "段1", "text": "印控克什米尔地区的寺庙发生踩踏事故。"}]
    question = "印控克什米尔地区寺庙踩踏事故死亡人数"
    answer = "印控克什米尔地区寺庙踩踏事故死亡人数为3人 [来源：文档5｜位置：段5]。"
    assert missing_anchors(question, answer, blocks) == ["克什米尔地区"]
    # The anchor is in another passage the answer also cites: no note.
    assert missing_anchors(question, answer + "事故发生在克什米尔 [来源：文档1｜位置：段1]。", blocks) == []
    # Nothing cited (a refusal): nothing to check.
    assert missing_anchors(question, "资料里没有找到能回答这个问题的内容。", blocks) == []
    # A team or city named without its last character is the same place.
    team = [{"source": "a", "locator": "b", "text": "德国与日本、西班牙同在E组。"}]
    assert missing_anchors("卡塔尔世界杯德国队在哪个组", "德国队在E组 [来源：a｜位置：b]。", team) == []


def test_missing_year_is_an_anchor_even_when_another_year_is_there():
    from rag_agent.guards import missing_anchors

    blocks = [{"source": "a", "locator": "b", "text": "2022年全国粮食总产量13731亿斤。"}]
    assert missing_anchors("2021年全国粮食产量是多少？", "2021年全国粮食总产量为13731亿斤 [来源：a｜位置：b]。",
                           blocks) == ["2021"]


class HarvestParents(FakeParentStore):
    def get(self, parent_id):
        return {**super().get(parent_id), "source": "统计公报.pdf", "locator": "page:2",
                "text": "2022年全国粮食总产量13731亿斤，比上年增加74亿斤。"}


def test_anchor_note_opens_the_answer_only_when_switched_on():
    answer = "2021年全国粮食总产量为13731亿斤 [来源：统计公报.pdf｜位置：page:2]。"
    question = "2021年全国粮食产量是多少？"
    settings = LeverSettings()
    settings.anchor_note = True
    graph = create_agent_graph(llm=RecordingLLM(answer=answer), retriever=FakeRetriever(), settings=settings,
                               parent_store=HarvestParents())
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config={"configurable": {"thread_id": "t"}})
    assert result["answer"].startswith("注意：引用的资料里没有出现“2021”")
    assert result["anchor_notes"]
    graph = create_agent_graph(llm=RecordingLLM(answer=answer), retriever=FakeRetriever(), settings=LeverSettings(),
                               parent_store=HarvestParents())
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config={"configurable": {"thread_id": "t"}})
    assert result["answer"] == answer and not result.get("anchor_notes")


# ---------------------------------------------------------------- conflicting values (code)
def found(question, answer, blocks):
    from rag_agent.guards import conflicting_values

    return [conflict.values for conflict in conflicting_values(question, answer, blocks)]


def block(source, locator, text):
    return {"source": source, "locator": locator, "text": text}


def test_conflict_between_two_numbers_with_the_same_unit_names_both_sources():
    from rag_agent.guards import conflicting_values

    blocks = [block("统计公报.pdf", "page:3", "2019年全年出生人口1465万人，人口出生率为10.48‰。"),
              block("新闻稿.docx", "段2", "2019年出生人口1465万人，人口出生率为5.48‰，比上年下降。")]
    answer = "2019年人口出生率为5.48‰ [来源：新闻稿.docx｜位置：段2]。"
    conflicts = conflicting_values("2019年人口出生率是多少？", answer, blocks)
    assert [c.values for c in conflicts] == [(("5.48‰", 1), ("10.48‰", 0))]
    assert conflicts[0].line(blocks) == (
        "注意：资料里的说法不一致，《新闻稿》段2写的是“5.48‰”，《统计公报》page:3写的是“10.48‰”，请点开出处核对。")


def test_conflict_between_names_places_and_dates():
    directors = [block("a", "1", "电影《星海》由导演史蒂文·斯皮尔伯格执导，2018年上映。"),
                 block("b", "2", "《星海》是布莱恩·辛格导演的一部科幻片。")]
    assert found("电影《星海》的导演是谁？", "《星海》的导演是布莱恩·辛格 [来源：b｜位置：2]。", directors) == [
        (("布莱恩·辛格", 1), ("史蒂文·斯皮尔伯格", 0))]
    unknown = [block("a", "1", "《远山》由郭帆执导。"), block("b", "2", "《远山》的导演是陈思诚。")]
    assert found("《远山》的导演是谁", "《远山》的导演是郭帆 [来源：a｜位置：1]。", unknown) == [(("郭帆", 0), ("陈思诚", 1))]
    hosts = [block("a", "1", "第七届世界军人运动会于2019年10月在武汉举办。"),
             block("b", "2", "2019年军运会在成都举办，共有100多个国家参赛。")]
    assert found("2019年军运会在哪个城市举办？", "2019年军运会在武汉举办 [来源：a｜位置：1]。", hosts) == [
        (("武汉", 0), ("成都", 1))]
    dates = [block("a", "1", "新品发布会定于2024年3月18日举行。"), block("b", "2", "新品发布会于2024年3月25日在上海举行。")]
    assert found("新品发布会是哪天？", "新品发布会在2024年3月18日举行 [来源：a｜位置：1]。", dates) == [
        (("2024年3月18日", 0), ("2024年3月25日", 1))]


def test_no_conflict_when_every_passage_carries_the_answers_value():
    blocks = [block("a", "1", "一线城市住宿费上限为每晚600元。"), block("b", "2", "住宿费上限600元，超标需说明原因，另有100元补贴。")]
    assert found("一线城市住宿费上限是多少？", "一线城市住宿费上限为每晚600元 [来源：a｜位置：1]。", blocks) == []


def test_no_conflict_for_another_year_or_another_numbered_subject():
    years = [block("a", "1", "2018年人口出生率为10.94‰。"), block("b", "2", "2019年人口出生率为10.48‰。")]
    assert found("2018年人口出生率是多少？", "2018年人口出生率为10.94‰ [来源：a｜位置：1]。", years) == []
    ships = [block("a", "1", "神舟十六号于2023年5月30日发射。"), block("b", "2", "神舟十七号预计于2023年10月26日发射。")]
    assert found("神舟十六号是哪天发射的？", "神舟十六号于2023年5月30日发射 [来源：a｜位置：1]。", ships) == []
    # A news dateline is when the report was filed, not the launch.
    dateline = [block("a", "1", "神舟十六号于2023年5月30日发射。"), block("b", "2", "新华社酒泉5月29日电 神舟十六号发射准备就绪。")]
    assert found("神舟十六号是哪天发射的？", "神舟十六号于2023年5月30日发射 [来源：a｜位置：1]。", dateline) == []


def test_no_conflict_between_things_asked_side_by_side_or_members_of_a_list():
    blocks = [block("a", "1", "2021年香港GDP增长率为6.4%。"), block("b", "2", "2021年澳门GDP增长率为18%。")]
    answer = "2021年香港GDP增长率为6.4% [来源：a｜位置：1]。澳门GDP增长率为18% [来源：b｜位置：2]。"
    assert found("2021年香港和澳门的GDP增长率分别是多少？", answer, blocks) == []
    members = [block("a", "1", "RCEP成员包括中国、日本、韩国。"), block("b", "2", "RCEP成员还有澳大利亚。")]
    assert found("RCEP包括哪些国家？", "RCEP包括中国、日本、韩国 [来源：a｜位置：1]。", members) == []
    # One of several names listed together is not a rival.
    listed = [block("a", "1", "《七人》的导演是洪金宝。"), block("b", "2", "《七人》由杜琪峰、徐克等七位导演联合执导。")]
    assert found("《七人》的导演是谁？", "《七人》的导演是洪金宝 [来源：a｜位置：1]。", listed) == []


def test_no_conflict_without_citations_or_for_values_the_question_already_names():
    blocks = [block("a", "1", "人口出生率为10.48‰。"), block("b", "2", "人口出生率为5.48‰。")]
    assert found("人口出生率是多少？", "资料里没有找到能回答这个问题的内容。", blocks) == []
    # "分别" with nothing to tell the parts apart: no guess.
    assert found("两个季度的营收分别是多少？", "营收为463亿元 [来源：a｜位置：1]。",
                 [block("a", "1", "营收为463亿元。"), block("b", "2", "营收为509亿元。")]) == []



def test_no_conflict_for_running_totals_how_questions_or_another_rank():
    # Running totals reported on different dates differ without contradicting each other.
    totals = [block("a", "1", "截至11月3日，累计有9.1万名旅客乘坐跨境列车。"),
              block("b", "2", "开行100多天来，累计有4.7万名旅客乘坐跨境列车。")]
    assert found("乘坐跨境列车的旅客有多少？", "累计有9.1万名旅客乘坐跨境列车 [来源：a｜位置：1]。", totals) == []
    # "How" questions have no single value per fact.
    stages = [block("a", "1", "7月30日应对台风启动应急响应。"), block("b", "2", "8月5日应对台风的应急响应解除。")]
    assert found("海事局是如何分阶段应对台风的？", "7月30日应对台风启动应急响应 [来源：a｜位置：1]。", stages) == []
    # Another rank of the same office: "第四任" asked, "第1位" written.
    ranks = [block("a", "1", "梁振英当选香港特区第四任行政长官。"), block("b", "2", "董建华是第1位香港行政长官。")]
    assert found("香港第四任行政长官是谁？", "香港第四任行政长官是梁振英 [来源：a｜位置：1]。", ranks) == []
    # A date of another year than the question's.
    quakes = [block("a", "1", "2023年2月6日发生强烈地震。"), block("b", "2", "规模最大的一次地震发生在2020年1月。")]
    assert found("2023年土耳其地震的时间", "2023年土耳其地震发生在2023年2月6日 [来源：a｜位置：1]。", quakes) == []


def test_conflict_finds_a_large_amount_with_all_its_decimals():
    # Seven significant digits: "43214.85" must not be read as "43214.9".
    gdp = [block("a", "1", "2021年上海市地区生产总值为43214.85亿元，比上年增长8.1%。"),
           block("b", "2", "2021年上海市地区生产总值为38700.58亿元。")]
    assert found("2021年上海市地区生产总值是多少？", "2021年上海市地区生产总值为43214.85亿元 [来源：a｜位置：1]。", gdp) == [
        (("43214.85亿元", 0), ("38700.58亿元", 1))]
    # The other passage writing the same amount is no rival.
    same = [gdp[0], block("b", "2", "上海2021年地区生产总值43,214.85亿元。")]
    assert found("2021年上海市地区生产总值是多少？", "2021年上海市地区生产总值为43214.85亿元 [来源：a｜位置：1]。", same) == []


def test_conflict_finds_an_amount_written_with_trailing_zeros():
    from rag_agent.guards import _amount_in

    profit = [block("a", "1", "2023年公司净利润为1.20亿元，同比增长5%。"), block("b", "2", "2023年公司净利润为0.98亿元。")]
    assert found("2023年公司净利润是多少？", "2023年公司净利润为1.2亿元 [来源：a｜位置：1]。", profit) == [
        (("1.2亿元", 0), ("0.98亿元", 1))]
    assert _amount_in(1.2, "净利润1.20亿元") and _amount_in(600, "上限600.0元")
    # Not part of a longer number.
    assert not _amount_in(12, "共12.5万人") and not _amount_in(1.2, "1.25亿元") and not _amount_in(600, "1600元")


class PolicyRetriever:
    def retrieve(self, query, mode, max_keep=None):
        return [
            SearchHit(id=f"c{n}", score=0.9 - n / 10, text=text, doc_id=f"{n}" * 24, parent_id=f"{n}" * 24 + "-p0",
                      source=source, heading="住宿标准", locator=f"section:{n}", retrieval_method="hybrid_rrf_bge_rerank",
                      rerank_score=0.9)
            for n, (source, text) in enumerate([("差旅制度2023.docx", "一线城市住宿费上限为每晚600元。"),
                                                ("差旅制度修订.docx", "一线城市住宿费上限调整为每晚500元。")], start=1)
        ]


class PolicyParents(FakeParentStore):
    def get(self, parent_id):
        return None  # the child passages are the context


def run_policy(llm, settings):
    graph = create_agent_graph(llm=llm, retriever=PolicyRetriever(), settings=settings, parent_store=PolicyParents())
    return graph.invoke({"messages": [HumanMessage(content="一线城市住宿费上限是多少？")]},
                        config={"configurable": {"thread_id": "t"}})


POLICY_ANSWER = "一线城市住宿费上限为每晚600元 [来源：差旅制度2023.docx｜位置：section:1]。"
POLICY_LINE = "注意：资料里的说法不一致，《差旅制度2023》section:1写的是“600元”，《差旅制度修订》section:2写的是“500元”，请点开出处核对。"


def test_conflict_finder_opens_the_answer_with_the_line_only_when_switched_on():
    settings = LeverSettings()
    settings.conflict_finder = True
    llm = RecordingLLM(answer=POLICY_ANSWER)
    result = run_policy(llm, settings)
    assert result["answer"] == POLICY_LINE + "\n\n" + POLICY_ANSWER
    assert result["conflict_notes"] == [POLICY_LINE]
    # The answer prompt is the one without the switch: the line costs no model call.
    assert sum("基于用户知识库文档回答问题" in system for system, _ in llm.prompts) == 1
    result = run_policy(RecordingLLM(answer=POLICY_ANSWER), LeverSettings())
    assert result["answer"] == POLICY_ANSWER and not result.get("conflict_notes")
