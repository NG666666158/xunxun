"""The unanswered-constraint line (config rejection_note; guards.unanswered;
evaluation/rejection_v1/REPORT.md): the documents have the question's topic but not its answer."""

from langchain_core.messages import HumanMessage

from rag_agent.edges import route_after_generation
from rag_agent.graph import create_agent_graph
from rag_agent.guards import unanswered
from tests.test_answer_levers import LeverSettings, RecordingLLM
from tests.test_graph_integration import FakeParentStore, FakeRetriever

CITE = " [来源：资料｜位置：段1]"


def block(text, source="资料", locator="段1"):
    return {"source": source, "locator": locator, "text": text}


def line(question, answer, texts, scope="value"):
    blocks = [block(text, locator=f"段{n}") for n, text in enumerate(texts, start=1)]
    found = unanswered(question, answer, blocks, scope=scope)
    return found.line() if found else ""


# ---------------------------------------------------------------- what the question pins down


def test_a_year_no_passage_names():
    texts = ["2020年全市粮食产量为1200万吨。", "2023年全市粮食种植面积继续扩大。"]
    note = line("2021年全市粮食总产量是多少？", "2021年全市粮食总产量为1200万吨" + CITE + "。", texts)
    assert note.startswith("资料里没有找到2021年的内容")
    assert "2020年" in note  # the nearest year the cited passage has


def test_a_year_named_anywhere_is_enough_for_all_but_not_for_cited():
    texts = ["2021年全市粮食产量为1200万吨。", "全市粮食种植面积继续扩大，产量创新高。"]
    answer = "2021年粮食产量创新高 [来源：资料｜位置：段2]。"
    assert line("2021年全市粮食产量是多少？", answer, texts, scope="all") == ""
    assert line("2021年全市粮食产量是多少？", answer, texts, scope="cited").startswith("资料里没有找到2021年的内容")


def test_a_season_is_named_by_its_first_year():
    texts = ["2019-20赛季联赛总决赛，A队以4比1夺得总冠军。"]
    assert line("2019-2020赛季联赛总冠军是哪个队？", "2019-2020赛季联赛总冠军是A队" + CITE + "。", texts) == ""


def test_a_numbered_subject_the_passages_do_not_have():
    texts = ["第六届论坛在杭州举行，参会人数为300人。"]
    note = line("第七届论坛有多少人参会？", "第七届论坛参会人数为300人" + CITE + "。", texts)
    assert note.startswith("资料里没有找到关于“第七届”的内容")


def test_a_company_known_by_its_own_word():
    texts = ["某物流公司向涿州送去了洗护套装和消毒用品。"]
    question = "杭州麦刻物联科技有限公司向涿州送去了哪些物资？"
    note = line(question, "该公司送去了洗护套装和消毒用品" + CITE + "。", texts)
    assert note.startswith("资料里没有找到关于") and "“杭州麦刻物联科技有限公司”" in note
    texts = ["杭州的麦刻共享KTV向涿州送去了饮用水。"]
    assert line(question, "送去了饮用水" + CITE + "。", texts) == ""


# ---------------------------------------------------------------- the answer's own value


def test_a_value_in_no_passage():
    texts = ["本届运动会的吉祥物评选历时半年，共收到作品五千余件。"]
    note = line("本届运动会的吉祥物叫什么？", "本届运动会的吉祥物叫“欢欢”" + CITE + "。", texts)
    assert note == "资料里没有找到这个问题的答案，下面回答里的“欢欢”在资料中没有出现，请核对。"


def test_a_value_written_differently_is_still_found():
    texts = ["当年清明档总票房1.20亿，观影人次342万。", "全市实现地区生产总值43214.85亿元。"]
    assert line("清明档票房是多少？", "清明档总票房为1.2亿" + CITE + "。", texts) == ""
    assert line("全市地区生产总值是多少？", "全市地区生产总值为43214.85亿元" + CITE + "。", texts) == ""


def test_a_value_about_another_year():
    texts = ["数据显示，2023年清明档总票房已突破1.92亿元，超过上一年同期。"]
    note = line("2022清明档票房是多少？", "2022年清明档票房为1.92亿元" + CITE + "。", texts)
    assert note.startswith("资料里没有找到2022年的内容") and "2023年" in note


def test_a_year_the_figure_is_only_compared_with_is_not_its_year():
    texts = ["2023年上半年，公司新签合同额301亿元，较2022年同期增长3.4%。"]
    assert line("2023年上半年公司新签合同额是多少？", "2023年上半年新签合同额为301亿元" + CITE + "。", texts) == ""


def test_a_question_about_two_years_takes_either():
    texts = ["2022年大会在阿布扎比举行。", "2023年大会在新加坡举行。"]
    answer = "2022年和2023年大会的举办地分别是阿布扎比和新加坡" + CITE + "。"
    assert line("2022年和2023年大会的举办地分别是哪里？", answer, texts) == ""


def test_a_value_not_as_the_role_asked():
    texts = ["影片讲述了冯晓光饰演的退休教师和一条狗的故事。", "拍摄地选在重庆，片中的狗名叫八筒。"]
    note = line("这部影片的导演是谁？", "这部影片的导演是冯晓光" + CITE + "。", texts)
    assert note == "资料里没有找到这个问题问的导演，下面回答里的“冯晓光”在资料里说的不是导演，请点开出处核对。"
    texts = ["影片由冯晓光执导，讲述了一位退休教师和一条狗的故事。"]
    assert line("这部影片的导演是谁？", "这部影片的导演是冯晓光" + CITE + "。", texts) == ""


def test_a_measure_not_the_one_asked():
    texts = ["1997年本地GDP相当于全国的18%，2021年只相当于全国的2%。"]
    note = line("2021年本地GDP增长了多少？", "2021年本地GDP占全国的比重为2%" + CITE + "。", texts)
    assert "增长" in note and note.startswith("资料里没有找到")
    texts = ["2021年本地GDP实质增长6.4%。"]
    assert line("2021年本地GDP增长了多少？", "2021年本地GDP增长6.4%" + CITE + "。", texts) == ""


def test_a_district_of_the_city_asked_is_the_city():
    texts = ["天津市遭遇强降雨。宝坻区共有28支青年突击队参与救灾。"]
    question = "天津市宝坻区有多少支青年突击队参与救灾？"
    assert line(question, "宝坻区共有28支青年突击队参与救灾" + CITE + "。", texts) == ""


# ---------------------------------------------------------------- nothing to check


def test_a_common_word_the_dictionary_tags_as_a_place_is_not_a_subject():
    # jieba's dictionary tags 深度 as a place: "设计深度" names nothing the passages must have.
    texts = ["双墩站长182米，宽23.9米，车站设计比较深，达31.7米。"]
    question = "双墩站的设计深度是多少？"
    assert line(question, "双墩站的设计深度为31.7米" + CITE + "。", texts) == ""


def test_a_count_is_not_a_model_number():
    # "覆盖3000多家" is a count; the passages write it "3000余家".
    texts = ["500万元体育消费券将在3000余家上海本地运动门店使用。"]
    question = "上海体育消费券的使用条件是什么？"
    answer = "消费券覆盖3000多家上海本地运动门店" + CITE + "。"
    assert line(question, answer, texts) == ""


def test_correct_answers_refusals_and_other_questions_get_no_line():
    texts = ["2022年本市考研报名人数为15万人，比上年增加1万人。"]
    assert line("2022年本市考研报名人数是多少？", "2022年本市考研报名人数为15万人" + CITE + "。", texts) == ""
    assert line("2022年本市考研报名人数是多少？", "资料里没有找到能回答这个问题的内容。", texts) == ""
    assert line("2022年本市考研报名人数是多少？", "报名人数为20万人。", texts) == ""  # cites nothing
    assert line("How many people applied in 2021?", "2021年报名人数为20万人" + CITE + "。", texts) == ""
    assert line("为什么报名人数增加了？", "因为扩招，报名人数增至20万人" + CITE + "。", texts) == ""
    assert line("学校如何通过扩招和宣传提升报名人数？", "通过多维度宣传，报名人数增至20万人" + CITE + "。", texts) == ""
    assert unanswered("2022年本市考研报名人数是多少？", "报名20万人" + CITE, [block(texts[0])], scope="off") is None


# ---------------------------------------------------------------- in the answer graph


class TicketParents(FakeParentStore):
    def get(self, parent_id):
        return {**super().get(parent_id), "source": "票房.txt", "locator": "段1",
                "text": "2023年清明档总票房已突破1.92亿元，超过上一年同期。"}


TICKET_ANSWER = "2022年清明档票房为1.92亿元 [来源：票房.txt｜位置：段1]。"


def run_tickets(settings, answer=TICKET_ANSWER, question="2022清明档票房是多少？"):
    llm = RecordingLLM(answer=answer)
    graph = create_agent_graph(llm=llm, retriever=FakeRetriever(), settings=settings, parent_store=TicketParents())
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config={"configurable": {"thread_id": "t"}})
    return result, llm


def test_the_line_opens_the_answer_only_when_switched_on():
    result, _ = run_tickets(LeverSettings())
    assert result["answer"] == TICKET_ANSWER and not result.get("rejection_notes")
    settings = LeverSettings()
    settings.rejection_note = "value"
    result, llm = run_tickets(settings)
    note = result["rejection_notes"][0]
    assert result["answer"] == note + "\n\n" + TICKET_ANSWER
    assert note.startswith("资料里没有找到2022年的内容")
    # No extra model call: one answer, and the fact check only sees the cited statement.
    assert sum("基于用户知识库文档回答问题" in system for system, _ in llm.prompts) == 1
    assert not any(note in user for system, user in llm.prompts if "基于用户知识库文档回答问题" not in system)


def test_the_line_replaces_the_anchor_line():
    settings = LeverSettings()
    settings.rejection_note = "value"
    settings.anchor_note = True
    result, _ = run_tickets(settings)
    assert result["answer"].startswith("资料里没有找到2022年的内容")
    assert "注意：引用的资料里没有出现" not in result["answer"] and not result["anchor_notes"]


def test_the_refusal_variant_answers_with_the_line_alone():
    settings = LeverSettings()
    settings.rejection_note = "value"
    settings.rejection_refusal = True
    result, llm = run_tickets(settings)
    assert result["refused"] and result["answer"].startswith("资料里没有找到2022年的内容，所以没有作答")
    assert not any("事实一致性检查器" in system for system, _ in llm.prompts)  # nothing to check
    # Something the question names is in the evidence: the line, and the answer below it.
    result, _ = run_tickets(settings, answer="清明档票房为1.92亿元 [来源：票房.txt｜位置：段1]。",
                            question="2023年清明档票房是多少？")
    assert not result.get("refused") and not result.get("rejection_notes")


def test_routing_after_a_refusal_by_the_code():
    assert route_after_generation({"unanswered_refusal": True}) == "finish"
    assert route_after_generation({"model_error": "x", "unanswered_refusal": True}) == "refuse"
    assert route_after_generation({}) == "fact_check"
