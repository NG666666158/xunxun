"""Answers drawing on several documents (rag_agent/multidoc_answer.py; config part_answer,
dated_evidence, conclusion_line, citation_repair; evaluation/multidoc_answer_v1/REPORT.md)."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from rag_agent import multidoc_answer as md
from rag_agent.graph import create_agent_graph
from rag_agent.nodes import AgentNodes
from retrieval.types import SearchHit
from tests.test_answer_levers import LeverSettings
from tests.test_graph_integration import FakeLLM


# ---------------------------------------------------------------- question shapes
def test_question_shapes():
    shape = md.question_shape
    assert shape("员工出差住宿费超标是否需要审批？") == "yesno"
    assert shape("这款产品支持离线使用吗？") == "yesno"
    assert shape("A 方案和 B 方案哪个更便宜？") == "choice"
    assert shape("首选是成都还是武汉？") == "choice"
    assert shape("展会在哪个城市举办，并且有多少家企业参展？") == "asks"
    assert shape("住宿费超标需要谁审批？") == "one"
    # A yes / no ask next to another interrogative is not a plain yes / no question.
    assert shape("是否需要审批，审批人是谁？") == "asks"
    assert shape("Does the TechCrunch article report a drop in revenue, and does the Verge article agree?") == "yesno"
    assert shape("After the report by The Verge on October 20, 2023, did the Music Business Worldwide report agree?") == "yesno"
    assert shape("Between the TechCrunch report on X and the report on Y, was there a change in tone?") == "yesno"
    assert shape("Which company, reported by The Verge and TechCrunch, has also launched a phone?") == "one"
    assert shape("Who founded the company, and when did it go public?") == "asks"
    assert shape("Considering the articles by Variety and The Hollywood Reporter, which character is voiced by X?") == "one"
    assert shape("Which team scored more in the final, the Springboks or the All Blacks?") == "choice"


def test_part_mode_needs_parts_and_a_shape_that_asks_several_things():
    assert md.part_mode("广州举办了哪些展会，并且有多少家企业参展？") == ["广州举办了哪些展会", "广州举办了 有多少家企业参展"] or \
        len(md.part_mode("广州举办了哪些展会，并且有多少家企业参展？")) == 2
    assert md.part_mode("住宿费超标需要谁审批？") == []
    assert md.part_mode("Which company, reported by The Verge and TechCrunch, has also launched a phone?") == []


# ---------------------------------------------------------------- per-part evidence
def test_passages_are_grouped_under_the_part_whose_words_they_mention():
    parts = ["黑龙江省在大会上发挥了哪些作用", "哈尔滨冰雪节吸引了多少游客"]
    texts = [
        "哈尔滨冰雪节今年吸引游客300万人次。",
        "黑龙江省在大会上发挥了桥梁作用，签约项目20个。",
        "天气晴朗。",
    ]
    groups = md.group_passages(parts, texts)
    assert groups.own == [[1], [0]]
    assert groups.other == [2]
    blocks = [f"[证据 {i + 1}｜来源：d{i + 1}｜位置：p]\n{text}" for i, text in enumerate(texts)]
    evidence = md.grouped_evidence(groups, blocks)
    assert evidence.index("【小问1 的证据】") < evidence.index("证据 2｜") < evidence.index("【小问2 的证据】")
    assert evidence.index("【小问2 的证据】") < evidence.index("证据 1｜") < evidence.index("【其它证据】")
    assert evidence.count("[证据 1｜") == 1  # each passage printed once


def test_a_part_without_its_own_passage_says_so():
    groups = md.PartGroups(["甲", "乙"], [[0], []], [[], []], [])
    assert "没有专门讲这一小问的证据" in md.grouped_evidence(groups, ["[证据 1｜来源：a｜位置：b]\n甲"])


# ---------------------------------------------------------------- dates
def test_temporal_questions():
    assert md.temporal_question("Did the report published on October 12, 2023 agree with the later one?")
    assert md.temporal_question("Was the Sporting News report before the Fortune article?")
    assert md.temporal_question("两份通知哪个先发布？")
    assert md.temporal_question("10月12日的报道说了什么？")
    assert not md.temporal_question("住宿费超标需要谁审批？")
    assert not md.temporal_question("Which company launched the phone?")


def test_date_text_and_dated_passages():
    assert md.date_text("2023-10-12T08:45:59+00:00") == "2023-10-12"
    assert md.date_text("2023/1/5") == "2023-01-05"
    assert md.date_text(1700000000.0)[:4] == "2023"
    assert md.date_text("") == "" and md.date_text("soon") == ""
    assert md.dated_text("正文", "2023-10-12") == "（文件日期：2023-10-12）\n正文"
    assert md.dated_text("正文", "") == "正文"


# ---------------------------------------------------------------- conclusion line
def test_the_models_conclusion_line_is_normalised():
    result = md.check_conclusion("Does A agree with B?", "**结论：** 否\n\nA says X [来源：a｜位置：1]。", "yesno")
    assert result.status == "model" and result.answer.startswith("结论：否。\n") and result.verdict == "否"
    result = md.check_conclusion("是否需要审批？", "结论：资料不足\n证据没有讲到。", "yesno")
    assert result.answer.startswith("结论：资料里的内容不足以判断。")


def test_a_verdict_in_the_first_sentence_becomes_the_conclusion_line():
    result = md.check_conclusion("Does A agree with B?", "Yes, both reports say X [来源：a｜位置：1].", "yesno")
    assert result.status == "derived" and result.answer.startswith("结论：是。\nYes, both")


def test_no_verdict_is_missing_and_a_refusal_gets_no_line():
    answer = "两篇报道的侧重点不同 [来源：a｜位置：1]。"
    assert md.check_conclusion("两篇报道是否一致？", answer, "yesno").status == "missing"
    declined = "当前知识库没有足够依据回答这个问题。"
    result = md.check_conclusion("是否需要审批？", declined, "yesno")
    assert result.status == "declined" and result.answer == declined
    assert md.check_conclusion("住宿费超标需要谁审批？", answer, "one").status == "none"


def test_the_verdict_reply_opens_or_replaces_the_line():
    q = "两篇报道是否一致？"
    assert md.with_verdict(q, "两篇报道侧重点不同。", "否").answer == "结论：否。\n两篇报道侧重点不同。"
    assert md.with_verdict(q, "结论：部分一致\n正文", "否").answer == "结论：否。\n正文"
    assert md.with_verdict(q, "正文", "看情况").status == "unresolved"


def test_one_conclusion_line_per_part_is_combined():
    answer = "结论：是\n甲文说了X [来源：a｜位置：1]。\n\n结论：否\n乙文没有说Y [来源：b｜位置：1]。"
    result = md.check_conclusion("Does A say X, while B says Y?", answer, "yesno")
    assert result.status == "combined" and result.verdict == "否"
    assert result.answer.startswith("结论：否。\n小问1：是\n甲文") and "小问2：否" in result.answer
    both = md.check_conclusion("Does A say X, while B says Y?", "结论：是\n…\n结论：是\n…", "yesno")
    assert both.verdict == "是"


def test_choice_conclusion_names_an_option():
    assert md.choice_options("首选是成都还是武汉？") == ["成都", "武汉"]
    assert md.choice_options("Which team scored more, the Springboks or the All Blacks?") == ["the Springboks", "the All Blacks"]
    result = md.check_conclusion("首选是成都还是武汉？", "结论：武汉\n因为……", "choice")
    assert result.answer.startswith("结论：武汉。") and result.status == "model"
    assert md.check_conclusion("首选是成都还是武汉？", "结论：重庆\n因为……", "choice").status == "missing"


def test_a_question_that_names_its_two_answers_is_either_or():
    q = "After the September report, did the December report show a consistent or inconsistent portrayal of Google?"
    assert md.question_shape(q) == "choice" and md.choice_options(q) == ["consistent", "inconsistent"]
    assert md.paired_options("did they show agreement or disagreement on rates?") == ["agreement", "disagreement"]
    assert md.paired_options("was it reported before or after the game?") == ["before", "after"]
    assert md.paired_options("did the source report a win or a loss for Dallas?") == ["win", "loss"]
    assert md.paired_options("did the X and Y articles report revenue or profit?") == []
    assert md.question_shape("Does the Verge article report a drop or a rise in sales?") == "yesno"
    # The model's word, its translation, or its negation picks the option.
    assert md.check_conclusion(q, "结论：inconsistent\n…", "choice").verdict == "inconsistent"
    assert md.check_conclusion(q, "结论：consistent\n…", "choice").verdict == "consistent"
    assert md.check_conclusion(q, "结论：不一致\n…", "choice").verdict == "inconsistent"
    assert md.check_conclusion(q, "结论：一致\n…", "choice").verdict == "consistent"
    assert md.check_conclusion(q, "结论：资料不足\n…", "choice").answer.startswith("结论：资料里的内容不足以判断。")
    assert "“consistent”或“inconsistent”" in md.conclusion_rule(q, "choice")
    assert md.with_verdict(q, "两篇报道一致。", "consistent", "choice").answer.startswith("结论：consistent。")
    assert "只输出其中一个：consistent、inconsistent 或 资料不足" in md.verdict_request(q, "…", "choice")


# ---------------------------------------------------------------- citation repair
BLOCKS = [
    {"source": "a.txt", "locator": "p1", "text": "上半年利润同比下降16.8%。"},
    {"source": "b.txt", "locator": "p1", "text": "装备制造业利润增长3.1%，占比34.3%。"},
    {"source": "c.txt", "locator": "p1", "text": "天气晴朗。"},
]


def test_a_statement_is_re_pointed_to_the_passage_with_its_numbers():
    answer = "装备制造业利润增长3.1%，占比34.3% [来源：a.txt｜位置：p1]。"
    fixed, n = md.repair_citations(answer, BLOCKS)
    assert n == 1 and fixed == "装备制造业利润增长3.1%，占比34.3% [来源：b.txt｜位置：p1]。"


def test_a_cited_passage_holding_some_facts_stays_and_the_other_is_added():
    answer = "利润下降16.8%，装备制造业增长3.1% [来源：a.txt｜位置：p1]。"
    fixed, n = md.repair_citations(answer, BLOCKS)
    assert n == 1 and "[来源：a.txt｜位置：p1][来源：b.txt｜位置：p1]" in fixed


def test_nothing_changes_when_the_citation_is_right_or_the_number_is_worked_out():
    right = "利润下降16.8% [来源：a.txt｜位置：p1]。"
    assert md.repair_citations(right, BLOCKS) == (right, 0)
    worked_out = "两者相差19.9个百分点 [来源：a.txt｜位置：p1]。"
    assert md.repair_citations(worked_out, BLOCKS) == (worked_out, 0)
    uncited = "利润增长3.1%。"
    assert md.repair_citations(uncited, BLOCKS) == (uncited, 0)


def test_words_of_the_question_and_weak_names_do_not_move_citations():
    blocks = [{"source": "x.txt", "locator": "p", "text": "He pleaded not guilty."},
              {"source": "y.txt", "locator": "p", "text": "Sam Bankman-Fried at TechCrunch Disrupt."}]
    answer = "The TechCrunch article says Sam Bankman-Fried pleaded not guilty [来源：x.txt｜位置：p]."
    assert md.repair_citations(answer, blocks, question="What did the TechCrunch article say?")[1] == 0
    assert "techcrunch" not in md.claim_facts(answer, "What did the TechCrunch article say?")


def test_passages_of_one_source_and_position_are_one_citation():
    blocks = [{"source": "a.txt", "locator": "p", "text": "利润下降。"},
              {"source": "a.txt", "locator": "p", "text": "利润下降16.8%。"},
              {"source": "b.txt", "locator": "p", "text": "增长3.1%。"}]
    answer = "利润下降16.8%，增长3.1% [来源：a.txt｜位置：p]。"
    fixed, n = md.repair_citations(answer, blocks)
    assert fixed == "利润下降16.8%，增长3.1% [来源：a.txt｜位置：p][来源：b.txt｜位置：p]。"


def test_defaults_are_the_reports_recommendation(tmp_path):
    from config import Settings

    settings = Settings(_env_file=None, data_dir=tmp_path)
    assert settings.conclusion_line == "check" and settings.citation_repair and settings.dated_evidence
    assert not settings.part_answer


# ---------------------------------------------------------------- in the answer graph
class MultiDocSettings(LeverSettings):
    context_token_budget = 5000
    part_answer = False
    dated_evidence = False
    conclusion_line = "off"
    citation_repair = False


PASSAGES = {
    "pa": ("a" * 32, "大会.docx", "黑龙江省在大会上发挥了桥梁作用，签约项目20个。"),
    "pb": ("b" * 32, "冰雪节.docx", "哈尔滨冰雪节今年吸引游客300万人次。"),
}


class TwoDocRetriever:
    def retrieve(self, query, mode="hybrid_rerank", **kwargs):
        return [SearchHit(id=f"c-{key}", score=0.9, text=text, doc_id=doc, parent_id=key, source=source,
                          heading="", locator="p1", retrieval_method="hybrid_rrf_bge_rerank", rerank_score=0.9 - i * 0.1)
                for i, (key, (doc, source, text)) in enumerate(PASSAGES.items())]


class TwoDocParents:
    def get(self, parent_id):
        doc, source, text = PASSAGES[parent_id]
        return {"parent_id": parent_id, "doc_id": doc, "source": source, "locator": "p1", "text": text}


class ScriptedLLM(FakeLLM):
    """The answer model says ``answer``; the verdict call says ``verdict``; every prompt is kept."""

    def __init__(self, answer, verdict="否"):
        super().__init__(grounded=True)
        self.answer, self.verdict, self.prompts = answer, verdict, []

    def invoke(self, messages):
        system = str(messages[0].content)
        user = "\n".join(str(message.content) for message in messages[1:])
        self.prompts.append((system, user))
        if system == md.VERDICT_PROMPT:
            return AIMessage(content=self.verdict)
        if "基于用户知识库文档回答问题" in system:
            return AIMessage(content=self.answer)
        return super().invoke(messages)

    def answer_prompt(self):
        return next(user for system, user in self.prompts if "基于用户知识库文档回答问题" in system)


def ask(llm, settings, question, **graph_args):
    graph = create_agent_graph(llm=llm, retriever=TwoDocRetriever(), settings=settings, parent_store=TwoDocParents(),
                               **graph_args)
    return graph.invoke({"messages": [HumanMessage(content=question)]}, config={"configurable": {"thread_id": "t"}})


QUESTION = "黑龙江省在大会上发挥了哪些作用，并且哈尔滨冰雪节吸引了多少游客？"
ANSWER = "小问1：发挥了桥梁作用 [来源：大会.docx｜位置：p1]。\n小问2：300万人次 [来源：冰雪节.docx｜位置：p1]。"


def test_switches_off_change_nothing():
    llm = ScriptedLLM(ANSWER)
    result = ask(llm, MultiDocSettings(), QUESTION)
    prompt = llm.answer_prompt()
    assert "【小问1 的证据】" not in prompt and "文件日期" not in prompt and "结论：" not in prompt
    assert result["answer"] == ANSWER and result["answer_steps"] == {}


def test_part_answer_groups_the_evidence_in_one_call():
    settings = MultiDocSettings()
    settings.part_answer = True
    llm = ScriptedLLM(ANSWER)
    result = ask(llm, settings, QUESTION)
    prompt = llm.answer_prompt()
    assert "【小问1 的证据】" in prompt and "【小问2 的证据】" in prompt and md.PART_GUIDANCE in prompt
    assert prompt.index("【小问1 的证据】") < prompt.index("大会.docx") < prompt.index("【小问2 的证据】") < prompt.index("冰雪节.docx")
    assert sum("基于用户知识库文档回答问题" in system for system, _ in llm.prompts) == 1
    assert result["answer_steps"]["parts"] == 2 and result["answer"] == ANSWER
    # A question that asks one thing keeps the usual evidence.
    llm = ScriptedLLM("桥梁作用 [来源：大会.docx｜位置：p1]。")
    ask(llm, settings, "黑龙江省在大会上发挥了什么作用？")
    assert "【小问1 的证据】" not in llm.answer_prompt()


def test_conclusion_line_from_the_model_and_from_the_verdict_call():
    settings = MultiDocSettings()
    settings.conclusion_line = "prompt"
    llm = ScriptedLLM("结论：否\n冰雪节吸引游客300万人次 [来源：冰雪节.docx｜位置：p1]。")
    result = ask(llm, settings, "哈尔滨冰雪节的游客是否超过500万人次？")
    assert md.CONCLUSION_YESNO in llm.answer_prompt()
    assert result["answer"].startswith("结论：否。\n") and result["answer_steps"]["conclusion"] == "model"
    assert not any(system == md.VERDICT_PROMPT for system, _ in llm.prompts)
    # No verdict in the answer: one short call names it.
    llm = ScriptedLLM("冰雪节吸引游客300万人次 [来源：冰雪节.docx｜位置：p1]。", verdict="否")
    result = ask(llm, settings, "哈尔滨冰雪节的游客是否超过500万人次？")
    assert result["answer"].startswith("结论：否。\n冰雪节") and result["answer_steps"]["conclusion"] == "asked"
    assert sum(system == md.VERDICT_PROMPT for system, _ in llm.prompts) == 1


def test_check_mode_keeps_the_prompt_and_only_asks_when_the_answer_gives_no_verdict():
    settings = MultiDocSettings()
    settings.conclusion_line = "check"
    committed = "不是，冰雪节吸引游客300万人次 [来源：冰雪节.docx｜位置：p1]。"
    llm = ScriptedLLM(committed)
    result = ask(llm, settings, "哈尔滨冰雪节的游客是否超过500万人次？")
    assert md.CONCLUSION_YESNO not in llm.answer_prompt()
    assert result["answer"] == committed and result["answer_steps"]["conclusion"] == "had"
    assert not any(system == md.VERDICT_PROMPT for system, _ in llm.prompts)
    llm = ScriptedLLM("冰雪节吸引游客300万人次 [来源：冰雪节.docx｜位置：p1]。", verdict="否")
    result = ask(llm, settings, "哈尔滨冰雪节的游客是否超过500万人次？")
    assert result["answer"].startswith("结论：否。\n") and result["answer_steps"]["conclusion"] == "asked"


def test_reading_the_answers_own_verdict():
    q = "Does A say X, while B says Y?"
    assert md.gives_verdict(q, "Does A say X?\nYes, A says X [来源：a｜位置：1].\nDoes B say Y?\nNo.", "yesno")
    # Markdown line breaks ("?  \n") between the repeated question and its answer.
    assert md.gives_verdict(q, "1. **Does A say X?**  \nNo, A does not say X. Instead, it says Z.", "yesno")
    assert md.gives_verdict(q, "小问1：是，A 说了 X。\n小问2：否。", "yesno")
    assert md.gives_verdict(q, "注意：引用的资料里没有出现“2023”。\n\nNo, B does not say Y.", "yesno")
    assert not md.gives_verdict(q, "The two reports differ in focus [来源：a｜位置：1].", "yesno")
    assert not md.gives_verdict(q, "当前知识库没有足够依据。", "yesno")
    c = "Did the December report show a consistent or inconsistent portrayal?"
    assert md.gives_verdict(c, "The portrayal is consistent: both say X.", "choice")
    assert md.gives_verdict(c, "The portrayal is inconsistent with the earlier one.", "choice")
    assert not md.gives_verdict(c, "Both reports discuss Google's market power.", "choice")


def test_dated_evidence_shows_the_file_date_on_time_questions_only():
    settings = MultiDocSettings()
    settings.dated_evidence = True
    dates = {"a" * 32: "2023-10-12T08:00:00+00:00", "b" * 32: "2023-12-06"}
    llm = ScriptedLLM("桥梁作用 [来源：大会.docx｜位置：p1]。")
    result = ask(llm, settings, "10月12日的大会报道说黑龙江省发挥了什么作用？",
                 document_dates=lambda ids: {i: dates[i] for i in ids if i in dates})
    prompt = llm.answer_prompt()
    assert "（文件日期：2023-10-12）\n黑龙江省" in prompt and "（文件日期：2023-12-06）" in prompt
    assert result["answer_steps"]["dated"] == 2
    # The fact check sees the same dated passages.
    check = next(user for system, user in llm.prompts if "事实一致性检查器" in system)
    assert "文件日期：2023-10-12" in check
    llm = ScriptedLLM("桥梁作用 [来源：大会.docx｜位置：p1]。")
    ask(llm, settings, "黑龙江省在大会上发挥了什么作用？", document_dates=lambda ids: {i: dates[i] for i in ids if i in dates})
    assert "文件日期" not in llm.answer_prompt()


def test_dated_evidence_uses_the_librarys_file_dates_in_the_app():
    nodes = AgentNodes(llm=None, retriever=None, settings=MultiDocSettings(),
                       documents=SimpleNamespace(rows=lambda: [{"file_id": "a" * 32, "mtime": 1697068800.0},
                                                               {"file_id": "z", "mtime": 1.0}]))
    dates = nodes._document_dates({"a" * 32})
    assert set(dates) == {"a" * 32} and dates["a" * 32].startswith("2023-10-1")


def test_citation_repair_runs_before_the_fact_check():
    settings = MultiDocSettings()
    settings.citation_repair = True
    llm = ScriptedLLM("冰雪节吸引游客300万人次 [来源：大会.docx｜位置：p1]。")
    result = ask(llm, settings, "哈尔滨冰雪节吸引了多少游客？")
    assert result["answer"] == "冰雪节吸引游客300万人次 [来源：冰雪节.docx｜位置：p1]。"
    assert result["answer_steps"]["repaired"] == 1
