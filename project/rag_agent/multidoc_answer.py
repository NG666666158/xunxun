"""Answers that draw on several documents, for a weak answer model.

Retrieval now brings the documents a multi-document question needs into the context
(evaluation/multidoc_v1/REPORT.md), but Qwen3-8B often fails to put them together: it answers
one part and forgets the other, mixes up which report said what, describes both sides of a
yes / no question without saying yes or no, and cites one passage for numbers that come from
another (evaluation/release_1_3/REPORT.md). Four helpers, each behind its own setting and
measured in evaluation/multidoc_answer_v1/REPORT.md; all of them are rules (no model call of
their own, except one short call when a yes / no answer gives no verdict at all):

* ``part_answer``: a question that asks several things gets its evidence grouped per part
  ("小问1 的证据"), in the same single answer call, and is answered part by part;
* ``dated_evidence``: a question about time ("before / after", "10月12日的报道") sees each
  passage's file date (the publish date of a news corpus);
* ``conclusion_line``: a yes / no or either-or question opens with one conclusion line
  ("结论：是"), checked and normalised by the code;
* ``citation_repair``: a statement whose cited passage lacks its numbers or names cites the
  passage that has them instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Iterable, Sequence

from retrieval.multipart import distinct_terms
from retrieval.question_parts import ASK_ZH, mentions, question_parts

from .answering import CITATION, evidence_for, multi_part_question

_CJK = re.compile(r"[㐀-鿿]")

# ---------------------------------------------------------------- question shapes
_AUX_EN = r"(?:does|did|do|is|was|were|are|has|have|had|can|could|will|would|should)"
_WH_EN = r"(?:who|whom|whose|what|which|when|where|why|how)"
_WH_START = re.compile(rf"^\s*{_WH_EN}\b", re.I)
_AUX_START = re.compile(rf"^\s*{_AUX_EN}\b", re.I)
# "After the report …, did …", "Between A and B, was there …": a leading clause, then the ask.
_LEAD_EN = re.compile(
    r"^\s*(?:after|before|between|following|in|according|considering|given|comparing|since|from|regarding|"
    r"based on|prior to|during|as)\b",
    re.I,
)
_AUX_AFTER_COMMA = re.compile(rf",\s*{_AUX_EN}\s+(?:the|there|it|this|that|these|those|they|a|an|both|any|all|[A-Z])", re.I)
# A second ask: "…, and when did it …", "…; what is …" (", who reportedly …" is a relative clause).
_SECOND_WH_EN = re.compile(rf"(?:[,;]\s*and\s+|;\s*|\band\s+){_WH_EN}\s+{_AUX_EN}\b", re.I)
_CHOICE_EN = re.compile(r"\b([^,?]{1,60}?)\s+or\s+([^,?]{1,60}?)\s*\?*\s*$", re.I)
_YESNO_ZH = re.compile(r"是否|是不是|有没有|能不能|会不会|对不对|可不可以|能否|可否|有无|吗\s*[？?]?\s*$")
_CHOICE_ZH = re.compile(r"还是|(?:谁|哪个|哪一个|哪家|哪位|哪一方|哪方|哪一年|哪国)(?:更|最|先|较|比较)")
# A yes / no question that names its two answers: "… consistent or inconsistent?", "… show agreement or
# disagreement …", "… before or after …", "… a win or a loss …" (the answer is one of the two words).
_PAIRED_EN = re.compile(r"\b([a-z][a-z-]{2,})\s*(?:\bor\b|/)\s*(?:an?\s+|the\s+)?([a-z][a-z-]{2,})\b", re.I)
_OPPOSITES = {
    frozenset(pair) for pair in (
        ("before", "after"), ("win", "loss"), ("same", "different"), ("more", "less"), ("higher", "lower"),
        ("increase", "decrease"), ("positive", "negative"), ("consistent", "change"), ("earlier", "later"),
        ("rise", "fall"), ("up", "down"), ("gain", "loss"), ("similar", "different"), ("true", "false"),
    )
}
_NEGATED = re.compile(r"^(?:in|un|dis|non-?|im|ir)")


def paired_options(question: str) -> list[str]:
    """The two answers a question names ("consistent or inconsistent" -> ["consistent", "inconsistent"])."""
    for match in _PAIRED_EN.finditer(question or ""):
        first, second = match.group(1), match.group(2)
        a, b = first.lower(), second.lower()
        if a == b:
            continue
        if _NEGATED.sub("", b) == a or _NEGATED.sub("", a) == b or frozenset((a, b)) in _OPPOSITES:
            return [first, second]
    return []


def question_shape(question: str) -> str:
    """"yesno", "choice", "asks" (several things asked at once) or "one" (one thing asked).

    "Which company, reported by X and Y, has also …" asks one thing through several clues: it is
    "one", although its clauses look like parts.
    """
    text = (question or "").strip()
    if not text:
        return "one"
    if _CJK.search(text):
        other_asks = [m for m in ASK_ZH.finditer(text) if m.group(0) != "是否"]
        if _CHOICE_ZH.search(text):
            return "choice"
        if _YESNO_ZH.search(text) and not other_asks:
            return "yesno"
        return "asks" if multi_part_question(text) else "one"
    if paired_options(text):
        return "choice"
    if _WH_START.match(text):
        if re.match(r"^\s*(?:which|who)\b", text, re.I) and choice_options(text):
            return "choice"
        return "asks" if _SECOND_WH_EN.search(text) else "one"
    if _AUX_START.match(text) or (_LEAD_EN.match(text) and _AUX_AFTER_COMMA.search(text)):
        return "yesno"
    # "Considering the articles by A and B, which character …?": clauses of context, one ask.
    return "asks" if re.search(rf"(?:^|[,;:]\s*){_WH_EN}\b", text, re.I) and _SECOND_WH_EN.search(text) else "one"


# ---------------------------------------------------------------- per-part evidence
@dataclass
class PartGroups:
    """The evidence of a multi-part question grouped per part (indexes into the evidence)."""

    parts: list[str]
    own: list[list[int]]  # the passages printed under each part (each passage under one part)
    also: list[list[int]]  # passages printed under another part that this part may use too
    other: list[int]  # passages that match no part


def group_passages(parts: Sequence[str], texts: Sequence[str]) -> PartGroups:
    """Each passage goes under the part whose distinctive words it mentions most (the coverage
    rule retrieval uses, retrieval/multipart.py); a passage that also mentions half of another
    part's words is pointed to from there. No model call."""
    terms = distinct_terms(list(parts))
    own: list[list[int]] = [[] for _ in parts]
    also: list[list[int]] = [[] for _ in parts]
    other: list[int] = []
    for index, text in enumerate(texts):
        shares = [mentions(text, words) for words in terms]
        best = max(shares) if shares else 0.0
        if best <= 0:
            other.append(index)
            continue
        home = shares.index(best)
        own[home].append(index)
        for i, share in enumerate(shares):
            if i != home and (share == best or share >= 0.5):
                also[i].append(index)
    return PartGroups(list(parts), own, also, other)


def grouped_evidence(groups: PartGroups, blocks: Sequence[str]) -> str:
    """The evidence blocks (numbered as usual) printed under the part they belong to."""
    lines = [f"这个问题有 {len(groups.parts)} 个小问，证据已按小问分好组（同一段证据可能对几个小问都有用）："]
    for n, part in enumerate(groups.parts, start=1):
        lines.append(f"小问{n}：{part}")
    for n, part in enumerate(groups.parts, start=1):
        lines.append(f"\n【小问{n} 的证据】")
        if not groups.own[n - 1] and not groups.also[n - 1]:
            lines.append("（没有专门讲这一小问的证据，请看其它小问的证据和其它证据）")
        for index in groups.own[n - 1]:
            lines.append(blocks[index])
        if groups.also[n - 1]:
            lines.append("（另见" + "、".join(f"证据 {index + 1}" for index in groups.also[n - 1]) + "）")
    if groups.other:
        lines.append("\n【其它证据】")
        lines.extend(blocks[index] for index in groups.other)
    return "\n".join(lines)


def part_mode(question: str, shape: str | None = None) -> list[str]:
    """The parts to answer one by one: a question that asks several things, or a yes / no or
    either-or question made of several parts ([] otherwise)."""
    shape = shape or question_shape(question)
    if shape not in ("asks", "yesno", "choice"):
        return []
    parts = question_parts(question)
    return parts if len(parts) >= 2 else []


PART_GUIDANCE = (
    "本题包含几个小问，证据已经按小问分好组。按小问的顺序逐一回答：每个小问单独一段，段首写“小问N：”和一句直接答案，"
    "再补充这一问需要的数字、条件，每句末尾标注出处。回答某个小问时先看这一小问下的证据，没有答案再看其它证据；"
    "都没有时写“资料里没有找到这一问的答案”，不要拿讲别的对象的资料来凑。不要补充问题没有问到的内容；全文控制在 500 字左右。"
)
PART_GUIDANCE_YESNO = (
    "本题由几部分组成，证据已经按部分分好组。先逐部分判断：每部分单独一段，段首写“小问N：是”或“小问N：否”，"
    "再用这一部分的证据说明理由，每句末尾标注出处；这一部分的证据没有讲到时写“小问N：资料不足”。"
    "每一部分都成立，整个问题才成立。全文控制在 500 字左右。"
)


# ---------------------------------------------------------------- dates
_MONTHS_EN = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
# "latest installment", "since then", "previously" are not about when reports were written: with them
# 62 of 301 MultiHop null questions counted as time questions, without them 11 (temporal 240 of 250
# either way).
_TIME_EN = re.compile(
    rf"\b(?:before|after|earlier|later|prior to|subsequent(?:ly)?|published on|reported on|as of)\b|"
    rf"\b(?:{_MONTHS_EN})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d\d\b|\b(?:19|20)\d\d-\d\d-\d\d\b",
    re.I,
)
_TIME_ZH = re.compile(
    r"之前|之后|以前|以后|此前|此后|早于|晚于|先后|哪个先|谁先|最早|最晚|最先|最新|更早|更晚|随后|期间|"
    r"(?:19|20)\d\d年\d{1,2}月|\d{1,2}月\d{1,2}[日号]|什么时候|何时|哪一年|哪一天|日期"
)


def temporal_question(question: str) -> bool:
    """Does the question turn on when something was written or happened ("before / after",
    "the report published on October 12, 2023", "哪个先", "10月12日的报道")?"""
    text = question or ""
    return bool(_TIME_ZH.search(text) or _TIME_EN.search(text))


def date_text(value: Any) -> str:
    """"2023-10-12" from an epoch (a file's mtime) or an ISO date / datetime string."""
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            return ""
    match = re.match(r"\s*((?:19|20)\d\d)[-/.](\d{1,2})[-/.](\d{1,2})", str(value))
    return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}" if match else ""


def dated_text(text: str, date: str) -> str:
    """A passage as the answer and the checks read it when its file date is shown."""
    return f"（文件日期：{date}）\n{text}" if date else text


# ---------------------------------------------------------------- conclusion line
CONCLUSION_YESNO = (
    "这是一个是非题。回答的第一行只写“结论：是”或“结论：否”，不要写别的词；问题由几部分组成时，每一部分都成立才写“是”，"
    "有一部分不成立就写“否”；证据完全没有讲到时写“结论：资料不足”。“结论：”全文只写这一次，分部分说明时用“小问1：”“小问2：”开头。"
    "从第二行起说明依据，每句末尾标注出处。"
)
CONCLUSION_CHOICE = (
    "这是一个二选一的问题。回答的第一行写“结论：”加上选中的那一项，照抄问题里的原词{options}；"
    "证据不足以判断时写“结论：资料不足”。“结论：”全文只写这一次。从第二行起说明依据，每句末尾标注出处。"
)
VERDICT_LINES = {"是": "结论：是。", "否": "结论：否。", "资料不足": "结论：资料里的内容不足以判断。"}
VERDICT_PROMPT = "你判断一段回答对问题给出的结论。只输出要求的那一个词，不要解释。"

_CONCLUSION = re.compile(r"^\W*(?:结论|答案|回答|总结|判断|conclusion|answer|verdict)\W*?[:：]\s*(.*)$", re.I)
_YES_VALUE = re.compile(r"^(?:是的?|对|成立|属实|正确|yes|true|correct)(?:[\W_]|$)", re.I)
_NO_VALUE = re.compile(r"^(?:否|不是|不对|不成立|不属实|no|false|incorrect)(?:[\W_]|$)", re.I)
_UNKNOWN_VALUE = re.compile(r"^(?:资料不足|证据不足|信息不足|无法判断|无法确定|不确定|资料里的内容不足以|insufficient)", re.I)
# The answer already declines ("当前知识库没有足够依据"): no conclusion is added to it.
_DECLINED = re.compile(r"当前知识库没有足够依据|资料里没有找到|资料里的内容不足以|没有足够(?:的)?依据")
# A value saying the negative of a pair ("不一致" for consistent / inconsistent).
_NEGATIVE_VALUE = re.compile(r"不|没有|未|非|无|否|disagree|inconsistent|unsuccessful|\bnot?\b", re.I)


def conclusion_rule(question: str, shape: str) -> str:
    """The answer's conclusion instruction for a yes / no or either-or question ("" otherwise)."""
    if shape == "yesno":
        return CONCLUSION_YESNO
    if shape == "choice":
        options = choice_options(question)
        named = f"（“{options[0]}”或“{options[1]}”）" if len(options) == 2 else ""
        return CONCLUSION_CHOICE.format(options=named)
    return ""


def _plain_line(line: str) -> str:
    return re.sub(r"[*_`#>]+", "", CITATION.sub("", line)).strip(" \t-：:")


def verdict_of(text: str) -> str | None:
    """"是" / "否" / "资料不足" for a conclusion value; None when it says something else."""
    value = _plain_line(text)
    if _UNKNOWN_VALUE.match(value):
        return "资料不足"
    if _NO_VALUE.match(value):
        return "否"
    if _YES_VALUE.match(value):
        return "是"
    return None


def choice_options(question: str) -> list[str]:
    """The two options of an either-or question ("A还是B", "…, A or B?", "… consistent or
    inconsistent …"), when written out."""
    text = (question or "").strip().rstrip("？?。 ")
    if _CJK.search(text):
        match = re.search(r"([^，,；;、]{1,20})还是([^，,；;、]{1,20})$", text)
        if match:
            # "首选是成都还是武汉": the first option starts after the last "是 / 为 / 在 / 选".
            first = re.sub(r"^.*(?:是|为|在|选)", "", match.group(1)) or match.group(1)
            return [first.strip(), re.sub(r"(?:更|呢|吗).*$", "", match.group(2)).strip()]
        return []
    paired = paired_options(text)
    if paired:
        return paired
    # "Which …, A or B?": the options after the last comma.
    if "," not in text:
        return []
    tail = text.rsplit(",", 1)[-1]
    match = _CHOICE_EN.search(tail + "?") if len(tail) <= 90 else None
    return [match.group(1).strip(), match.group(2).strip()] if match else []


@dataclass
class Conclusion:
    answer: str
    # "model" (its own line), "combined" (its lines per part, combined by the code), "derived" (from
    # its first sentence), "asked" (the short verdict call), "missing", "unresolved", "declined", "none"
    status: str
    verdict: str = ""


def _pick(value: str, options: list[str]) -> str:
    """The option a conclusion value names: by its words, or, for a pair of a word and its
    negation ("consistent / inconsistent"), by whether the value is negative ("不一致")."""
    value = _plain_line(value).rstrip("。.")
    if not value:
        return ""
    if not options:
        return value if len(value) <= 40 else ""
    low = value.lower()
    named = [o for o in options if o and o.lower() in low]
    if len(named) == 1 or (len(named) == 2 and named[0].lower() in named[1].lower()):
        # "inconsistent" contains "consistent": the longer one is the one named.
        return max(named, key=len)
    if len(options) == 2 and verdict_of(value) is None:
        lows = {o.lower() for o in options}
        negative = [o for o in options if _NEGATED.match(o.lower()) and _NEGATED.sub("", o.lower()) in lows]
        if negative:
            positive = next(o for o in options if o not in negative)
            return negative[0] if _NEGATIVE_VALUE.search(value) else positive
    return ""


def _conclusion_lines(lines: list[str]) -> list[tuple[int, str]]:
    found = []
    for index, line in enumerate(lines):
        match = _CONCLUSION.match(_plain_line(line))
        if match:
            found.append((index, match.group(1)))
    return found


def check_conclusion(question: str, answer: str, shape: str) -> Conclusion:
    """The answer of a yes / no or either-or question with one normalised conclusion line first.

    * the model wrote "结论：…" with a clear value: the line becomes "结论：是。" (否 / 资料不足 / the
      chosen option);
    * it wrote one "结论：" line per part: the code combines them (one "否" makes the whole "否"; the
      question asks whether every part holds), puts that line first, and the part lines read
      "小问1：…";
    * no such line, but the first sentence starts with a verdict ("是的，…", "No, …"): the line is
      added from it;
    * otherwise "missing": the caller asks the model for the verdict in one short call
      (``with_verdict``) or leaves the answer as it is.
    """
    if shape not in ("yesno", "choice"):
        return Conclusion(answer, "none")
    lines = answer.splitlines()
    first = next((i for i, line in enumerate(lines) if _plain_line(line)), None)
    if first is None:
        return Conclusion(answer, "missing")
    head = _plain_line(lines[first])
    if _DECLINED.search(head):
        return Conclusion(answer, "declined")
    found = _conclusion_lines(lines)
    if shape == "choice":
        options = choice_options(question)
        for _, value in found:
            if verdict_of(value) == "资料不足":
                return _with_line(lines, found, VERDICT_LINES["资料不足"], "model", "资料不足")
            option = _pick(value, options)
            if option:
                return _with_line(lines, found, f"结论：{option}。", "model", option)
        return Conclusion(answer, "missing")
    verdicts = [verdict_of(value) for _, value in found]
    known = [v for v in verdicts if v]
    if len(found) >= 2 and known:
        overall = "否" if "否" in known else "资料不足" if "资料不足" in known else "是"
        for n, (index, value) in enumerate(found, start=1):
            lines[index] = f"小问{n}：{_plain_line(value)}"
        return Conclusion(f"{VERDICT_LINES[overall]}\n" + "\n".join(lines), "combined", overall)
    if found and known:
        return _with_line(lines, found, VERDICT_LINES[known[0]], "model", known[0])
    if not found:
        sentence = re.split(r"[。！!？?\n]", head, maxsplit=1)[0]
        verdict = verdict_of(sentence)
        if verdict in ("是", "否"):
            return Conclusion(f"{VERDICT_LINES[verdict]}\n{answer}", "derived", verdict)
    return Conclusion(answer, "missing")


def _with_line(lines: list[str], found: list[tuple[int, str]], line: str, status: str, verdict: str) -> Conclusion:
    """``line`` first: in place of the model's own conclusion line when that opens the answer."""
    first = next((i for i, text in enumerate(lines) if _plain_line(text)), 0)
    if found and found[0][0] == first:
        lines = [*lines[:first], line, *lines[first + 1:]]
        return Conclusion("\n".join(lines), status, verdict)
    return Conclusion(line + "\n" + "\n".join(lines), status, verdict)


# A sentence that opens with a yes / no ("是的，…", "No, the article …", "小问2：否").
_VERDICT_HEAD = re.compile(r"^(?:小问\s*\d+\s*[:：]\s*)?(?:是的?|否|不是|yes|no)(?:[\W_]|$)", re.I)


def gives_verdict(question: str, answer: str, shape: str) -> bool:
    """Does the answer already give its verdict in its own words? A conclusion line, a part line
    ("小问1：是"), or a sentence opening with yes / no as the answer's first sentence or right after
    a sentence that repeats the question; for an either-or question, one of the two options named
    in its first sentences."""
    text = CITATION.sub("", answer or "")
    sentences = [s.strip().strip("*#>-").strip() for s in re.findall(r"[^。！!？?\n]+[。！!？?\n]?", text)[:24]]
    sentences = [s for s in sentences if s and not s.startswith(("注意：", "资料里写的是"))]
    if not sentences or _DECLINED.search(sentences[0]):
        return False
    if shape == "choice":
        options = choice_options(question)
        if len(options) != 2:
            return True  # nothing to check against
        for sentence in sentences[:3]:
            if sentence.endswith(("?", "？")):
                continue
            match = _CONCLUSION.match(sentence)
            if _pick(match.group(1) if match else sentence, options) and (match or _named_once(sentence, options)):
                return True
        return False
    after_question = True
    for sentence in sentences:
        if sentence.endswith(("?", "？")):
            after_question = True
            continue
        match = _CONCLUSION.match(sentence)
        if match and verdict_of(match.group(1)) in ("是", "否"):
            return True
        if (after_question or re.match(r"^小问\s*\d+", sentence)) and _VERDICT_HEAD.match(sentence):
            return True
        after_question = False
    return False


def _named_once(sentence: str, options: list[str]) -> bool:
    """Exactly one of the two options is named ("inconsistent" also contains "consistent")."""
    low = sentence.lower()
    named = [o for o in options if o.lower() in low]
    return len(named) == 1 or (len(named) == 2 and named[0].lower() in named[1].lower() and low.count(named[0].lower()) == 1)


def with_verdict(question: str, answer: str, reply: str, shape: str = "yesno") -> Conclusion:
    """The answer with the conclusion line from the short verdict call (unchanged if unclear)."""
    text = reply.strip().splitlines()[0] if reply.strip() else ""
    lines = answer.splitlines()
    found = _conclusion_lines(lines)
    if shape == "choice":
        verdict = "资料不足" if verdict_of(text) == "资料不足" else _pick(text, choice_options(question))
        if not verdict:
            return Conclusion(answer, "unresolved")
        line = VERDICT_LINES["资料不足"] if verdict == "资料不足" else f"结论：{verdict}。"
        return _with_line(lines, found, line, "asked", verdict)
    verdict = verdict_of(text)
    if not verdict:
        return Conclusion(answer, "unresolved")
    # Replaces "结论：部分一致" and the like when the model wrote such a line first.
    return _with_line(lines, found, VERDICT_LINES[verdict], "asked", verdict)


def verdict_request(question: str, answer: str, shape: str = "yesno") -> str:
    body = CITATION.sub("", answer).strip()
    options = choice_options(question) if shape == "choice" else []
    if len(options) == 2:
        ask = (f"这段回答选的是“{options[0]}”还是“{options[1]}”，还是资料不足？"
               f"只输出其中一个：{options[0]}、{options[1]} 或 资料不足。")
    else:
        ask = "这段回答的结论是“是”“否”，还是“资料不足”？只输出其中一个词。"
    return f"问题：{question}\n\n回答：\n{body[:1500]}\n\n{ask}"


# ---------------------------------------------------------------- citation repair
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_YEAR = re.compile(r"(?:19|20)\d\d")
_MODEL_NAME = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z\-]*\d[A-Za-z0-9\-.]*|(?<![A-Za-z0-9])[A-Z]{2,}(?![A-Za-z])")
_QUOTED = re.compile(r"[《“\"「]([^》”\"」]{2,30})[》”\"」]")
_LATIN_WORD = re.compile(r"(?<![A-Za-z])[A-Za-z][A-Za-z'’.&-]{2,}")
_CAPITALISED = re.compile(r"(?<=[a-z0-9,;:)]\s)[A-Z][a-zA-Z'’&-]{2,}")
_EN_STOP = frozenset("the this that these those and but for with from also however while after before according".split())
_FULL_WIDTH = {ord(f): ord(h) for f, h in zip("０１２３４５６７８９．％", "0123456789.%")}


def _normal(text: str) -> str:
    text = CITATION.sub("", text or "").translate(_FULL_WIDTH)
    text = re.sub(r"(?<=\d)[,，](?=\d{3})", "", text)
    return re.sub(r"\s+", " ", text).lower()


def _number_key(number: str) -> str:
    return number.rstrip("0").rstrip(".") if "." in number else number


def claim_facts(claim: str, question: str = "") -> dict[str, bool]:
    """The checkable facts of a statement, each marked strong or not: numbers (two digits or
    more, or a decimal; not years), model names and acronyms, quoted names, and Latin names
    (in a Chinese statement every Latin word, in an English one the capitalised words inside the
    sentence). A bare two-digit number and an English capitalised word are weak: they say little
    about which passage is meant. What the question itself says is given, not a fact the
    passage must hold ("the TechCrunch article …")."""
    raw = CITATION.sub("", claim or "")
    body = _normal(raw)
    asked = _normal(question)
    asked_numbers = {_number_key(n) for n in _NUMBER.findall(asked)}
    facts: dict[str, bool] = {}
    for number in _NUMBER.findall(body):
        if _YEAR.fullmatch(number) or (len(number) < 2 and "." not in number):
            continue
        key = _number_key(number)
        if key not in asked_numbers:
            facts[key] = facts.get(key, False) or "." in number or len(number) >= 3
    for term in _MODEL_NAME.findall(raw):
        # "A100", "GPT-4" name one thing; an acronym ("FTX", "CEO") is mostly the subject.
        facts[term.lower().strip(".-")] = bool(re.search(r"\d", term))
    for quoted in _QUOTED.findall(raw):
        facts[_normal(quoted).strip(" ,，。.")] = True
    chinese = bool(_CJK.search(raw))
    for word in _LATIN_WORD.findall(raw) if chinese else _CAPITALISED.findall(raw):
        word = word.lower().strip("'’.-")
        if word.endswith(("'s", "’s")):
            word = word[:-2]
        if len(word) >= 3 and word not in _EN_STOP:
            facts.setdefault(word, chinese and len(word) >= 4)
    return {fact: strong for fact, strong in facts.items() if fact and not (_NUMBER.fullmatch(fact) is None and fact in asked)}


def _passage_facts(text: str) -> tuple[str, set[str]]:
    body = _normal(text)
    return body, {_number_key(n) for n in _NUMBER.findall(body)}


def _has(fact: str, passage: tuple[str, set[str]]) -> bool:
    body, numbers = passage
    if _NUMBER.fullmatch(fact):
        return fact in numbers
    return fact in body


def repair_citations(answer: str, blocks: Sequence[dict[str, Any]], question: str = "",
                     max_added: int = 2) -> tuple[str, int]:
    """Re-point the citations of statements whose cited passages lack their numbers or names.

    For each cited statement (as ``cited_claims`` splits them): when a fact of it is in none of
    its cited passages, the passages that hold the missing facts are cited instead of the cited
    passages that hold none (cited passages that hold some stay). Up to ``max_added`` citations
    are added, each holding at least one strong missing fact (or all the missing ones). Numbers
    the answer worked out itself are in no passage and change nothing. Passages with the same
    source and position are one citation. Returns the answer and how many statements were
    re-pointed.
    """
    if not answer or not blocks:
        return answer, 0
    keys: dict[tuple[str, str], list[int]] = {}
    for index, block in enumerate(blocks):
        keys.setdefault((str(block.get("source", "")), str(block.get("locator", ""))), []).append(index)
    cites = list(keys)
    # The file name is part of what a citation shows ("SBF, riding high on FTX, …").
    texts = [key[0] + "\n" + "\n".join(str(blocks[i].get("text", "")) for i in keys[key]) for key in cites]
    passages = [_passage_facts(text) for text in texts]
    key_of = {i: n for n, key in enumerate(cites) for i in keys[key]}
    context = _Repair(cites, passages, key_of, question, max_added, list(blocks))
    changed = 0
    out_lines = []
    for line in answer.split("\n"):
        matches = list(CITATION.finditer(line))
        if not matches:
            out_lines.append(line)
            continue
        runs: list[list[re.Match]] = []
        for match in matches:
            between = line[runs[-1][-1].end() : match.start()] if runs else None
            if runs and not between.strip(" \t，,。；;、"):
                runs[-1].append(match)
            else:
                runs.append([match])
        pieces, position = [], 0
        for run in runs:
            start, end = run[0].start(), run[-1].end()
            claim = line[position:start]
            refs = [(m.group(1).strip(), (m.group(2) or "").strip()) for m in run]
            replacement = context.repaired(claim, refs)
            pieces.append(line[position:start])
            if replacement is None:
                pieces.append(line[start:end])
            else:
                pieces.append(replacement)
                changed += 1
            position = end
        pieces.append(line[position:])
        out_lines.append("".join(pieces))
    return "\n".join(out_lines), changed


@dataclass
class _Repair:
    cites: list[tuple[str, str]]  # (source, locator) of each citation
    passages: list[tuple[str, set[str]]]  # what each citation's passages hold
    key_of: dict[int, int]  # evidence block -> citation
    question: str
    max_added: int
    blocks: list[dict[str, Any]]

    def repaired(self, claim: str, refs) -> str | None:
        facts = claim_facts(claim, self.question)
        if not facts:
            return None
        indexes, _ = evidence_for(refs, self.blocks)
        cited = list(dict.fromkeys(self.key_of[i] for i in indexes))
        holds = lambda n, fact: _has(fact, self.passages[n])  # noqa: E731
        missing = {fact for fact in facts if not any(holds(n, fact) for n in cited)}
        if not any(facts[fact] for fact in missing):
            return None  # only weak facts missing (a name the passage writes as "he"): left as cited
        keep = [n for n in cited if any(holds(n, fact) for fact in facts)]
        added: list[int] = []
        while missing and len(added) < self.max_added:
            best, best_found = None, set()
            for n in range(len(self.cites)):
                if n in keep or n in added:
                    continue
                found = {fact for fact in missing if holds(n, fact)}
                if len(found) > len(best_found):
                    best, best_found = n, found
            if best is None or not (any(facts[f] for f in best_found) or best_found == missing):
                break
            added.append(best)
            missing -= best_found
        if not added:
            return None
        return "".join(f"[来源：{self.cites[n][0]}｜位置：{self.cites[n][1]}]" for n in [*keep, *added])
