"""Suggested questions written from a file's name and outline, on this machine, without a model.

A file contributes questions of three kinds:

* about the whole file, chosen by what kind of document its name says it is
  ("XX 检测报告.pdf" -> "《XX 检测报告》的检测结论是什么？");
* about one of its sections, from the parsed heading path
  ("1.1.5 ReAct 循环" -> "ReAct 循环是如何工作的？");
* one "find a file" question ("找一下钠基安全电池的检测报告").

Everything here is a candidate: a question only enters the pool once retrieval shows
its first hit is the file it was written from (see ``service.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from pathlib import Path
from typing import Any

ASK = "ask"
FIND = "find"

MAX_SECTION_CANDIDATES = 5


@dataclass(frozen=True)
class Candidate:
    question: str
    qtype: str  # ASK | FIND
    template: str  # which rule wrote it, kept for evaluation


# ---------------- titles ----------------

_BRACKET_PREFIX = re.compile(r"^(?:[【\[][^】\]]{0,20}[】\]]|附件\s*\d*(?=\s))\s*")
_LEADING_NUMBER = re.compile(r"^[\s　]*\d{1,2}(?:[.、．\-_]|\s)+(?=\D)")
_COPY_SUFFIX = re.compile(r"\s*[(（]\d{1,2}[)）]\s*$")
_EDITION_SUFFIX = re.compile(r"\s*[(（]?(?:盖章版|润色版|最终版|定稿|初稿|修订版|中文版|英文版)[)）]?\s*$")
_VERSION_SUFFIX = re.compile(r"\s*[Vv]\d+(?:\.\d+)*\s*$")
_DATE_SUFFIX = re.compile(
    r"(?<=[一-鿿）)A-Za-z\s])[\s\-_]*(?:20\d{2}(?:[.\-]?\d{1,2}){0,2}|20|\d{1,2}-\d{1,2})\s*$"
)
_UNCLOSED = re.compile(r"\s*[（(][^）)]*$")
_EMPTY_PARENS = re.compile(r"[（(]\s*[）)]")
_SEPARATORS = re.compile(r"\s*(?:--+|——|_)\s*")
_SPACES = re.compile(r"[\s　]+")


def clean_title(name: str) -> str:
    """A file name as a reader would say it: no extension, list numbers, copies or versions."""
    title = Path(name).stem if Path(name).suffix else name
    title = _BRACKET_PREFIX.sub("", title.strip())
    title = _LEADING_NUMBER.sub("", title)
    for _ in range(3):
        before = title
        for pattern in (_COPY_SUFFIX, _EDITION_SUFFIX, _VERSION_SUFFIX, _DATE_SUFFIX, _UNCLOSED):
            title = pattern.sub("", title)
        if title == before:
            break
    title = _SEPARATORS.sub("：", title).strip("：-— ")
    title = _SPACES.sub(" ", title).strip()
    return title or Path(name).stem


def title_key(name: str) -> str:
    """Same key for a file and its copies or other formats (``X.docx``, ``X.pdf``, ``X (1).docx``)."""
    return re.sub(r"[\s：:·\-_]+", "", clean_title(name)).lower()


# ---------------- what kind of document ----------------

# (kind, pattern on the title, noun used in "find a file", whole-file questions)
_KINDS: tuple[tuple[str, re.Pattern[str], str, tuple[str, ...]], ...] = (
    ("report", re.compile(r"检测报告|测试报告|检验报告|表征报告|检测|COA|MSDS|SDS", re.I), "检测报告",
     ("《{t}》的检测结论是什么？", "《{t}》检测了哪些项目？")),
    ("plan", re.compile(r"商业计划书|计划书|商业计划"), "商业计划书",
     ("《{t}》的核心产品和商业模式是什么？", "《{t}》对市场空间是怎么分析的？")),
    ("minutes", re.compile(r"纪要|会议记录"), "会议纪要",
     ("《{t}》安排了哪些工作重点？", "《{t}》总结了哪些上周工作？")),
    ("news", re.compile(r"新闻稿|公众号|动态"), "新闻稿",
     ("《{t}》里双方围绕哪些议题进行了交流？",)),
    ("guide", re.compile(r"指南|申报"), "申报材料",
     ("《{t}》的申报条件有哪些？", "《{t}》的主要内容是什么？")),
    ("notice", re.compile(r"通知|公告|邮件"), "通知",
     ("《{t}》有哪些要求和时间安排？",)),
    ("policy", re.compile(r"办法|措施|方案|政策|规定|制度"), "政策文件",
     ("《{t}》有哪些扶持措施？", "《{t}》的主要内容是什么？")),
    ("spec", re.compile(r"规格书|产品手册|手册|说明书|白皮书"), "产品资料",
     ("《{t}》列出了哪些关键参数？", "《{t}》主要讲了什么？")),
    ("intro", re.compile(r"简介|介绍|概况"), "介绍材料",
     ("《{t}》介绍了哪些核心业务？",)),
    ("agreement", re.compile(r"协议|框架"), "协议",
     ("《{t}》约定了哪些合作内容？",)),
    ("letter", re.compile(r"拜访函|邀请函|函$"), "函件",
     ("《{t}》的主要事由是什么？",)),
)
_TABLE_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def document_kind(title: str, ext: str = "") -> tuple[str, str, tuple[str, ...]]:
    for kind, pattern, noun, questions in _KINDS:
        if pattern.search(title):
            return kind, noun, questions
    if ext.lower() in _TABLE_EXTENSIONS:
        return "table", "表格", ("《{t}》汇总了哪些内容？",)
    return "general", "文件", ("《{t}》主要讲了什么？",)


# Questions written from what the opening says it is about (product sheets, proposals).
_TEXT_CUES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"痛点"), "{t}主要解决了哪些痛点？"),
    (re.compile(r"核心原理"), "{t}的核心原理是什么？"),
    (re.compile(r"关键技术突破|技术突破"), "{t}有哪些关键技术突破？"),
    (re.compile(r"应用场景|应用领域"), "{t}可以用在哪些场景？"),
)


# ---------------- headings ----------------

_HEADING_NUMBER = re.compile(
    r"^(?:第\s*[一二三四五六七八九十百零\d]+\s*[章节部分篇讲]\s*|"
    r"[（(]?[一二三四五六七八九十]+[）)、.．]\s*|"
    r"\d+(?:\.\d+)*[、.．]?\s*|"
    r"[A-Za-z][.、)]\s*|n\s+)"
)
_GENERIC_HEADING = re.compile(
    r"^(?:表格\s*\d*|第?\s*\d+\s*页.*|页\s*\d+.*|工作表.*|图\s*\d.*|附件.*|附录.*|目录|前言|引言|序言?|摘要|概要|"
    r"总结|小结|结语|致谢|参考文献|声明|重要声明|说明|备注|注意事项|封面.*|扉页|版权.*|"
    r"本章小结|习题|思考题|练习|案例|样品照片|.*报告编号.*|.*TEST REPORT.*|.*Test.*)$",
    re.I,
)
_FORMULA_CHARS = re.compile(r"[=+<>|{}\[\]#@$%^*~]")
_CJK = re.compile(r"[一-鿿]")
_LATIN_TERM = re.compile(r"[A-Za-z]{2,}")
_PARENTHESES = re.compile(r"[（(][^）)]{0,30}[）)]")
_CONCEPT_END = re.compile(
    r"(?:循环|机制|模型|算法|架构|原理|技术|方法|框架|工程|协议|模式|系统|网络|编码|学习|嵌入|注意力|微调|"
    r"范式|流程|策略|结构|工艺|材料|设备|芯片|电池|电解质|催化剂)$"
)
_HOW_END = re.compile(r"(?:循环|机制|算法|架构|原理|流程|工艺|结构)$")
_LIST_END = re.compile(
    r"(?:优势|痛点|前景|场景|要求|措施|条件|规划|安排|目标|价值|特点|特性|指标|参数|风险|壁垒|格局|需求|"
    r"功能|分工|任务|难点|挑战|方向|内容|步骤|环节|政策|建议)$"
)
# Headings that start like a sentence ("将经验写入参数", "为分类任务微调…") read badly as a noun.
_VERB_START = re.compile(r"^(?:将|把|从|让|用|对|为|如何|为什么|怎样|走向|构建|推动|探索|助力|了解|共)")
_SPACED_CJK = re.compile(r"(?<=[一-鿿]) (?=[一-鿿])")
_LABEL_PREFIX = re.compile(r"^[^·]{1,6}\s*·\s*")
_ACRONYM = re.compile(r"[A-Z]{2,}")


def clean_heading(heading: str) -> str:
    """The last part of a heading path, without list numbers; "" when it says nothing."""
    parts = [part.strip() for part in str(heading).split(" > ") if part.strip()]
    if not parts:
        return ""
    text = _SPACED_CJK.sub("", _SPACES.sub(" ", parts[-1]))
    text = _LABEL_PREFIX.sub("", _HEADING_NUMBER.sub("", text).strip())
    text = re.split(r"[，,]", text, maxsplit=1)[0].strip()
    if "：" in text or ":" in text:
        left, _, right = re.split(r"([：:])", text, maxsplit=1)
        right = right.strip()
        text = right if 2 <= len(right) <= 14 and "、" not in right else left.strip()
    text = _PARENTHESES.sub("", text).strip(" 　-—:：,，。.")
    if not (2 <= len(text) <= 20) or _GENERIC_HEADING.match(text) or _FORMULA_CHARS.search(text):
        return ""
    if not _CJK.search(text) and not _ACRONYM.search(text):
        return ""
    if sum(char.isdigit() for char in text) > len(text) / 3 or (len(text) > 14 and re.search(r"\d", text)):
        return ""
    return text


def _heading_question(heading: str, title: str, kind: str = "general") -> tuple[str, str]:
    # Concept-like headings stand on their own ("ReAct 循环"); others need the file's name.
    # News headings are slogans, and sentence-like ones are no noun: ask what the part says.
    if kind == "news" or _VERB_START.search(heading):
        return f"《{title}》中“{heading}”部分讲了什么？", "section_about"
    if _HOW_END.search(heading) and (_LATIN_TERM.search(heading) or len(heading) >= 4):
        return f"{heading}是如何工作的？", "section_how"
    if _LIST_END.search(heading):
        return f"《{title}》提到的{heading}有哪些？", "section_list"
    if _CONCEPT_END.search(heading) or _LATIN_TERM.search(heading):
        return f"什么是{heading}？", "section_what"
    return f"《{title}》中“{heading}”部分讲了什么？", "section_about"


def _section_headings(units: list[dict[str, Any]], title: str) -> list[str]:
    seen: list[str] = []
    title_compact = re.sub(r"\s+", "", title)
    for unit in units:
        heading = clean_heading(unit.get("heading", ""))
        compact = re.sub(r"\s+", "", heading)
        if not heading or heading in seen or compact in title_compact or title_compact in compact:
            continue
        seen.append(heading)
    return seen


def _spread(items: list[str], count: int, rng: random.Random) -> list[str]:
    """Up to ``count`` items spread over the whole list (not all from chapter 1)."""
    if count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[min(len(items) - 1, int(step * index + rng.random() * step))] for index in range(count)]


# ---------------- find a file ----------------

_DATE_IN_TITLE = re.compile(r"20\d{2}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?")
_CODE_TAIL = re.compile(r"[\s\-_]*[A-Z]{1,4}-?\d{4,}[A-Z]?$")


def find_question(title: str, kind: str, noun: str) -> str:
    date = _DATE_IN_TITLE.search(title)
    if date:
        rest = title.replace(date.group(0), "").strip("：:-_ ")
        if rest:
            return f"找一下{date.group(0)}的{rest}"
    topic = _CODE_TAIL.sub("", title)
    if kind not in ("general", "table"):
        for word in (noun, "报告", "测试报告", "检测报告", "检验报告"):
            topic = topic.replace(word, "")
    topic = _EMPTY_PARENS.sub("", topic).strip("：:-_ 的")
    if 2 <= len(topic) <= 24 and topic != title and topic not in ("附件", "模板", "资料", "文件"):
        return f"找一下{topic}的{noun}"
    return f"找一下《{title}》这个文件"


# ---------------- all candidates for one file ----------------


def candidates(name: str, parsed: dict[str, Any], seed: str = "") -> list[Candidate]:
    """Question candidates for one file, best first, at most one "find a file" question."""
    rng = random.Random(seed or name)
    title = clean_title(name)
    ext = Path(name).suffix
    kind, noun, file_questions = document_kind(title, ext)
    units = list(parsed.get("units") or [])
    out: list[Candidate] = []

    headings = _section_headings(units, title)
    # Concept-like sections first: they make the most natural questions.
    ranked = sorted(headings, key=lambda h: 0 if (_CONCEPT_END.search(h) or _LIST_END.search(h)) else 1)
    concept = [h for h in ranked if _CONCEPT_END.search(h) or _LIST_END.search(h)]
    other = [h for h in ranked if h not in concept]
    picked = _spread(concept, min(MAX_SECTION_CANDIDATES, len(concept)), rng)
    picked += _spread(other, max(0, min(2, MAX_SECTION_CANDIDATES - len(picked))), rng)

    # A one-section document (product sheets): its subject reads best in plain words.
    subject = title
    if len(units) <= 2 and headings:
        subject = headings[0]
    opening = " ".join(str(unit.get("text", "")) for unit in units[:2])[:1200]
    # Product sheets open with "针对……的痛点"; elsewhere these words prove nothing.
    for cue, template in _TEXT_CUES if len(units) <= 3 and "针对" in opening[:200] else ():
        if cue.search(opening):
            out.append(Candidate(template.format(t=subject), ASK, "text_cue"))
            if len(out) >= 2:
                break

    out += [Candidate(template.format(t=title), ASK, f"file_{kind}") for template in file_questions]
    for heading in picked:
        question, template = _heading_question(heading, title, kind)
        out.append(Candidate(question, ASK, template))
    out.append(Candidate(find_question(title, kind, noun), FIND, f"find_{kind}"))

    unique: dict[str, Candidate] = {}
    for candidate in out:
        unique.setdefault(candidate.question, candidate)
    return list(unique.values())
