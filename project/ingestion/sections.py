"""Section headings written as ordinary paragraphs: "三、支持强度与方式", "第二章 申报条件", a bold line.

Official papers written in Word rarely use heading styles: a heading is a bold paragraph, or
just a numbered one. Without styles the whole file was one unit, cut every 700 characters
wherever that fell, so the passage that states the amounts began with the end of two other
sections and matched the question badly (evaluation/recall_wording_v1/REPORT.md).

``sectioned`` turns a file's paragraphs into units that start at those headings and carry them
as their heading. The heading line stays in the text, so a line wrongly taken for a heading
only adds a boundary; nothing is lost.

* level 1: "一、…", "第三章 …" (a short line without a sentence end), also as a lead-in
  ("三、支持强度与方式。项目分三档…"), and a short bold line before any numbered heading;
* level 2: "（一）…" as a short line of its own, or a short bold line under a numbered heading.
  A section is only cut at these when it is long (``split_above`` characters); a short section
  stays whole, so its parts are read together.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .models import DocumentUnit

_CN = "一二三四五六七八九十百"
_SENTENCE_END = re.compile(r"[。；;！？!?]")
_LEVEL1 = re.compile(rf"^(?:第\s*[{_CN}\d]+\s*[章节篇]|第\s*[{_CN}\d]+\s*部分|[{_CN}]+\s*、)\s*\S.{{0,29}}$")
# "三、支持强度与方式。项目分三档给予支持……": the heading opens its own paragraph.
_LEVEL1_LEAD = re.compile(rf"^([{_CN}]+\s*、\s*[^。；;：:，,！？!?\s]{{2,24}})[。：:]\s*\S")
_LEVEL2 = re.compile(rf"^[（(]\s*[{_CN}]+\s*[)）]\s*[^。；;：:，,！？!?]{{1,24}}$")
BOLD_MAX = 40
# A numbered "heading" every line or two is a list ("一、张三 二、李四"), not a document's sections.
MIN_AVERAGE_SECTION = 60


@dataclass(frozen=True)
class Paragraph:
    text: str
    bold: bool = False  # every run of it is bold
    styled: int = 0  # the level of a Word heading style (0: an ordinary paragraph)


def heading_level(text: str, bold: bool = False) -> tuple[int, str] | None:
    """``(level, heading)`` when the paragraph is, or opens with, a section heading."""
    text = text.strip()
    if not text:
        return None
    if len(text) <= 36 and not _SENTENCE_END.search(text) and not text.endswith(("：", ":", "，", ",")):
        if _LEVEL1.match(text):
            return 1, text
        if _LEVEL2.match(text):
            return 2, text
        if bold and len(text) <= BOLD_MAX:
            return 3, text  # placed by ``sectioned``: a title before the numbered headings, a sub-heading under them
    lead = _LEVEL1_LEAD.match(text)
    if lead:
        return 1, lead.group(1)
    return None


def sectioned(paragraphs: list[Paragraph], split_above: int = 1200) -> list[DocumentUnit] | None:
    """Units that start at the headings found among ``paragraphs``; None when the file has no
    such headings (the caller keeps its usual units).

    Word heading styles keep working as before: a styled heading is not part of the text and
    resets the headings below its level.
    """
    found = [(heading_level(p.text, p.bold) if not p.styled else None) for p in paragraphs]
    numbered = [f for f in found if f and f[0] == 1]
    plain = sum(len(p.text) for p in paragraphs if not p.styled)
    if len(numbered) < 2 or plain / len(numbered) < MIN_AVERAGE_SECTION:
        # Bold lines alone are sections too, when there are several of them.
        bold = [f for f in found if f and f[0] == 3]
        if numbered or len(bold) < 2 or plain / len(bold) < MIN_AVERAGE_SECTION:
            return None
        found = [(1, f[1]) if f and f[0] == 3 else None for f in found]
    units: list[DocumentUnit] = []
    stack: list[str] = []  # styled headings, then the level-1 heading found here
    styled_depth = 0
    section: list[tuple[str, str]] = []  # (sub-heading or "", paragraph)
    seen_numbered = False

    def flush() -> None:
        if not section:
            return
        heading = " > ".join(stack)
        length = sum(len(text) for _, text in section)
        subs = [sub for sub, _ in section if sub]
        if length > split_above and len(subs) >= 2:
            current: list[str] = []
            label = ""
            for sub, text in section:
                if sub and current:
                    units.append(DocumentUnit(text="\n".join(current), heading=" > ".join(filter(None, [heading, label]))))
                    current = []
                if sub:
                    label = sub
                current.append(text)
            if current:
                units.append(DocumentUnit(text="\n".join(current), heading=" > ".join(filter(None, [heading, label]))))
        else:
            units.append(DocumentUnit(text="\n".join(text for _, text in section), heading=heading))
        section.clear()

    for paragraph, heading in zip(paragraphs, found, strict=True):
        if paragraph.styled:
            flush()
            del stack[paragraph.styled - 1:]
            stack.append(paragraph.text)
            styled_depth = len(stack)
            seen_numbered = False
            continue
        if heading and heading[0] == 3:
            heading = (2, heading[1]) if seen_numbered else None  # a bold title before the sections is text
        if heading and heading[0] == 1:
            flush()
            del stack[styled_depth:]
            stack.append(heading[1])
            seen_numbered = True
            section.append(("", paragraph.text))
        elif heading:
            section.append((heading[1], paragraph.text))
        else:
            section.append(("", paragraph.text))
    flush()
    return [unit for unit in units if unit.text.strip()]
