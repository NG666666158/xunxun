"""A survey answer goes file by file, one cited line each (answering.PROFILES["survey"]): one
bad line should cost that line, not the answer (evaluation/survey_recall_v2/REPORT.md).

* ``repair_names``: a citation whose file name is in no evidence block but close to exactly one
  ("…方向创新产业任务…" for "…方向创新任务…") cites that block's file. A name the model shortened
  already resolves (answering.evidence_for); a mistyped one fails the whole answer.
* ``drop_lines``: the lines of the statements the fact check could not confirm are taken out;
  the answer keeps the others, numbered again.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Iterable, Sequence

from .answering import CITATION, cited_claims

# How close a cited name must be to a block's file name (difflib ratio on the names without
# spaces, case and extension), and by how much it must beat the next closest one.
NAME_RATIO = 0.8
NAME_MARGIN = 0.05
_ITEM = re.compile(r"^(\s*)(\d+)([.、．)）]\s*)")


def _key(name: str) -> str:
    name = re.sub(r"\s+", "", name or "").lower()
    return re.sub(r"\.(?:docx?|pdf|txt|md|xlsx?|pptx?|csv|html?)$", "", name)


def closest_source(name: str, sources: Iterable[str]) -> str | None:
    """The one source ``name`` most likely means, or None when none is close or two are."""
    key = _key(name)
    if not key:
        return None
    scored = sorted(((SequenceMatcher(None, key, _key(source)).ratio(), source) for source in set(sources)), reverse=True)
    if not scored or scored[0][0] < NAME_RATIO:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < NAME_MARGIN:
        return None
    return scored[0][1]


def repair_names(answer: str, blocks: Sequence[dict[str, Any]]) -> tuple[str, int]:
    """``answer`` with each citation of a file name no block has re-pointed to the block file it
    is closest to (see ``closest_source``); and how many were changed."""
    sources = [str(block.get("source", "")) for block in blocks]
    known = {_key(source) for source in sources}
    changed = 0

    def fix(match: re.Match) -> str:
        nonlocal changed
        name = match.group(1).strip()
        head = re.split(r"\s*[｜|，,;；]\s*", name, maxsplit=1)[0]
        key = _key(head)
        if not key or key in known or any(key in other for other in known):
            return match.group(0)  # a block's name, or a shortened one (evidence_for resolves it)
        source = closest_source(head, sources)
        if source is None:
            return match.group(0)
        changed += 1
        return match.group(0).replace(head, source, 1)

    return CITATION.sub(fix, answer or ""), changed


def _similar(claim: str, flagged: str) -> bool:
    a, b = re.sub(r"\s+", "", claim), re.sub(r"\s+", "", flagged)
    if not a or not b:
        return False
    return a in b or b in a or SequenceMatcher(None, a, b).ratio() >= 0.6


def drop_lines(answer: str, flagged: Sequence[str]) -> tuple[str, int, int]:
    """``answer`` without the lines whose cited statements match ``flagged`` (the fact check's
    unsupported statements); the numbered items left are numbered again. Returns the answer,
    the statements dropped and the cited statements kept."""
    claims = [claim for claim, _ in cited_claims(answer)]
    bad = [claim for claim in claims if any(_similar(claim, text) for text in flagged)]
    if not bad:
        return answer, 0, len(claims)
    kept_lines: list[str] = []
    for line in (answer or "").split("\n"):
        if CITATION.search(line) and any(re.sub(r"\s+", "", claim)[:40] in re.sub(r"\s+", "", line) for claim in bad):
            continue
        kept_lines.append(line)
    number = 0
    out: list[str] = []
    for line in kept_lines:
        match = _ITEM.match(line)
        if match:
            number += 1
            line = f"{match.group(1)}{number}{match.group(3)}{line[match.end():]}"
        out.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return text, len(bad), len([claim for claim, _ in cited_claims(text)])


# ---------------------------------------------------------------- files the answer left out
def _cited_keys(answer: str) -> set[str]:
    return {_key(re.split(r"\s*[｜|，,;；]\s*", name.strip(), maxsplit=1)[0]) for name, _ in CITATION.findall(answer or "")}


def _is_cited(source: str, cited: set[str]) -> bool:
    key = _key(source)
    return any(key == name or (name and (name in key or key in name)) for name in cited)


_COLLECTION = re.compile(r"集合|汇编|汇总|清单|合集|问答|手册|一览")
# Topic words too general to say what a file is about.
_GENERAL_TOPIC = frozenset("企业 单位 个人 平台 方面 这类 一类 政府 地方 国家".split())


def about_topic(source: str, topics: Sequence[str]) -> bool:
    """Does the file's name carry one of the question's topic words (survey.survey_topics), and
    is it not a collection of other programmes?"""
    stem = str(source or "").rsplit(".", 1)[0]
    if _COLLECTION.search(stem):
        return False
    words = {word for topic in topics for word in topic.split() if len(word) >= 2 and word not in _GENERAL_TOPIC}
    return any(word in stem for word in words)


_HEADING_LEAD = re.compile(r"^(?:[一二三四五六七八九十]+、[^\s，。；]{1,10}\s+|[（(][一二三四五六七八九十\d]+[）)]\s*)")


def _clauses(lines: Sequence[str]) -> str:
    """The excerpt's sentences joined by ；, without a section heading in front and without
    one said twice (an excerpt joins passages, which repeat a clause)."""
    out: list[str] = []
    for line in lines:
        for clause in re.split(r"[。；;]", line):
            clause = clause.strip()
            while _HEADING_LEAD.match(clause):  # "三、基本条件 （一）…"
                clause = _HEADING_LEAD.sub("", clause, count=1).strip()
            if clause and clause not in out:
                out.append(clause)
    return "；".join(out)


def missing_lines(answer: str, blocks: Sequence[dict[str, Any]], ask: str,
                  answers: Any, rule_only: Any = None) -> tuple[str, list[str]]:
    """``answer`` with one line added for each evidence file it does not cite whose passage
    answers the ask (``answers(text, ask)``, survey.answers_in), copied from that passage with
    its citation, numbered after the answer's last item; ``rule_only(source)``: only files whose
    name says they state rules. Returns the answer and the files added."""
    if not ask:
        return answer, []
    cited = _cited_keys(answer)
    added: list[tuple[str, str]] = []
    seen: set[str] = set()
    for block in blocks:
        source = str(block.get("source", ""))
        if not source or source in seen or _is_cited(source, cited):
            continue
        text = str(block.get("text", ""))
        if not answers(text, ask) or (rule_only is not None and not rule_only(source)):
            continue
        seen.add(source)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and lines[0].startswith("【"):
            lines = lines[1:]
        body = _clauses(lines)
        if not body:
            continue
        name = source.rsplit(".", 1)[0]
        locator = str(block.get("locator") or "未标注位置")
        added.append((source, f"**{name}**：{body}。[来源：{source}｜位置：{locator}]"))
    if not added:
        return answer, []
    rows = (answer or "").split("\n")
    last = max((i for i, line in enumerate(rows) if _ITEM.match(line)), default=None)
    if last is None:
        return (answer.rstrip() + "\n\n" + "\n\n".join(line for _, line in added)).strip(), [s for s, _ in added]
    number = int(_ITEM.match(rows[last]).group(2))
    lead, sep = _ITEM.match(rows[last]).group(1), _ITEM.match(rows[last]).group(3)
    new = [f"\n{lead}{number + i}{sep}{line}" for i, (_, line) in enumerate(added, start=1)]
    end = last  # the last item may go on over several lines
    while end + 1 < len(rows) and rows[end + 1].strip() and not _ITEM.match(rows[end + 1]):
        end += 1
    rows = rows[: end + 1] + new + rows[end + 1:]
    return "\n".join(rows), [s for s, _ in added]
