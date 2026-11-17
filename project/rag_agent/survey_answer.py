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
