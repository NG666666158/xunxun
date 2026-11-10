"""Files a question names, by their names alone (no model call).

"答辩PPT用了哪几步制备工艺" names "04 答辩PPT.pptx": its answer should come from that file,
not from a similar project's business plan that describes a process at greater length
(evaluation/multidoc_v1/REPORT.md, method 5). A name counts when it is written in 《》 or
when a file's name, without its extension and leading number ("04 ", "1-", "0."), appears
in the question (at least four characters).
"""

from __future__ import annotations

import re
from typing import Iterable

_TITLE = re.compile(r"《([^》]{1,160})》")
_EXTENSION = re.compile(r"\.(?:pdf|docx?|xlsx?|pptx?|md|txt|csv|png|jpe?g)$", re.I)
_NUMBERING = re.compile(r"^\s*\d+\s*[.、．\-_\s]\s*")
_COPY_MARKS = re.compile(r"\s*[(（]\s*\d{1,2}\s*[)）]\s*$")
MIN_NAME_CHARS = 4


def loose(text: str) -> str:
    return re.sub(r"[\s_\-·—–]+", "", text or "").lower()


def bare_name(source: str) -> str:
    """"04 答辩PPT.pptx" -> "答辩PPT"; "激光雷达技术解析 (1).docx" -> "激光雷达技术解析"."""
    stem = _EXTENSION.sub("", (source or "").strip())
    return _COPY_MARKS.sub("", _NUMBERING.sub("", stem)).strip()


def named_documents(question: str, documents: Iterable[tuple[str, str]]) -> list[str]:
    """Doc ids of the files ``question`` names; ``documents`` is (doc_id, source) pairs.

    Titles in 《》 match a file whose bare name contains them or is contained in them.
    Without 《》, the longest bare name written out in the question wins (every file that
    shares it: same-named files in different folders all count).
    """
    pairs = [(doc_id, source, loose(bare_name(source))) for doc_id, source in documents]
    titles = [loose(t) for t in _TITLE.findall(question or "") if len(loose(t)) >= 2]
    if titles:
        found = []
        for doc_id, _, name in pairs:
            if name and any(t == name or (len(t) >= MIN_NAME_CHARS and (t in name or name in t)) for t in titles):
                found.append(doc_id)
        return list(dict.fromkeys(found))
    text = loose(_TITLE.sub(" ", question or ""))
    best: list[str] = []
    best_len = 0
    for doc_id, _, name in pairs:
        if len(name) < MIN_NAME_CHARS or name not in text:
            continue
        if len(name) > best_len:
            best, best_len = [doc_id], len(name)
        elif len(name) == best_len:
            best.append(doc_id)
    return list(dict.fromkeys(best))
