"""Pick the four home-page cards from the verified question pool.

Every candidate gets a score from four parts, then cards are drawn by weighted random
sampling (Efraimidis–Spirakis keys ``u ** (1 / score)``), so good questions come up
often but the page is never the same twice:

* freshness  — how recently the source file was added;
* similarity — overlap with the user's recent questions (only for "接着上次");
* novelty    — how rarely the question has been shown, and not just now;
* feedback   — clicks over impressions, smoothed.

At most one card comes from the question history, and only about half the time; the
others come from the library itself. No two cards share a source file (copies and other
formats of one document count as the same file), and cards from one folder are drawn
less often once a card from it is on the page.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import os
import random
import re
from typing import Any, Callable, Iterable

from .templates import FIND, clean_title

NEW_FILE_DAYS = 7
HISTORY_CARD_CHANCE = 0.5
MIN_HISTORY_SIMILARITY = 0.25
# A pool question this close to something already asked is not suggested again.
ASKED_SIMILARITY = 0.85
SAME_FOLDER_FACTOR = 0.35
LABEL_TITLE_CHARS = 14

KIND_NEW = "new"
KIND_CONTINUE = "continue"
KIND_INTEREST = "interest"
KIND_FIND = "find"

_STOPWORDS = {
    "什么", "哪些", "如何", "怎么", "怎样", "是", "的", "了", "吗", "呢", "一下", "找", "找一下", "文件",
    "中", "讲", "部分", "提到", "有", "主要", "这个", "那个", "我", "你", "请", "帮", "帮我", "里", "和",
    "与", "及", "在", "对", "关于", "工作", "是什么", "有哪些",
}


@dataclass
class Card:
    id: str
    question: str
    kind: str
    label: str
    file_id: str
    file_name: str
    qtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "kind": self.kind,
            "label": self.label,
            "doc_id": self.file_id,
            "file_name": self.file_name,
            "qtype": self.qtype,
        }


def default_tokenizer(text: str) -> list[str]:
    from retrieval.embedded_store import tokenize

    return tokenize(text)


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _label_title(name: str) -> str:
    title = clean_title(name)
    return title if len(title) <= LABEL_TITLE_CHARS else title[: LABEL_TITLE_CHARS - 1] + "…"


def label_for(kind: str, file_name: str) -> str:
    # Neutral wording: say where the question comes from, never what we "noticed".
    return {
        KIND_NEW: "新加入",
        KIND_CONTINUE: "接着上次的话题",
        KIND_FIND: "找文件",
    }.get(kind) or f"来自《{_label_title(file_name)}》"


class Sampler:
    def __init__(
        self,
        tokenizer: Callable[[str], list[str]] | None = None,
        rng: random.Random | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.tokenizer = tokenizer or default_tokenizer
        self.rng = rng or random.Random()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    # ---------------- scores ----------------

    def _tokens(self, text: str) -> set[str]:
        return {token for token in self.tokenizer(text) if token not in _STOPWORDS and len(token.strip()) > 0}

    @staticmethod
    def _overlap(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / math.sqrt(len(a) * len(b))

    def similarity(self, candidate_tokens: set[str], history: list[tuple[set[str], float]]) -> float:
        """Best recency-weighted overlap with a recent question (0..1)."""
        return max((self._overlap(candidate_tokens, tokens) * weight for tokens, weight in history), default=0.0)

    def _history(self, recent: Iterable[dict[str, Any]], now: datetime) -> list[tuple[set[str], float]]:
        history = []
        for item in recent:
            asked = _parse_time(item.get("asked_at")) or now
            age_days = max(0.0, (now - asked).total_seconds() / 86400)
            history.append((self._tokens(item.get("question", "")), 0.5 ** (age_days / 14)))
        return history

    @staticmethod
    def freshness(added: datetime | None, now: datetime) -> float:
        if added is None:
            return 0.0
        return math.exp(-max(0.0, (now - added).total_seconds() / 86400) / 10)

    @staticmethod
    def novelty(row: dict[str, Any], now: datetime) -> float:
        value = 1 / math.sqrt(1 + int(row.get("shown_count") or 0))
        shown = _parse_time(row.get("last_shown_at"))
        if shown is not None and now - shown < timedelta(hours=12):
            value *= 0.3
        return value

    @staticmethod
    def feedback(row: dict[str, Any]) -> float:
        return (int(row.get("click_count") or 0) + 1) / (int(row.get("shown_count") or 0) + 2)

    # ---------------- drawing ----------------

    def _draw(self, items: list[tuple[dict[str, Any], float]], taken_files: set[str], taken_folders: dict[str, int]):
        best, best_key = None, -1.0
        for row, score in items:
            if row["_file_key"] in taken_files:
                continue
            score *= SAME_FOLDER_FACTOR ** taken_folders.get(row["_folder"], 0)
            key = self.rng.random() ** (1 / max(score, 1e-6))
            if key > best_key:
                best, best_key = row, key
        return best

    def pick(
        self,
        candidates: list[dict[str, Any]],
        recent_questions: list[dict[str, Any]] | None = None,
        n: int = 4,
        exclude: Iterable[str] = (),
        cold_start: bool | None = None,
    ) -> list[Card]:
        """Choose ``n`` cards. Each candidate row carries ``added_at`` and ``path`` of its file."""
        now = self.clock()
        excluded = set(exclude)
        recent = list(recent_questions or [])
        history = self._history(recent, now)
        if cold_start is None:
            cold_start = not recent and not any(
                (row.get("click_count") or 0) or row.get("dismissed") for row in candidates
            )

        rows = []
        for row in candidates:
            if row["id"] in excluded:
                continue
            tokens = self._tokens(row["question"] + " " + clean_title(row.get("file_name", "")))
            similarity = self.similarity(tokens, history) if history else 0.0
            if history and self.similarity(self._tokens(row["question"]), [(t, 1.0) for t, _ in history]) >= ASKED_SIMILARITY:
                continue
            added = _parse_time(row.get("added_at"))
            rows.append({
                **row,
                "_file_key": row.get("title_key") or row["file_id"],
                "_folder": os.path.dirname(str(row.get("path", ""))).lower(),
                "_new": added is not None and now - added <= timedelta(days=NEW_FILE_DAYS),
                "_fresh": self.freshness(added, now),
                "_sim": similarity,
                "_novel": self.novelty(row, now),
                "_feedback": self.feedback(row),
            })

        def scored(kind: str) -> list[tuple[dict[str, Any], float]]:
            if kind == KIND_FIND:
                pool = [r for r in rows if r["qtype"] == FIND]
            else:
                pool = [r for r in rows if r["qtype"] != FIND]
            if kind == KIND_NEW:
                pool = [r for r in pool if r["_new"]]
                return [(r, 0.05 + 0.5 * r["_fresh"] + 0.3 * r["_novel"] + 0.15 * r["_feedback"]) for r in pool]
            if kind == KIND_CONTINUE:
                pool = [r for r in pool if r["_sim"] >= MIN_HISTORY_SIMILARITY]
                return [(r, 0.05 + 0.6 * r["_sim"] + 0.25 * r["_novel"] + 0.1 * r["_feedback"]) for r in pool]
            # The library itself: freshness, novelty and feedback; no history.
            return [(r, 0.05 + 0.25 * r["_fresh"] + 0.45 * r["_novel"] + 0.25 * r["_feedback"]) for r in pool]

        if cold_start:
            plan = [KIND_NEW, KIND_NEW, KIND_INTEREST, KIND_FIND]
        else:
            plan = []
            if history and self.rng.random() < HISTORY_CARD_CHANCE:
                plan.append(KIND_CONTINUE)
            plan += [KIND_NEW, KIND_FIND]
        plan += [KIND_INTEREST] * max(0, n - len(plan))
        plan = plan[:n]

        cards: list[Card] = []
        taken_files: set[str] = set()
        taken_folders: dict[str, int] = {}
        taken_ids: set[str] = set()
        for kind in plan:
            row = self._draw(scored(kind), taken_files, taken_folders)
            if row is None and kind != KIND_INTEREST:
                # Nothing new / nothing related / no find question: fall back to the library.
                kind = KIND_INTEREST
                row = self._draw(scored(kind), taken_files, taken_folders)
            if row is None:
                row = self._draw(scored(KIND_FIND), taken_files, taken_folders)
                kind = KIND_FIND
            if row is None or row["id"] in taken_ids:
                continue
            taken_ids.add(row["id"])
            taken_files.add(row["_file_key"])
            taken_folders[row["_folder"]] = taken_folders.get(row["_folder"], 0) + 1
            cards.append(Card(
                id=row["id"], question=row["question"], kind=kind,
                label=label_for(kind, row.get("file_name", "")), file_id=row["file_id"],
                file_name=row.get("file_name", ""), qtype=row["qtype"],
            ))
        return cards


def tokens_for_tests(text: str) -> list[str]:
    """A dependency-free tokenizer: words, plus CJK characters in pairs."""
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    cjk = re.sub(r"[^一-鿿]", " ", text)
    for run in cjk.split():
        tokens += [run[i : i + 2] for i in range(max(1, len(run) - 1))]
    return tokens
