"""Item 1: would a card's 领域 (what a policy file supports, written by Qwen3-8B at import) find
the guides that never use the question's topic word?

  python evaluation/survey_recall_v3/probe_cards.py [--cards work/cards_fields.jsonl] [--field fields|summary]

For each survey question (tuning and held-out) and each of its topics: the files whose card field
matches the topic (``card_matches`` below), how many of them are gold, and whether the gold files
the files-first search misses ("cut" / "search" in losses) are among them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3  # noqa: E402

import re  # noqa: E402

from retrieval.survey import RULE_TYPES, survey_topics  # noqa: E402

# ---------------------------------------------------------------- what a file supports (tried, not kept)
# Two-character pieces every programme paper shares: a match on them says nothing.
_GENERIC_PIECES = frozenset(
    "企业 政策 补贴 发展 管理 项目 财政 资金 支持 服务 建设 产业 工作 实施 申报 申请 资助 扶持 奖励 "
    "政府 公共 社会 经济 行业 领域 相关 方面 推广 应用 培育 专项 计划 方案 办法 细则 指南 通知 执行 解读 规范".split()
)
_FIELD_SPLIT = re.compile(r"[、，,；;/\s]+")


def _pieces(word: str) -> set[str]:
    return {word[i : i + 2] for i in range(len(word) - 1)} - _GENERIC_PIECES


def card_matches(topics: list[str], fields: str, doc_type: str = "") -> bool:
    """Is a file whose card says it supports ``fields`` ("科技创新、技术攻关") about one of the
    question's ``topics`` ("创新")? A topic word inside a field or a field inside a topic word,
    or a shared two-character piece that is not common to every programme paper ("改造" in
    "智能化改造" and "技术改造"). Only cards of rule-stating types count."""
    if not fields or doc_type not in RULE_TYPES:
        return False
    words = [w for topic in topics for w in topic.split() if len(w) >= 2 and w not in _GENERIC_PIECES]
    if not words:
        return False
    for field in _FIELD_SPLIT.split(fields):
        if len(field) < 2:
            continue
        for word in words:
            if word in field or field in word or _pieces(word) & _pieces(field):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default=str(round3.WORK / "cards_fields.jsonl"))
    parser.add_argument("--field", default="fields")
    args = parser.parse_args()
    cards = {row["name"]: row["card"] for row in round3.read_rows(Path(args.cards))}
    sets = [("tuning", round3.surveys()), ("holdout", round3.read_rows(round3.V2 / "holdout.jsonl"))]
    report = []
    for set_name, questions in sets:
        totals = {"matched": 0, "gold_matched": 0, "gold": 0, "questions": len(questions)}
        for q in questions:
            gold = {g["file"] for g in q["gold"]}
            topics = survey_topics(q["question"])
            matched = {name for name, card in cards.items()
                       if card_matches(topics, str(card.get(args.field, "")), str(card.get("doc_type", "")))}
            totals["matched"] += len(matched)
            totals["gold_matched"] += len(matched & gold)
            totals["gold"] += len(gold)
            report.append({"set": set_name, "id": q["id"], "topics": topics, "matched": len(matched),
                           "gold_matched": sorted(matched & gold), "gold_unmatched": sorted(gold - matched),
                           "other": sorted(matched - gold)[:8]})
        print(set_name, totals)
    round3.write_rows(round3.WORK / f"probe_cards_{args.field}.jsonl", report)
    for row in report:
        print(row["set"], row["id"], row["topics"], "matched", row["matched"], "gold", len(row["gold_matched"]),
              "unmatched gold:", [n[:14] for n in row["gold_unmatched"]], "other:", [n[:12] for n in row["other"][:4]])


if __name__ == "__main__":
    main()
