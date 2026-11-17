"""Round two's extra questions over the recall_wording_v1 library (written before any lever).

  python evaluation/survey_recall_v2/make_questions.py

holdout.jsonl   survey questions in other words than the 22 of round one ("tuning" set), with
                the same kind of gold (every guide of the group); levers are chosen on the
                tuning set and checked here
negative.jsonl  survey-shaped questions about programmes the library has no guide for: the
                right outcome is nothing kept, or an answer that says the files lack it
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recall_wording_v1"))

import corpus_spec as spec  # noqa: E402
from make_corpus import EXTENSION  # noqa: E402

HOLDOUT = [
    ("innovation", "time", "各类创新项目一般几月份截止？"),
    ("innovation", "who", "申报创新项目一般要满足什么条件？"),
    ("talent", "time", "人才政策一般什么时候可以申请？"),
    ("talent", "fund", "人才补贴通常最高能拿多少？"),
    ("training", "who", "技能培训补贴通常哪些人可以领？"),
    ("training", "time", "培训补贴一般什么时候申领？"),
    ("ip", "who", "专利商标资助一般对申请人有什么要求？"),
    ("ip", "fund", "知识产权方面各项资助标准是多少？"),
    ("startup", "time", "创业补贴一般什么时候申请？"),
    ("upgrade", "time", "技改项目通常什么时候申报？"),
    ("upgrade", "fund", "技改和绿色制造的奖补一般是多少？"),
    ("market", "who", "外贸和会展补贴一般什么企业能申请？"),
    ("market", "fund", "开拓市场方面的补贴一般有多少？"),
    ("rnd", "time", "研发奖励和高企培育一般什么时候申报？"),
    ("platform", "time", "实验室和众创空间一般什么时候申报？"),
    ("talent", "who", "各地的人才补贴都要求什么学历？"),
]

NEGATIVE = [
    ("fund", "宠物医院一般能拿多少补贴？"),
    ("fund", "各地的游艇码头建设补贴分别是多少？"),
    ("who", "围棋培训机构通常有哪些申报条件？"),
    ("time", "海外留学贷款一般什么时候申请？"),
    ("fund", "电竞战队一般能拿到多少扶持？"),
    ("who", "古籍修复项目通常有哪些门槛？"),
    ("time", "各地的马拉松赛事补贴都是几月申报？"),
    ("fund", "民宿装修一般能补多少钱？"),
]


def main() -> None:
    files = {g["id"]: g["title"] + EXTENSION[g["fmt"]] for g in spec.GUIDES}
    rows = []
    for number, (group, kind, question) in enumerate(HOLDOUT, start=1):
        gold = [{"file": files[g["id"]], "marker": g[kind][2], "amounts": g["amounts"] if kind == "fund" else []}
                for g in spec.GUIDES if g["cat"] == group]
        rows.append({"id": f"H{number:02d}", "type": "survey", "kind": kind, "group": group, "question": question, "gold": gold})
    negative = [{"id": f"N{number:02d}", "type": "negative", "kind": kind, "group": "", "question": question, "gold": []}
                for number, (kind, question) in enumerate(NEGATIVE, start=1)]
    for name, items in (("holdout.jsonl", rows), ("negative.jsonl", negative)):
        with (HERE / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in items:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(len(rows), "holdout,", len(negative), "negative")


if __name__ == "__main__":
    main()
