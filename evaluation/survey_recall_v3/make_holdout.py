"""Round three's unseen survey questions, written after the levers were chosen (item 4): other
wordings than the tuning, held-out and detector sets, with and without 一般 / 通常, over the same
library and with the same kind of gold (every guide of the group).

  python evaluation/survey_recall_v3/make_holdout.py   # writes holdout2.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recall_wording_v1"))

import corpus_spec as spec  # noqa: E402
from make_corpus import EXTENSION  # noqa: E402

QUESTIONS = [
    ("innovation", "fund", "科技创新类项目最多能资助多少？"),
    ("innovation", "time", "搞创新的项目申报截止日期一般在几月？"),
    ("talent", "who", "引进人才的补贴通常对年龄或学历有什么要求？"),
    ("talent", "fund", "人才来了能领到哪些钱？"),
    ("training", "fund", "职业培训方面的补助标准是多少？"),
    ("training", "who", "参加技能提升培训的补贴谁可以申请？"),
    ("ip", "fund", "申请专利能拿到多少奖励？"),
    ("ip", "time", "商标和专利的资助一般几月份受理？"),
    ("startup", "who", "自己开公司创业，申请扶持需要什么条件？"),
    ("startup", "fund", "开办企业能拿到哪些创业补贴？"),
    ("upgrade", "fund", "工厂设备更新改造有多少补贴？"),
    ("market", "time", "外贸企业开拓市场的资金一般什么时候申请？"),
    ("rnd", "who", "什么样的企业能申请研发奖励？"),
    ("platform", "fund", "建孵化器、实验室这类载体能拿多少钱？"),
]


def main() -> None:
    files = {g["id"]: g["title"] + EXTENSION[g["fmt"]] for g in spec.GUIDES}
    with (HERE / "holdout2.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for number, (group, kind, question) in enumerate(QUESTIONS, start=1):
            gold = [{"file": files[g["id"]], "marker": g[kind][2], "amounts": g["amounts"] if kind == "fund" else []}
                    for g in spec.GUIDES if g["cat"] == group]
            row = {"id": f"U{number:02d}", "type": "survey", "kind": kind, "group": group, "question": question, "gold": gold}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(len(QUESTIONS), "-> holdout2.jsonl")


if __name__ == "__main__":
    main()
