"""Doc cards for the synthetic library, written by Qwen3-8B the way the app writes them after an
import (retrieval/doc_cards.py: file name + the first doc_card_chars characters), paced.

  python evaluation/survey_recall_v3/make_cards.py --variant shipped   # the app's card prompt
  python evaluation/survey_recall_v3/make_cards.py --variant fields    # + 领域 (what a policy file supports)

Writes work/cards_<variant>.jsonl (one card per file, with the reply's token count); resumes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3  # noqa: E402
from round3 import harness  # noqa: E402

from retrieval import doc_cards  # noqa: E402

# The variant tried (not kept): the card also says what a policy paper supports (领域), in the
# model's general words, for a survey question to find a guide that never uses its word.
FIELDS_RULE = (
    "- 领域：通知、办法、细则、指南、方案这类政策文件支持或管理的事项属于哪些领域，写2到4个概括的词，"
    "用顿号分开（比如一份乡村道路养护资金办法写“交通、乡村振兴、基础设施”）；其他文件留空。\n"
)
CARD_PROMPT_FIELDS = (
    "你是文件归档助手。根据文件名和文件开头的内容判断这是一份什么文件，只输出一行 JSON，不要解释：\n"
    '{{"类型": "", "对象": "", "时间": "", "摘要": "", "领域": ""}}\n'
    + (doc_cards._CARD_RULES + FIELDS_RULE).replace("{", "{{").replace("}", "}}") +
    "\n文件名：{name}\n开头内容：{text}"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("shipped", "fields"), required=True)
    parser.add_argument("--pause", type=float, default=8.0)
    args = parser.parse_args()
    from langchain_core.messages import HumanMessage
    from services.model_clients import build_chat_model

    config = harness.settings()
    model = build_chat_model(config)
    prompt = doc_cards.CARD_PROMPT if args.variant == "shipped" else CARD_PROMPT_FIELDS
    folder = harness.index_dir("base")
    out = round3.WORK / f"cards_{args.variant}.jsonl"
    done = {row["name"] for row in round3.read_rows(out)}
    rows = round3.read_rows(out)
    for path in sorted((folder / "parsed").glob("*.json")):
        parsed = json.loads(path.read_text(encoding="utf-8"))
        name = Path(str(parsed.get("source", ""))).name
        if name in done:
            continue
        opening = doc_cards.opening_text(parsed, int(config.doc_card_chars))
        started = time.monotonic()
        for attempt in range(5):
            try:
                message = model.invoke([HumanMessage(content=prompt.format(name=name, text=opening or "（无文字内容）"))])
                break
            except Exception as exc:  # 限流 and other transient errors
                print("retry", name, type(exc).__name__, flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            continue
        usage = getattr(message, "usage_metadata", None) or {}
        try:
            card = doc_cards.parse_card(str(message.content))
        except (ValueError, TypeError):
            card = {}
        field = re.search(r'"领域"\s*:\s*"([^"\n]*)"', str(message.content))
        if field:
            card["fields"] = field.group(1).strip()
        rows.append({"name": name, "doc_id": path.stem, "card": card, "raw": str(message.content)[:400],
                     "tokens": int(usage.get("total_tokens") or 0), "seconds": round(time.monotonic() - started, 2)})
        round3.write_rows(out, rows)
        print(len(rows), name, card.get("doc_type"), card.get("fields", ""), "|", card.get("summary", ""), flush=True)
        time.sleep(args.pause)


if __name__ == "__main__":
    main()
