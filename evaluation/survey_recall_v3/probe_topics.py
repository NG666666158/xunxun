"""How a file that never names a question's topic could still be found: the reranker's topic score
against each gold file's passages, its name, its file entry (name, headings, opening; no model)
and its 支持方向-like section, next to the same scores of the other guides.

  python evaluation/survey_recall_v3/probe_topics.py [--offline]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3 as shared  # noqa: E402
from round3 import harness  # noqa: E402

from retrieval.file_index import build_entry_text  # noqa: E402
from retrieval.survey import survey_topics  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--ids", default="S01,S02,S03,S04,S15,S21,S12,S19,S22")
    args = parser.parse_args()
    retriever, recorder, counter, store, cache = harness.open_retriever("base", offline=args.offline)
    folder = harness.index_dir("base")
    chunks = harness.all_chunks(folder / "knowledge.sqlite3")
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["source"], []).append(c)
    entries = {}
    for path in (folder / "parsed").glob("*.json"):
        parsed = json.loads(path.read_text(encoding="utf-8"))
        name = Path(str(parsed.get("source", ""))).name
        entries[name] = build_entry_text(name, parsed)
    questions = {q["id"]: q for q in shared.surveys()}
    out = []
    for qid in args.ids.split(","):
        q = questions[qid]
        gold = {g["file"] for g in q["gold"]}
        for topic in survey_topics(q["question"]):
            names = sorted(entries)
            def scores(texts):
                items = retriever.reranker.rerank(topic, texts, top_n=len(texts))
                got = [0.0] * len(texts)
                for item in items:
                    got[item.index] = item.score
                return got
            s_name = scores([Path(n).stem for n in names])
            s_entry = scores([entries[n] for n in names])
            rank_entry = sorted(range(len(names)), key=lambda i: -s_entry[i])
            rank_name = sorted(range(len(names)), key=lambda i: -s_name[i])
            for g in sorted(gold):
                i = names.index(g)
                out.append({"q": qid, "topic": topic, "file": g, "name": round(s_name[i], 4), "name_rank": rank_name.index(i) + 1,
                            "entry": round(s_entry[i], 4), "entry_rank": rank_entry.index(i) + 1})
            top = [(Path(names[i]).stem[:20], round(s_entry[i], 3)) for i in rank_entry[:6]]
            print(qid, topic, "top entries:", top)
    for row in out:
        print(row)
    cache.close()

if __name__ == "__main__":
    main()
