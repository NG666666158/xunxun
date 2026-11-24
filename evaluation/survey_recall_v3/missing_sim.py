"""Item 3 offline: what the "missing file" line (survey_answer.missing_lines) would add to answers
already written, with the retrieval of the same settings (cached, no model call).

  python evaluation/survey_recall_v3/missing_sim.py --answers ../survey_recall_v2/work/answers/final.jsonl --questions tuning

For each answer: the files added, whether each is a gold file of the question, and the score of
the answer before and after (files cited, amounts stated; recall_wording_v1/answers.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3  # noqa: E402
from round3 import harness  # noqa: E402

V2 = Path(__file__).resolve().parent.parent / "survey_recall_v2"
sys.path.insert(0, str(V2.parent / "recall_wording_v1"))
import answers as v1  # noqa: E402

from rag_agent import survey_answer  # noqa: E402
from rag_agent.answering import retrieval_limits  # noqa: E402
from retrieval.survey import answers_in, rule_name, survey_question, survey_topics  # noqa: E402


def questions(kind: str) -> dict[str, dict]:
    if kind == "tuning":
        return {q["id"]: q for q in round3.surveys()}
    path = V2 / f"{kind}.jsonl" if kind in ("holdout", "negative") else Path(kind)
    return {q["id"]: q for q in round3.read_rows(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--questions", default="tuning")
    parser.add_argument("--name", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    retriever, recorder, counter, store, cache = harness.open_retriever("base", offline=True, **harness.parse_overrides(args.overrides))
    qs = questions(args.questions)
    out = {"any": [], "rule": [], "topic": []}
    for row in round3.read_rows(Path(args.answers)):
        q = qs.get(row["id"])
        if q is None or row.get("refused"):
            continue
        survey = survey_question(q["question"], True)
        hits = retriever.retrieve(q["question"], mode="hybrid_rerank", **retrieval_limits("survey"))
        blocks = [{"source": h.source, "locator": h.locator or h.heading, "text": h.text} for h in hits]
        gold = {g["file"] for g in q.get("gold", [])}
        before = v1.score(q, row["answer"], False)
        topics = survey_topics(q["question"])
        on_topic = lambda source: rule_name(source) and survey_answer.about_topic(source, topics)  # noqa: E731
        for variant, rule in (("any", None), ("rule", rule_name), ("topic", on_topic)):
            text, added = survey_answer.missing_lines(row["answer"], blocks, survey.ask if survey else "", answers_in, rule)
            after = v1.score(q, text, False)
            out[variant].append({"id": row["id"], "added": added, "gold_added": [s for s in added if s in gold],
                                 "before": before, "after": after, "answer": text if added else ""})
    summary = {}
    for variant, rows in out.items():
        summary[variant] = {
            "answers": len(rows), "answers_with_lines": sum(bool(r["added"]) for r in rows),
            "lines": sum(len(r["added"]) for r in rows), "gold_lines": sum(len(r["gold_added"]) for r in rows),
            "files_cited": [sum(r["before"]["files_cited"] for r in rows), sum(r["after"]["files_cited"] for r in rows)],
            "amounts_right": [sum(r["before"]["amounts_right"] for r in rows), sum(r["after"]["amounts_right"] for r in rows)],
            "three_or_more": [sum(r["before"]["files_cited"] >= 3 for r in rows), sum(r["after"]["files_cited"] >= 3 for r in rows)],
            "added": [(r["id"], r["added"], r["gold_added"]) for r in rows if r["added"]],
        }
    round3.write_rows(round3.WORK / f"missing_{args.name}.jsonl", out["any"] + out["rule"] + out["topic"])
    round3.write_json(round3.RESULTS / f"missing_{args.name}.json", summary)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "added"} for k, v in summary.items()}, ensure_ascii=False, indent=1))
    for variant in summary:
        for item in summary[variant]["added"]:
            print(variant, item)
    cache.close()


if __name__ == "__main__":
    main()
