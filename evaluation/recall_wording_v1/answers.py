"""Answers on Qwen3-8B for the survey questions (and a few fact questions as a guard), through
the app's own graph: analyse, retrieve, build context, answer, fact check.

  python evaluation/recall_wording_v1/answers.py --name before [--index base] [--pause 30] [--set field=value ...]
  python evaluation/recall_wording_v1/answers.py --name after --set wording_expansion=true --set survey_route=true ...

Scored by rules, no judge model:
  files     of a question's gold files, how many the answer cites
  amounts   of the gold files' amounts (fund questions), how many the answer states; an
            amount counts when the answer cites its file and contains the figure
  refused   the reply is a refusal ("资料里没有找到…")
Answers are kept in work/answers/<name>.jsonl (synthetic text); the summary in
results/answers_<name>.json. Calls are paced: Qwen3-8B's tokens-per-minute limit is low.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402

from rag_agent.answering import CITATION  # noqa: E402

GUARD = ("A1-1", "A2-1", "A3-1", "B1-1", "B2-1", "C1-1", "C2-1", "D1-1", "E1-1", "E2-1", "F1-1", "G1-1", "G2-1", "H1-1", "I1-1", "I3-1")


def stated(amount: str, answer: str) -> bool:
    """Is the figure in the answer ("6万" also as "6 万元"; "0.8万" must not match "10.8万")?"""
    number, unit = re.match(r"([\d.:]+)(.*)", amount).groups()
    return bool(re.search(rf"(?<![\d.]){re.escape(number)}\s*{re.escape(unit)}", answer))


def score(question: dict, answer: str, refused: bool) -> dict:
    sources = {Path(name.strip()).stem if "." in name else name.strip() for name, _ in CITATION.findall(answer)}
    cited = [g for g in question["gold"] if Path(g["file"]).stem in sources]
    amounts = [(g, a) for g in question["gold"] for a in g["amounts"]]
    right = [(g, a) for g, a in amounts if g in cited and stated(a, answer)]
    first = [g for g in question["gold"] if g["amounts"] and g in cited and stated(g["amounts"][0], answer)]
    return {"gold_files": len(question["gold"]), "files_cited": len(cited), "amounts": len(amounts), "amounts_right": len(right),
            "files_with_first_amount": len(first), "refused": bool(refused)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--index", default="base")
    parser.add_argument("--pause", type=float, default=30.0)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    overrides = harness.parse_overrides(args.overrides)
    retriever, recorder, counter, store, cache = harness.open_retriever(args.index, **overrides)
    config = retriever.settings

    from langchain_core.messages import HumanMessage
    from db.parent_store_manager import ParentStoreManager
    from rag_agent.graph import create_agent_graph
    from services.model_clients import build_chat_model

    graph = create_agent_graph(build_chat_model(config), retriever, config, parent_store=ParentStoreManager(config))
    questions = [q for q in harness.load_questions() if q["type"] == "survey" or q["id"] in GUARD]
    if args.only:
        questions = [q for q in questions if q["id"] in set(args.only.split(","))]
    folder = harness.WORK / "answers"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{args.name}.jsonl"
    done = {}
    if path.is_file():
        done = {row["id"]: row for row in (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())}
    rows = []
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            if question["id"] in done:
                rows.append(done[question["id"]])
                continue
            started = time.perf_counter()
            for attempt in range(3):
                state = graph.invoke({"messages": [HumanMessage(content=question["question"])]},
                                     {"configurable": {"thread_id": f"{args.name}-{question['id']}-{attempt}"}})
                if "限流" not in str(state.get("answer", "")) and not state.get("model_error"):
                    break
                time.sleep(60)
            answer = str(state.get("answer", ""))
            row = {"id": question["id"], "type": question["type"], "kind": question["kind"], "question": question["question"],
                   "profile": state.get("answer_profile"), "answer": answer, "grounded": bool(state.get("grounded")),
                   "attempts": state.get("retrieval_attempts"), "passages": len(state.get("context_docs") or []),
                   "passage_files": len({d.get("doc_id") for d in state.get("context_docs") or []}),
                   "seconds": round(time.perf_counter() - started, 1), **score(question, answer, state.get("refused"))}
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(row["id"], row["profile"], "files", row["files_cited"], "/", row["gold_files"], "amounts", row["amounts_right"], "/",
                  row["amounts"], "refused" if row["refused"] else "", f"{row['seconds']} s", flush=True)
            time.sleep(args.pause)

    def block(items: list[dict]) -> dict:
        if not items:
            return {}
        return {"questions": len(items), "refused": sum(r["refused"] for r in items),
                "gold_files": sum(r["gold_files"] for r in items), "files_cited": sum(r["files_cited"] for r in items),
                "files_cited_per_question": round(statistics.fmean(r["files_cited"] for r in items), 2),
                "three_or_more_files": sum(r["files_cited"] >= 3 for r in items),
                "amounts": sum(r["amounts"] for r in items), "amounts_right": sum(r["amounts_right"] for r in items),
                "files_with_first_amount": sum(r["files_with_first_amount"] for r in items),
                "passage_files": round(statistics.fmean(r["passage_files"] for r in items), 1),
                "seconds_median": statistics.median(r["seconds"] for r in items)}

    summary = {"name": args.name, "index": args.index, "overrides": overrides,
               "survey": block([r for r in rows if r["type"] == "survey"]),
               "survey_fund": block([r for r in rows if r["type"] == "survey" and r["kind"] == "fund"]),
               "fact_guard": block([r for r in rows if r["type"] == "fact"])}
    if not args.only or len(rows) > 5:
        (harness.RESULTS / f"answers_{args.name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    store.close()
    cache.close()


if __name__ == "__main__":
    main()
