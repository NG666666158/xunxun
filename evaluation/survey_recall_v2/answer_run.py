"""Answers on Qwen3-8B through the app's graph, for the tuning survey questions (with the fact
guard of round one), the held-out ones or the negative ones.

  python evaluation/survey_recall_v2/answer_run.py --name files --questions tuning [--set field=value ...] [--pause 30]

Scored by rules as in evaluation/recall_wording_v1/answers.py (files cited, amounts stated,
refused). A negative question is right when the answer is a refusal or cites none of the
library's guides. Answers: work/answers/<name>[_set].jsonl; summary: results/answers_<name>[_set].json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shared  # noqa: E402
from shared import harness  # noqa: E402

import answers as v1  # noqa: E402  (evaluation/recall_wording_v1/answers.py)

from rag_agent.answering import CITATION  # noqa: E402


def block(items: list[dict]) -> dict:
    if not items:
        return {}
    return {"questions": len(items), "refused": sum(r["refused"] for r in items),
            "gold_files": sum(r["gold_files"] for r in items), "files_cited": sum(r["files_cited"] for r in items),
            "files_cited_per_question": round(statistics.fmean(r["files_cited"] for r in items), 2),
            "three_or_more_files": sum(r["files_cited"] >= 3 for r in items),
            "amounts": sum(r["amounts"] for r in items), "amounts_right": sum(r["amounts_right"] for r in items),
            "passage_files": round(statistics.fmean(r["passage_files"] for r in items), 1),
            "context_chars": round(statistics.fmean(r.get("context_chars", 0) for r in items)),
            "seconds_median": statistics.median(r["seconds"] for r in items)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--questions", default="tuning", choices=("tuning", "holdout", "negative"))
    parser.add_argument("--pause", type=float, default=30.0)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--survey-only", action="store_true", help="tuning set without the fact guard")
    args = parser.parse_args()
    overrides = harness.parse_overrides(args.overrides)
    retriever, recorder, counter, store, cache = harness.open_retriever("base", **overrides)
    config = retriever.settings

    from langchain_core.messages import HumanMessage
    from db.parent_store_manager import ParentStoreManager
    from rag_agent.graph import create_agent_graph
    from services.model_clients import build_chat_model

    graph = create_agent_graph(build_chat_model(config), retriever, config, parent_store=ParentStoreManager(config))
    if args.questions == "tuning":
        questions = [q for q in harness.load_questions() if q["type"] == "survey" or (q["id"] in v1.GUARD and not args.survey_only)]
    else:
        questions = shared.read_rows(shared.HERE / f"{args.questions}.jsonl")
    suffix = "" if args.questions == "tuning" else f"_{args.questions}"
    path = shared.WORK / "answers" / f"{args.name}{suffix}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["id"]: row for row in shared.read_rows(path)}
    rows = []
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            if question["id"] in done:
                rows.append(done[question["id"]])
                continue
            started = time.perf_counter()
            for attempt in range(3):
                state = graph.invoke({"messages": [HumanMessage(content=question["question"])]},
                                     {"configurable": {"thread_id": f"sv2-{args.name}-{question['id']}-{attempt}"}})
                if "限流" not in str(state.get("answer", "")) and not state.get("model_error"):
                    break
                time.sleep(60)
            answer = str(state.get("answer", ""))
            docs = state.get("context_docs") or []
            row = {"id": question["id"], "type": question["type"], "kind": question["kind"], "question": question["question"],
                   "profile": state.get("answer_profile"), "answer": answer, "grounded": bool(state.get("grounded")),
                   "passages": len(docs), "passage_files": len({d.get("doc_id") for d in docs}),
                   "passage_sources": sorted({str(d.get("source")) for d in docs}),
                   "context_chars": sum(len(str(d.get("text", ""))) for d in docs),
                   "seconds": round(time.perf_counter() - started, 1), **v1.score(question, answer, state.get("refused"))}
            if question["type"] == "negative":
                row["cites"] = len(CITATION.findall(answer))
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(row["id"], row["profile"], "files", row["files_cited"], "/", row["gold_files"], "amounts", row["amounts_right"], "/",
                  row["amounts"], "refused" if row["refused"] else "", f"{row['seconds']} s", flush=True)
            time.sleep(args.pause)
    summary = {"name": args.name, "questions_set": args.questions, "overrides": overrides,
               "survey": block([r for r in rows if r["type"] == "survey"]),
               "survey_fund": block([r for r in rows if r["type"] == "survey" and r["kind"] == "fund"]),
               "fact_guard": block([r for r in rows if r["type"] == "fact"])}
    negative = [r for r in rows if r["type"] == "negative"]
    if negative:
        summary["negative"] = {"questions": len(negative), "refused": sum(r["refused"] for r in negative),
                               "answered_with_citations": sum(bool(r.get("cites")) and not r["refused"] for r in negative)}
    shared.write_json(shared.RESULTS / f"answers_{args.name}{suffix}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    store.close()
    cache.close()


if __name__ == "__main__":
    main()
