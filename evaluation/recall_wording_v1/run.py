"""Retrieval over the synthetic library, question by question, with one configuration.

  python evaluation/recall_wording_v1/run.py --name base [--index base] [--set field=value ...] [--offline]

For each question it records what the reranker was sent (the candidate pool), where the gold
passage ranks after reranking, and what the retriever finally keeps. Per-question records go
to work/runs/<name>.jsonl; the summary to results/<name>.json.

Measures
  pool      the gold passage is among the candidates sent to the reranker
  rank      its place after reranking (median / mean of those in the pool; top3 = share ranked <= 3)
  final     the gold passage is among the passages kept for the answer
  survey    of a survey question's gold files, how many have their gold passage kept (and how
            many have any passage kept)
  cost      passages scored and estimated rerank tokens per question, query texts embedded
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402

from rag_agent.answering import classify_question, retrieval_limits  # noqa: E402
from retrieval import multipart  # noqa: E402


def run_question(question: dict, retriever, recorder, counter, chunks: list[dict], by_text: dict[str, str]) -> dict:
    recorder.reset()
    before = (counter.texts, counter.requests)
    profile = classify_question(question["question"], bool(getattr(retriever.settings, "survey_route", False)))
    started = time.perf_counter()
    hits = retriever.retrieve(question["question"], mode="hybrid_rerank", **retrieval_limits(profile))
    seconds = time.perf_counter() - started
    # The pool: everything scored for this question, with its best score against any query.
    scores: dict[str, float] = {}
    for call in recorder.calls:
        for text, score in call["scores"].items():
            chunk_id = by_text.get(text)
            if chunk_id is not None:
                scores[chunk_id] = max(score, scores.get(chunk_id, -1.0))
    order = sorted(scores, key=lambda cid: -scores[cid])
    final = [hit.id for hit in hits]
    final_sources = {hit.source for hit in hits}
    gold_rows = []
    for gold in question["gold"]:
        ids = harness.gold_chunk_ids(chunks, gold)
        ranks = [order.index(cid) + 1 for cid in ids if cid in scores]
        gold_rows.append({
            "file": gold["file"], "chunks": len(ids), "in_pool": bool(ranks), "rank": min(ranks) if ranks else None,
            "score": max((scores[cid] for cid in ids if cid in scores), default=None),
            "in_final": any(cid in final for cid in ids), "file_in_final": gold["file"] in final_sources,
        })
    trace = multipart.last_trace()
    return {
        "id": question["id"], "type": question["type"], "kind": question["kind"], "profile": profile,
        "format": question.get("format", ""), "gold": gold_rows, "pool": len(scores), "final": len(final),
        "final_files": len(final_sources), "top_score": scores[order[0]] if order else None,
        "rerank_calls": len(recorder.calls), "rerank_docs": sum(len(c["texts"]) for c in recorder.calls),
        "rerank_tokens": sum(multipart.estimate_tokens(c["query"], c["texts"]) for c in recorder.calls),
        "embed_texts": counter.texts - before[0], "embed_requests": counter.requests - before[1],
        "seconds": round(seconds, 3), "trace": getattr(trace, "recall", None) if trace else None,
    }


def summarize(rows: list[dict]) -> dict:
    def block(items: list[dict]) -> dict:
        golds = [g for row in items for g in row["gold"]]
        ranks = [g["rank"] for g in golds if g["rank"]]
        return {
            "questions": len(items), "gold_passages": len(golds),
            "pool": round(sum(g["in_pool"] for g in golds) / max(1, len(golds)), 3),
            "rank_median": statistics.median(ranks) if ranks else None,
            "rank_mean": round(statistics.fmean(ranks), 1) if ranks else None,
            "top3": round(sum(1 for g in golds if g["rank"] and g["rank"] <= 3) / max(1, len(golds)), 3),
            "final": round(sum(g["in_final"] for g in golds) / max(1, len(golds)), 3),
            "file_final": round(sum(g["file_in_final"] for g in golds) / max(1, len(golds)), 3),
        }

    out = {"fact": block([r for r in rows if r["type"] == "fact"]), "generic": block([r for r in rows if r["type"] == "generic"])}
    for fmt in ("bold", "txt", "style", "md"):
        out[f"generic_{fmt}"] = block([r for r in rows if r["type"] == "generic" and r["format"] == fmt])
    for kind in ("fund", "time", "who"):
        out[f"fact_{kind}"] = block([r for r in rows if r["type"] == "fact" and r["kind"] == kind])
    for fmt in ("bold", "txt", "style", "md"):
        out[f"fact_{fmt}"] = block([r for r in rows if r["type"] == "fact" and r["format"] == fmt])
    surveys = [r for r in rows if r["type"] == "survey"]
    out["survey"] = block(surveys)
    out["survey"].update({
        "gold_files_per_question": round(statistics.fmean(len(r["gold"]) for r in surveys), 2) if surveys else 0,
        "gold_passages_kept": round(statistics.fmean(sum(g["in_final"] for g in r["gold"]) for r in surveys), 2) if surveys else 0,
        "gold_files_kept": round(statistics.fmean(sum(g["file_in_final"] for g in r["gold"]) for r in surveys), 2) if surveys else 0,
        "all_gold_passages": round(sum(all(g["in_final"] for g in r["gold"]) for r in surveys) / max(1, len(surveys)), 3),
        "three_or_more": round(sum(sum(g["in_final"] for g in r["gold"]) >= 3 for r in surveys) / max(1, len(surveys)), 3),
        "final_passages": round(statistics.fmean(r["final"] for r in surveys), 1) if surveys else 0,
    })
    out["cost"] = {
        "rerank_docs": round(statistics.fmean(r["rerank_docs"] for r in rows), 1),
        "rerank_tokens": round(statistics.fmean(r["rerank_tokens"] for r in rows)),
        "rerank_calls": round(statistics.fmean(r["rerank_calls"] for r in rows), 2),
        "embed_texts": round(statistics.fmean(r["embed_texts"] for r in rows), 2),
        "local_seconds_median": round(statistics.median(r["seconds"] for r in rows), 3),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--index", default="base")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--only", default="", help="comma-separated question ids")
    args = parser.parse_args()
    overrides = harness.parse_overrides(args.overrides)
    retriever, recorder, counter, store, cache = harness.open_retriever(args.index, offline=args.offline, **overrides)
    chunks = harness.all_chunks(harness.index_dir(args.index) / "knowledge.sqlite3")
    by_text = {c["text"]: c["id"] for c in chunks}
    questions = harness.load_questions()
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    rows = []
    for question in questions:
        rows.append(run_question(question, retriever, recorder, counter, chunks, by_text))
    runs = harness.WORK / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / f"{args.name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"name": args.name, "index": args.index, "overrides": overrides, "chunks": len(chunks), **summarize(rows)}
    if not args.only:
        harness.RESULTS.mkdir(exist_ok=True)
        (harness.RESULTS / f"{args.name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("fact", "generic", "survey", "cost")}, ensure_ascii=False, indent=1))
    store.close()
    cache.close()


if __name__ == "__main__":
    main()
