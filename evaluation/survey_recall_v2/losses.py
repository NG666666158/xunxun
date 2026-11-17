"""Where each gold file of a survey question is lost, step by step.

  python evaluation/survey_recall_v2/losses.py --name main [--answers work/answers/main.jsonl] [--set field=value ...]

Retrieval runs once per question with the given configuration (cached model outputs), and
each gold file gets the first step that loses it:

  search    its gold passage is not among the candidates of a search four times as deep
            (neither for the question nor for its expanded wording)
  cut       found by that deeper search, but not sent to the reranker (pool size, per-file cap,
            or with survey_file_first: its file is not among the candidate files)
  file_passage  other passages of its file were scored, not the gold one (per-file cap)
  refused   scored, but the whole question was dropped as "no evidence" (best score below the floor)
  floor     scored below survey_floor
  shape     the passage does not look like an answer (no amount / no date)
  passage   its file is kept, with another passage
  files     its file is not among the files kept (survey_files, the passage limit)
  kept      the gold passage is in what the retriever returns

With --answers (the records of answers.py for the same configuration), the kept files go on:

  unused    kept, but the answer does not cite the file
  no_amount cited, none of its amounts stated (fund questions)
  part      cited, some of its amounts stated
  full      cited, every amount stated (other kinds: cited)

Per-question records: work/losses/<name>.jsonl; the table: results/losses_<name>.json.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shared as common  # noqa: E402
from shared import harness  # noqa: E402

from rag_agent.answering import CITATION, classify_question, retrieval_limits  # noqa: E402
from retrieval import wording  # noqa: E402
from retrieval.survey import has_shape, survey_question  # noqa: E402

RETRIEVAL_STEPS = ("search", "cut", "file_passage", "refused", "floor", "shape", "passage", "files", "kept")
ANSWER_STEPS = ("unused", "no_amount", "part", "full")
DEPTH = 4


def stated(amount: str, answer: str) -> bool:
    import re

    number, unit = re.match(r"([\d.:]+)(.*)", amount).groups()
    return bool(re.search(rf"(?<![\d.]){re.escape(number)}\s*{re.escape(unit)}", answer))


def retrieval_step(question: dict, retriever, recorder, counter, chunks, by_text) -> dict:
    text = question["question"]
    recorder.reset()
    profile = classify_question(text, bool(getattr(retriever.settings, "survey_route", False)))
    hits = retriever.retrieve(text, mode="hybrid_rerank", **retrieval_limits(profile))
    scores: dict[str, float] = {}
    for call in recorder.calls:
        for passage, score in call["scores"].items():
            chunk_id = by_text.get(passage)
            if chunk_id is not None:
                scores[chunk_id] = max(score, scores.get(chunk_id, -1.0))
    order = sorted(scores, key=lambda cid: -scores[cid])
    by_id = {c["id"]: c for c in chunks}
    final = {hit.id for hit in hits}
    final_sources = {hit.source for hit in hits}
    survey = survey_question(text)
    settings = retriever.settings
    limit = max(retriever.candidate_pool(), int(getattr(settings, "survey_pool", 60))) * DEPTH
    queries = [text]
    expansion = wording.expand(text) if getattr(settings, "wording_expansion", False) else None
    if expansion is not None:
        queries.append(expansion.text)
    deep: dict[str, int] = {}
    dense: dict[str, int] = {}
    sparse: dict[str, int] = {}
    for query in queries:
        vector = counter.embed_query(query)
        for rank, hit in enumerate(retriever.store.search_hybrid(query, vector, limit=limit), start=1):
            deep[hit.id] = min(rank, deep.get(hit.id, rank))
        for rank, hit in enumerate(retriever.store.search_dense(vector, limit=len(chunks)), start=1):
            dense[hit.id] = min(rank, dense.get(hit.id, rank))
        for rank, hit in enumerate(retriever.store.search_sparse(query, limit=len(chunks)), start=1):
            sparse[hit.id] = min(rank, sparse.get(hit.id, rank))
    floor = float(getattr(settings, "survey_floor", 0.01))
    rows = []
    for gold in question["gold"]:
        ids = harness.gold_chunk_ids(chunks, gold)
        scored = [cid for cid in ids if cid in scores]
        best = max(scored, key=lambda cid: scores[cid]) if scored else None
        if any(cid in final for cid in ids):
            step = "kept"
        elif not scored:
            scored_sources = {by_id[cid]["source"] for cid in scores}
            step = ("file_passage" if gold["file"] in scored_sources
                    else "cut" if any(cid in deep for cid in ids) else "search")
        elif not hits:
            step = "refused"
        elif survey is not None and scores[best] < floor:
            step = "floor"
        elif survey is not None and not any(has_shape(by_id[cid]["text"], survey.ask) for cid in scored):
            step = "shape"
        elif gold["file"] in final_sources:
            step = "passage"
        else:
            step = "files"
        rows.append({
            "file": gold["file"], "step": step, "score": round(scores[best], 4) if best else None,
            "rank": order.index(best) + 1 if best else None,
            "deep_rank": min((deep[cid] for cid in ids if cid in deep), default=None),
            "dense_rank": min((dense[cid] for cid in ids if cid in dense), default=None),
            "sparse_rank": min((sparse[cid] for cid in ids if cid in sparse), default=None),
            "file_kept": gold["file"] in final_sources, "amounts": gold["amounts"],
        })
    return {
        "id": question["id"], "kind": question["kind"], "group": question["group"], "question": text, "profile": profile,
        "ask": survey.ask if survey else None, "scored": len(scores), "top_score": round(scores[order[0]], 4) if order else None,
        "kept": len(hits), "kept_files": len(final_sources), "kept_sources": sorted(final_sources), "gold": rows,
    }


def answer_step(row: dict, answer: dict | None) -> None:
    """Adds ``use`` to each gold file of ``row`` from the answer's text."""
    if answer is None:
        return
    text = answer["answer"]
    sources = {Path(name.strip()).stem if "." in name else name.strip() for name, _ in CITATION.findall(text)}
    row["refused_answer"] = bool(answer.get("refused"))
    for gold in row["gold"]:
        cited = Path(gold["file"]).stem in sources
        right = [a for a in gold["amounts"] if stated(a, text)] if cited else []
        gold["cited"] = cited
        gold["amounts_right"] = len(right)
        if not cited:
            gold["use"] = "unused"
        elif not gold["amounts"]:
            gold["use"] = "full"
        else:
            gold["use"] = "full" if len(right) == len(gold["amounts"]) else "part" if right else "no_amount"


def summarize(rows: list[dict]) -> dict:
    golds = [(row, gold) for row in rows for gold in row["gold"]]
    steps = Counter(gold["step"] for _, gold in golds)
    out = {
        "questions": len(rows), "gold_files": len(golds),
        "steps": {step: steps.get(step, 0) for step in RETRIEVAL_STEPS},
        "by_kind": {kind: {step: sum(1 for r, g in golds if r["kind"] == kind and g["step"] == step) for step in RETRIEVAL_STEPS}
                    for kind in ("fund", "time", "who")},
        "three_or_more": sum(sum(g["step"] == "kept" for g in row["gold"]) >= 3 for row in rows),
        "all_kept": sum(all(g["step"] == "kept" for g in row["gold"]) for row in rows),
    }
    if any("use" in gold for _, gold in golds):
        kept = [(r, g) for r, g in golds if g["file_kept"]]
        use = Counter(g["use"] for _, g in kept)
        out["use_of_kept_files"] = {step: use.get(step, 0) for step in ANSWER_STEPS}
        out["cited_without_being_kept"] = sum(1 for r, g in golds if not g["file_kept"] and g.get("cited"))
        out["files_cited"] = sum(1 for _, g in golds if g.get("cited"))
        out["amounts"] = sum(len(g["amounts"]) for _, g in golds)
        out["amounts_right"] = sum(g.get("amounts_right", 0) for _, g in golds)
        out["amounts_of_kept_files"] = sum(len(g["amounts"]) for _, g in kept)
        out["amounts_lost_before_the_answer"] = out["amounts"] - out["amounts_of_kept_files"]
        out["questions_refused"] = sum(bool(row.get("refused_answer")) for row in rows)
    return out


def negatives(rows: list[dict]) -> dict:
    """Questions about programmes the library lacks: what is kept anyway."""
    return {"questions": len(rows), "kept_nothing": sum(row["kept"] == 0 for row in rows),
            "kept_files": [row["kept_files"] for row in rows], "top_scores": [row["top_score"] for row in rows],
            "survey_profile": sum(row["profile"] == "survey" for row in rows)}


def table(rows: list[dict]) -> str:
    lines = ["| Question | Kind | Gold files | Kept | Lost at | Answer cites | Amounts stated |", "|---|---|---:|---:|---|---:|---:|"]
    for row in rows:
        lost = Counter(g["step"] for g in row["gold"] if g["step"] != "kept")
        where = ", ".join(f"{step} {count}" for step, count in lost.items()) or "-"
        kept = sum(g["step"] == "kept" for g in row["gold"])
        answered = bool(row["gold"]) and "use" in row["gold"][0]
        cited = sum(bool(g.get("cited")) for g in row["gold"]) if answered else "-"
        amounts = sum(len(g["amounts"]) for g in row["gold"])
        right = f"{sum(g.get('amounts_right', 0) for g in row['gold'])} / {amounts}" if amounts and answered else "-"
        if not row["gold"]:
            where = f"kept {row['kept']} passages from {row['kept_files']} files, top score {row['top_score']}"
        lines.append(f"| {row['id']} {row['question']} | {row['kind']} | {len(row['gold'])} | {kept} | {where} | {cited} | {right} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--index", default="base")
    parser.add_argument("--answers", default="")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--questions", default="tuning", choices=("tuning", "holdout", "negative"))
    args = parser.parse_args()
    overrides = harness.parse_overrides(args.overrides)
    retriever, recorder, counter, store, cache = harness.open_retriever(args.index, offline=args.offline, **overrides)
    # Every scoring of the question counts, also the second one (wording_rescore) that runs on
    # another thread: a passage's score is the better of the two, as in the retriever.
    import types

    recorder._local = types.SimpleNamespace()
    chunks = harness.all_chunks(harness.index_dir(args.index) / "knowledge.sqlite3")
    by_text = {c["text"]: c["id"] for c in chunks}
    answers = {row["id"]: row for row in common.read_rows(Path(args.answers))} if args.answers else {}
    rows = []
    questions = common.surveys() if args.questions == "tuning" else common.read_rows(common.HERE / f"{args.questions}.jsonl")
    for question in questions:
        row = retrieval_step(question, retriever, recorder, counter, chunks, by_text)
        answer_step(row, answers.get(question["id"]))
        rows.append(row)
    suffix = "" if args.questions == "tuning" else f"_{args.questions}"
    common.write_rows(common.WORK / "losses" / f"{args.name}{suffix}.jsonl", rows)
    summary = {"name": args.name, "questions_set": args.questions, "overrides": overrides,
               **(negatives(rows) if args.questions == "negative" else summarize(rows))}
    # Per question, each gold file's step (synthetic names only: this goes into results/).
    summary["rows"] = [{"id": row["id"], "question": row["question"], "kind": row["kind"], "kept_files": row["kept_files"],
                        "gold": [{"file": g["file"], "step": g["step"], "use": g.get("use"), "score": g["score"],
                                  "amounts": len(g["amounts"]), "amounts_right": g.get("amounts_right")} for g in row["gold"]]}
                       for row in rows]
    common.write_json(common.RESULTS / f"losses_{args.name}{suffix}.json", summary)
    sys.stdout.reconfigure(encoding="utf-8")
    print(table(rows))
    import json

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    store.close()
    cache.close()


if __name__ == "__main__":
    main()
