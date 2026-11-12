"""Type questions ("找一下简历") on a data directory: the doc_cards_v1 method and scores.

    python -I evaluation/doc_cards_after_import/questions.py --data <data copy> \
        --questions <doc_cards_v1 questions.jsonl> --manifest <doc_cards_v1 manifest.json> \
        --label public_batch8 --env project/.env

Per question, on the quick-search path (rules only, as /files/search and the chat fast path run
it; real bge-m3 and bge-reranker-v2-m3, no chat model):
* hit@1 / hit@3: a right file first / in the top 3 (a "pure" question, nothing but the type, is
  right with any file of that type; the others name one file);
* type@5: of the top 5, how many really are of the type (of min(5, files of that type));
* mention@5: files in the top 5 that only mention the word; mention_first: such a file above
  every real one;
* type_recall: of the files of the type, how many the list marks as that type (a pure question
  lists every one);
* untyped: candidate files still without a card (the answer then says the list may be short).
Writes ``results/questions_<label>.json`` (scores; file names in the public set are anonymous)
and ``work/questions_<label>.jsonl`` (the question texts quote names from the data).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out
from common import HERE, load_key, read_jsonl, use_project  # noqa: E402

NOW = datetime(2026, 11, 9, 12, 0)


class Recorder:
    """Stands in for the card builder: remembers which files a search moved up."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def prioritize(self, doc_ids) -> int:
        ids = list(doc_ids)
        self.asked.extend(ids)
        return len(ids)


def score(question: dict, files: list[dict], types: dict[str, str], mentions: dict[str, str]) -> dict:
    names = [f["name"] for f in files]
    gold = set(question["gold"])
    rank = next((i for i, name in enumerate(names, start=1) if name in gold), None)
    asked = question["doc_type"]
    top5 = names[:5]
    possible = min(5, sum(1 for t in types.values() if t == asked))
    first_real = next((i for i, name in enumerate(names) if types.get(name) == asked), None)
    first_mention = next((i for i, name in enumerate(names) if mentions.get(name) == asked), None)
    of_type = sum(1 for t in types.values() if t == asked)
    marked = sum(1 for f in files if types.get(f["name"]) == asked and "type" in f["matched_by"])
    return {
        "rank": rank, "hit1": float(rank == 1), "hit3": float(rank is not None and rank <= 3),
        "type5": sum(types.get(n) == asked for n in top5) / possible if possible else 0.0,
        "mention5": sum(1 for n in top5 if mentions.get(n) == asked),
        "mention_first": float(first_mention is not None and (first_real is None or first_mention < first_real)),
        "type_recall": marked / of_type if of_type else 0.0,
    }


def summarize(rows: list[dict]) -> dict:
    def block(rs: list[dict]) -> dict:
        times = sorted(r["elapsed_ms"] for r in rs)
        keys = ("hit1", "hit3", "type5", "mention5", "mention_first", "type_recall")
        return {"n": len(rs), **{key: round(statistics.mean(r[key] for r in rs), 3) for key in keys},
                "p50_ms": times[len(times) // 2], "max_ms": times[-1]}

    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    return {"all": block(rows), "by_category": {c: block(rs) for c, rs in groups.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--env", type=Path)
    parser.add_argument("--categories", default="", help="only these categories (pure,named,topic)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    use_project()
    load_key(args.env)

    from config import Settings
    from core.user_settings import apply_user_settings
    from library import FileRegistry
    from retrieval.embedded_store import EmbeddedStore
    from retrieval.file_search import FileSearchService
    from services.model_clients import SiliconFlowEmbeddings, SiliconFlowReranker

    questions = read_jsonl(args.questions)
    if args.categories:
        questions = [q for q in questions if q["category"] in args.categories.split(",")]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    types = {item["name"]: item["type"] for item in manifest}
    mentions = {item["name"]: item["mentions"] for item in manifest if item.get("mentions")}
    settings = apply_user_settings(Settings(_env_file=None, data_dir=args.data.resolve()))
    settings.doc_type_search = True
    store = EmbeddedStore(settings, db_path=settings.embedded_db_path)
    registry = FileRegistry(settings.registry_path)
    service = FileSearchService(store, SiliconFlowEmbeddings(settings), SiliconFlowReranker(settings), registry,
                                settings, clock=lambda: NOW)
    service.cards = Recorder()  # counts the files still without a card; builds nothing
    service.search("预热")
    rows = []
    for question in questions:
        started = time.perf_counter()
        result = service.search(question["question"], limit=100)
        elapsed = round((time.perf_counter() - started) * 1000)
        files = result["files"]
        row = {"id": question["id"], "category": question["category"], "elapsed_ms": elapsed,
               "untyped": result.get("untyped", 0),
               "names": [f["name"] for f in files[:8]],
               "reasons": [("type" if "type" in f["matched_by"] else "mention" if "mention" in f["matched_by"] else "")
                           for f in files[:8]],
               **score(question, files, types, mentions)}
        rows.append(row)
        print(f"{question['id']} rank={row['rank']} type5={row['type5']:.2f} recall={row['type_recall']:.2f} "
              f"{elapsed}ms", flush=True)
    summary = summarize(rows)
    (HERE / "work").mkdir(exist_ok=True)
    (HERE / "work" / f"questions_{args.label}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    per_question = [{k: r[k] for k in ("id", "category", "rank", "hit1", "hit3", "type5", "mention5",
                                        "mention_first", "type_recall", "untyped", "elapsed_ms")}
                    | {"top5_types": [types.get(n, "?") + ("·提到" if n in mentions else "") + ("·按类型" if x == "type" else "")
                                      for n, x in zip(r["names"][:5], r["reasons"][:5])]}
                    for r in rows]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"questions_{args.label}.json").write_text(
        json.dumps({"summary": summary, "questions": per_question}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
