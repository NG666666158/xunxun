"""Where one passage ends up for one question, on any index folder (a copy of a real library).

  python evaluation/recall_wording_v1/probe.py --data <folder with knowledge.sqlite3> \
      --question "…" --file "<part of the file name>" --marker "<text only the passage has>" [--set field=value ...]

Prints numbers and file counts only, never passage text: the passage's place in the vector
and BM25 rankings, whether it reached the reranker, its place and score after reranking, and
whether it was kept. Model outputs are cached under RW_WORK_DIR (point it outside the
repository when the library is private).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402

from rag_agent.answering import classify_question, retrieval_limits  # noqa: E402
from retrieval.embedded_store import build_match_query  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--depth", type=int, default=300)
    args = parser.parse_args()
    folder = Path(args.data)
    retriever, recorder, counter, store, cache = harness.open_retriever(folder, **harness.parse_overrides(args.overrides))
    chunks = harness.all_chunks(folder / "knowledge.sqlite3")
    marker = harness.squeeze(args.marker)
    gold = {c["id"] for c in chunks if args.file in c["source"] and marker in harness.squeeze(c["text"])}
    by_text = {c["text"]: c for c in chunks}
    out: dict = {"chunks": len(chunks), "files": len({c["doc_id"] for c in chunks}), "gold_chunks": len(gold)}

    def place(hits) -> int | None:
        return next((i for i, hit in enumerate(hits, start=1) if hit.id in gold), None)

    vector = counter.embed_query(args.question)
    out["dense_rank"] = place(store.search_dense(vector, limit=args.depth))
    out["bm25_rank"] = place(store.search_sparse(args.question, limit=args.depth))
    out["bm25_terms"] = len(build_match_query(args.question).split(" OR "))
    profile = classify_question(args.question)
    recorder.reset()
    hits = retriever.retrieve(args.question, mode="hybrid_rerank", **retrieval_limits(profile))
    scores: dict[str, float] = {}
    for call in recorder.calls:
        for text, score in call["scores"].items():
            if text in by_text:
                cid = by_text[text]["id"]
                scores[cid] = max(score, scores.get(cid, -1.0))
    order = sorted(scores, key=lambda cid: -scores[cid])
    ids = {c["id"]: c for c in chunks}
    out.update({
        "profile": profile, "pool": len(scores), "pool_files": len({ids[c]["doc_id"] for c in scores}),
        "pool_top3_files_share": round(sum(n for _, n in Counter(ids[c]["doc_id"] for c in scores).most_common(3)) / max(1, len(scores)), 2),
        "in_pool": any(c in scores for c in gold),
        "rank": next((i for i, c in enumerate(order, start=1) if c in gold), None),
        "score": round(max((scores[c] for c in gold if c in scores), default=0.0), 3),
        "top_score": round(scores[order[0]], 3) if order else None,
        "kept": len(hits), "kept_files": len({hit.doc_id for hit in hits}),
        "gold_kept": any(hit.id in gold for hit in hits), "gold_file_kept": any(args.file in hit.source for hit in hits),
        "rerank_calls": len(recorder.calls), "rerank_docs": sum(len(c["texts"]) for c in recorder.calls),
    })
    print(json.dumps(out, ensure_ascii=False, indent=1))
    store.close()
    cache.close()


if __name__ == "__main__":
    main()
