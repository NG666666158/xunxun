"""Failure examples for the report: per missed gold, where it was lost and what won instead.

For each selected question: the query and its BM25 tokens (jieba), the gold document's
best ranks in each route, the pool, the rerank order and the final hits, the gold's
rerank score against the cutoff, and the passages that outranked it.

Usage:
  python -I evaluation/suite/examples.py crud_rag base [--type multi_3doc] [--tag digits]
         [--loss not_retrieved] [--n 5] [--negatives]
Prints Markdown (read locally: it quotes dataset text, so it is not committed).
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out

from common import WORK, index_doc_id, iter_jsonl  # noqa: E402
import report  # noqa: E402
import runner  # noqa: E402

from retrieval.embedded_store import tokenize  # noqa: E402


def chunk_text(conn: sqlite3.Connection, prefix: str) -> str:
    row = conn.execute("SELECT text FROM chunks WHERE id >= ? AND id < ? LIMIT 1", (prefix, prefix + "g")).fetchone()
    return row[0] if row else ""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("run")
    parser.add_argument("--type", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--loss", default="")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--negatives", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    records, meta = report.load(args.dataset, args.run)
    settings = meta["retrieval_settings"]
    ds = runner.load_dataset(args.dataset)
    questions = {q["id"]: q for q in ds.questions}
    corpus = {}
    if isinstance(ds, runner.Prepared):
        corpus = {index_doc_id(args.dataset, row["doc_id"]): row for row in iter_jsonl(WORK / args.dataset / "prepared" / "corpus.jsonl")}
    conn = sqlite3.connect(f"{(ds.data_dir / 'knowledge.sqlite3').resolve().as_uri()}?mode=ro", uri=True)
    chosen = []
    for record in records:
        if args.type and record["type"] != args.type:
            continue
        if args.tag and args.tag not in record["tags"]:
            continue
        if args.negatives:
            if not record["answerable"]:
                chosen.append((record, []))
            continue
        losses = report.attribute(record, settings)
        if args.loss:
            losses = [l for l in losses if l["loss"] == args.loss]
        if losses:
            chosen.append((record, losses))
    random.Random(args.seed).shuffle(chosen)
    print(f"{len(chosen)} questions match; showing {min(args.n, len(chosen))}\n")
    for record, losses in chosen[: args.n]:
        q = questions[record["id"]]
        query = record.get("query_used") or q["query"]
        print(f"### {record['id']} ({record['type']}, profile {record['profile']}, max_keep {record['max_keep']})")
        print(f"- 问题：{query}")
        print(f"- BM25 词：{' / '.join(dict.fromkeys(tokenize(query)))}")
        if q.get("gold_answer"):
            print(f"- 标准答案：{q['gold_answer'][:150]}")
        for loss in losses:
            gold_doc = ds._gold[record["id"]][loss["gold"]] if hasattr(ds, "_gold") else ""
            print(f"- 金标 {loss['gold']}：{report.LOSS_NAMES[loss['loss']]}；dense 名次 {loss['dense_rank']}，BM25 名次 {loss['bm25_rank']}，"
                  f"池内名次 {loss['pool_rank']}，重排名次 {loss['rerank_rank']}，分数 {loss.get('score')}（最高 {loss.get('top_score')}）")
            if gold_doc in corpus:
                print(f"  - 金标文档开头：{corpus[gold_doc]['text'][:160].replace(chr(10), ' ')}")
        print("- 重排前 5：")
        for rank, hit in enumerate(record["stages"]["reranked"][:5], start=1):
            mark = "✔" if hit.get("g") else " "
            print(f"  {rank}. {mark} r={hit.get('r'):.3f} {chunk_text(conn, hit['c'])[:110].replace(chr(10), ' ')}")
        print(f"- 切后保留 {len(record['stages']['final'])} 条，金标命中 {sum(bool(h.get('g')) for h in record['stages']['final'])} 条\n")


if __name__ == "__main__":
    main()
