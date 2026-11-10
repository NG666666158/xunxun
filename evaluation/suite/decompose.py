"""What-if: rerank multi-part questions part by part (same reranker, no model change).

A question that asks two or three things ("…对哪些省份采取了何种响应，并且安徽省预计会…？")
is scored against each passage as a whole; a passage that answers one part gets a low
score and the cutoffs drop it. Here the question is split at its joining words, every
part scores the recorded pool, and each passage keeps its best score (the whole question
included). The app's selection then runs on those scores with the question type's cap
(or a raised one for multi-part questions).

Only questions ``replay.multi_part`` flags are touched; the others keep their record.
New (part, passage) pairs go through the rerank cache (one request per part).

Usage:
  python -I evaluation/suite/decompose.py crud_rag base [--types multi_2doc,multi_3doc] [--workers 2]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out

import common  # noqa: E402
from common import RESULTS, WORK  # noqa: E402
from cache import Cache, CachedReranker  # noqa: E402
import metrics  # noqa: E402
import replay  # noqa: E402
import report  # noqa: E402
import runner  # noqa: E402

_SPLIT = re.compile(r"[，,；;]?\s*(?:并且|同时|此外|另外|以及|而且|并|且)(?=[^，,。？?]{2,})|[；;]|？(?=.)")


def parts(query: str) -> list[str]:
    pieces = []
    for piece in _SPLIT.split(query):
        # "…包括哪些措施，俄罗斯对此有何反应？": a comma between two asks splits too.
        halves = re.split(r"[，,]", piece)
        current = ""
        for half in halves:
            if current and replay._ASKS_ZH.search(current) and replay._ASKS_ZH.search(half):
                pieces.append(current)
                current = half
            else:
                current = f"{current}，{half}" if current else half
        pieces.append(current)
    pieces = [p.strip(" ，,。？?") for p in pieces]
    pieces = [p for p in pieces if len(p) >= 6]
    return pieces if len(pieces) >= 2 else []


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("run")
    parser.add_argument("--types", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    records, meta = report.load(args.dataset, args.run)
    settings = meta["retrieval_settings"]
    ds = runner.load_dataset(args.dataset)
    queries = {q["id"]: q["query"] for q in ds.questions}
    config = common.settings(data_dir=ds.data_dir)
    cache = Cache(WORK / "cache.sqlite3")
    reranker = CachedReranker(cache, config, offline=args.offline)
    db = ds.data_dir / "knowledge.sqlite3"
    types = set(args.types.split(",")) if args.types else None
    chosen = [r for r in records if (types is None or r["type"] in types) and r["answerable"]]
    targets = [r for r in chosen if replay.multi_part(queries[r["id"]]) and parts(queries[r["id"]])]
    print(f"{len(chosen)} questions, {len(targets)} split into parts", flush=True)

    def texts_of(pool: list[dict]) -> list[str]:
        conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
        out = [conn.execute("SELECT text FROM chunks WHERE id >= ? AND id < ? LIMIT 1", (h["c"], h["c"] + "g")).fetchone()[0]
               for h in pool]
        conn.close()
        return out

    def rescore(record: dict) -> dict:
        query = queries[record["id"]]
        pool = record["stages"]["pool"]
        texts = texts_of(pool)
        best = [h.get("r") or 0.0 for h in pool]
        for part in parts(query):
            for i, score in enumerate(reranker.scores(part, texts)):
                best[i] = max(best[i], score)
        new_pool = [{**h, "r": s} for h, s in zip(pool, best)]
        return {**record, "stages": {**record["stages"], "pool": new_pool}}

    with ThreadPoolExecutor(args.workers) as executor:
        rescored = {r["id"]: r for r in executor.map(rescore, targets)}
    variants = {
        "baseline": [runner.expand(r) for r in chosen],
        "parts": [runner.expand(rescored.get(r["id"], r)) for r in chosen],
    }
    rows = []
    for name, recs in variants.items():
        for cap_name, cap in (("type_cap", None), ("multi_part_10", lambda r: max(10, replay.profile_for(r["profile"]).max_passages)
                                                   if r["id"] in rescored else replay.profile_for(r["profile"]).max_passages)):
            v = replay.Variant(f"{name}+{cap_name}", max_keep=cap)
            out = replay.evaluate(recs, queries, settings, v)
            out["pool_rerank_recall@10"] = metrics.summarize(recs, settings, stages=("reranked",))["reranked"]["recall@10"]
            rows.append(out)
    print("\n| 打分 + 上限 | 切后 R@all | 成功率 | MRR | nDCG@10 | 条数 | 保留精度 | 切前 R@10 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['variant']} | {r['recall@all']:.3f} | {r['success@all']:.3f} | {r['mrr']:.3f} | {r['ndcg@10']:.3f} | "
              f"{r['items']:.2f} | {r['kept_precision']:.3f} | {r['pool_rerank_recall@10']:.3f} |")
    print(f"\nlive rerank calls: {reranker.live.calls}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"decompose_{args.dataset}.json").write_text(json.dumps({"questions": len(chosen), "split": len(targets), "rows": rows},
                                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    cache.close()


if __name__ == "__main__":
    main()
