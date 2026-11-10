"""Replay the retriever's selection on recorded pools: what-if for cutoffs and caps.

Uses the app's own ``select_hits`` / ``diversify`` on the rerank scores runner.py
recorded, so no model call is needed. The baseline replay must reproduce the recorded
final hits exactly (checked). A variant changes one of:

* ``min_score`` / ``relative`` / ``min_keep``: the score cutoffs and the floor count;
* ``max_keep``: the cap of each question type ("fact" / "explain" / "compare") or a
  function of the record;
* ``per_document``: the per-document cap;
* ``empty_below``: keep nothing when the best score is under this value (instead of
  ``min_keep`` passages);
* ``pool``: only the first N candidates of the RRF pool are reranked.

Fusion what-ifs (``fusion``): re-fuse the recorded dense and BM25 top lists with weights
and report how many golds the new pool holds (no rerank: new candidates have no score).

Usage:
  python -I evaluation/suite/replay.py crud_rag:base duretrieval:base ...
Prints a table per dataset; writes evaluation/suite_v1/results/replay.json.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out

from common import RESULTS  # noqa: E402
import metrics  # noqa: E402
import report  # noqa: E402
import runner  # noqa: E402

from rag_agent.answering import profile_for  # noqa: E402
from retrieval.hybrid_retriever import comparison_topic, diversify  # noqa: E402
from retrieval.selection import SelectionPolicy, select_hits  # noqa: E402
from retrieval.types import SearchHit  # noqa: E402


@dataclass
class Variant:
    name: str
    min_score: float | None = None
    relative: float | None = None
    min_keep: int | None = None
    per_document: int | None = None
    max_keep: Callable[[dict], int] | None = None
    empty_below: float | None = None
    pool: int | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _hits(rows: list[dict]) -> list[SearchHit]:
    out = []
    for row in rows:
        hit = SearchHit(id=row["c"], score=0.0, text="", doc_id=row["d"], parent_id=row["p"], source=row["d"],
                        heading="", locator="", retrieval_method="replay", rerank_score=row.get("r") or 0.0)
        hit.row = row  # type: ignore[attr-defined]
        out.append(hit)
    return out


def select(record: dict, query: str, settings: dict, variant: Variant) -> list[dict]:
    pool = record["stages"]["pool"]
    if variant.pool:
        pool = pool[: variant.pool]
    order = sorted(range(len(pool)), key=lambda i: (-(pool[i].get("r") or 0.0), i))
    ranked = _hits([pool[i] for i in order])
    if not ranked:
        return []
    # The run's own floor (settings recorded before it existed have none), unless the variant sets one.
    floor = settings.get("rerank_no_evidence_below", 0.0) if variant.empty_below is None else variant.empty_below
    if floor and (ranked[0].rerank_score or 0.0) < floor:
        return []
    profile = profile_for(record["profile"])
    max_keep = variant.max_keep(record) if variant.max_keep else profile.max_passages
    cap = max(1, min(max_keep, settings["rerank_top_k"] if not variant.max_keep else max(max_keep, settings["rerank_top_k"])))
    own_per_document = settings["rerank_per_document"] if profile.per_document is None else profile.per_document
    policy = SelectionPolicy(
        min_score=settings["rerank_min_score"] if variant.min_score is None else variant.min_score,
        relative=settings["rerank_relative_score"] if variant.relative is None else variant.relative,
        min_keep=min(settings["rerank_min_keep"] if variant.min_keep is None else variant.min_keep, cap),
        max_keep=cap,
        per_document=own_per_document if variant.per_document is None else variant.per_document,
    )
    kept = select_hits(ranked, policy)
    if comparison_topic(query) is not None:
        k = min(cap, max(len(kept), 2))
        kept = diversify(ranked, k, [])
    return [h.row for h in kept]  # type: ignore[attr-defined]


def evaluate(records: list[dict], queries: dict[str, str], settings: dict, variant: Variant) -> dict:
    replayed = []
    for record in records:
        final = select(record, queries[record["id"]], settings, variant)
        replayed.append({**record, "stages": {**record["stages"], "final": final}})
    s = metrics.summarize(replayed, settings, stages=("final",))
    out = {"variant": variant.name, "notes": variant.notes}
    if "final" in s:
        out.update({k: s["final"][k] for k in ("recall@5", "recall@all", "success@all", "mrr", "ndcg@10", "precision@5",
                                                "kept_precision", "items")})
    if "negative" in s:
        out["neg_returned"] = s["negative"]["returned"]
        out["neg_empty"] = statistics.fmean(len(r["stages"]["final"]) == 0 for r in replayed if not r["answerable"])
    return out


def fusion(records: list[dict], weights: tuple[float, float], k: int = 60, limit: int = 40) -> float:
    """Gold recall of an RRF pool re-fused from the recorded route lists with (dense, bm25) weights."""
    found, total = 0, 0
    for record in records:
        if not record["answerable"]:
            continue
        scores: dict[str, float] = {}
        golds: dict[str, list[int]] = {}
        for weight, hits in zip(weights, (record["stages"]["dense"], record["stages"]["bm25"])):
            for rank, hit in enumerate(hits, start=1):
                scores[hit["c"]] = scores.get(hit["c"], 0.0) + weight / (k + rank)
                golds[hit["c"]] = hit.get("g") or []
        top = sorted(scores, key=lambda c: -scores[c])[:limit]
        covered = {g for c in top for g in golds[c]}
        found += len(covered)
        total += record["n_gold"]
    return found / max(1, total)


# A question asking two things at once: two interrogatives, or a joining word before a
# second ask ("…对哪些省份…，并且安徽省预计会…？"). English: two wh-/aux words.
_ASKS_ZH = re.compile(r"哪些|哪个|哪里|哪[一两几]|什么|如何|怎样|怎么|多少|几[个项条种次天年]|何种|为何|为什么|是否")
_JOIN_ZH = re.compile(r"并且|同时|此外|另外|以及|分别")
_ASKS_EN = re.compile(r"\b(?:who|what|which|when|where|how|why|did|does|do|is|was|were|are|has|have)\b", re.I)


def multi_part(query: str) -> bool:
    if re.search(r"[一-鿿]", query):
        return len(_ASKS_ZH.findall(query)) >= 2 or bool(_JOIN_ZH.search(query))
    return len(_ASKS_EN.findall(query)) >= 2


def variants(settings: dict) -> list[Variant]:
    def keep(n: int) -> Callable[[dict], int]:
        return lambda record: n

    def multi_part_cap(record: dict) -> int:
        # A question-type rule that also recognised multi-part questions by their wording.
        default = profile_for(record["profile"]).max_passages
        return max(10, default) if record.get("multi_part") else default

    return [
        Variant("baseline"),
        Variant("empty_below_0.1", empty_below=0.1, notes="最高分 < 0.1 时一条不返回（不再保底 3 条）"),
        Variant("empty_below_0.3", empty_below=0.3, notes="最高分 < 0.3 时一条不返回"),
        Variant("min_score_0.3", min_score=0.3, notes="绝对下限 0.1→0.3（仍保底 3 条）"),
        Variant("min_keep_1", min_keep=1, notes="保底 3→1 条"),
        Variant("relative_0.3", relative=0.3, notes="相对下限 0.1→0.3"),
        Variant("per_doc_2", per_document=2, notes="单文档上限 4→2"),
        Variant("per_doc_off", per_document=0, notes="取消单文档上限"),
        Variant("multi_part_10", max_keep=multi_part_cap, notes="按措辞识别出的多部分问题至少给 10 条"),
        Variant("multi_part_10_doc2", max_keep=multi_part_cap, per_document=2, notes="多部分问题至少 10 条 + 单文档上限 2"),
        Variant("fix_combined", max_keep=multi_part_cap, per_document=2, empty_below=0.1,
                notes="多部分问题至少 10 条 + 单文档上限 2 + 最高分 < 0.1 时不返回"),
        Variant("max_keep_12", max_keep=keep(12), notes="所有题都给 12 条（上限参照）"),
        Variant("max_keep_12_doc2", max_keep=keep(12), per_document=2, notes="所有题 12 条 + 单文档上限 2（上限参照）"),
        Variant("pool_20", pool=20, notes="只重排 RRF 前 20 条"),
    ]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    out = {}
    for spec in args.runs:
        dataset, _, run_name = spec.partition(":")
        records, meta = report.load(dataset, run_name or "base")
        settings = meta["retrieval_settings"]
        ds = runner.load_dataset(dataset)
        queries = {q["id"]: q["query"] for q in ds.questions}
        for record in records:  # questions run with a rewritten query keep it
            queries[record["id"]] = record.get("query_used") or queries[record["id"]]
            record["multi_part"] = multi_part(queries[record["id"]])
        # The baseline replay must give back what the retriever returned.
        mismatched = [r["id"] for r in records
                      if [h["c"] for h in select(r, queries[r["id"]], settings, Variant("baseline"))] != [h["c"] for h in r["stages"]["final"]]]
        rows = [evaluate(records, queries, settings, v) for v in variants(settings)]
        fused = {f"{w[0]}:{w[1]}": fusion(records, w) for w in ((1, 1), (1, 0), (0, 1), (1.5, 1), (1, 1.5), (2, 1))}
        out[dataset] = {"rows": rows, "replay_mismatches": len(mismatched), "fusion_pool_recall": fused}
        print(f"\n## {dataset} (replay mismatches: {len(mismatched)} of {len(records)})\n")
        print("| 变体 | 说明 | 切后 R@5 | 切后 R@all | 成功率 | MRR | nDCG@10 | 条数 | 保留精度 | 负例返回条数 | 负例空结果 |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            print(f"| {r['variant']} | {r['notes']} | {r.get('recall@5', 0):.3f} | {r.get('recall@all', 0):.3f} | {r.get('success@all', 0):.3f} | "
                  f"{r.get('mrr', 0):.3f} | {r.get('ndcg@10', 0):.3f} | {r.get('items', 0):.2f} | {r.get('kept_precision', 0):.3f} | "
                  f"{r.get('neg_returned', float('nan')):.2f} | {r.get('neg_empty', float('nan')):.0%} |")
        print("\nRRF 权重（dense:bm25）下 40 条候选池的金标召回：", json.dumps({k: round(v, 3) for k, v in fused.items()}))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "replay.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
