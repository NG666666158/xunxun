"""Retrieval metrics over runner.py records.

A record holds, per question, the ranked hits of every stage of the pipeline:

* ``dense`` / ``bm25``  each route's own ranking (before fusion);
* ``pool``              the RRF-fused candidates sent to the reranker, in RRF order;
* ``reranked``          the pool in rerank-score order: **before** the threshold cut;
* ``final``             what the retriever returns: **after** the cut (absolute and relative
                        score floors, at least ``rerank_min_keep``, the question type's cap,
                        the per-document cap; comparison questions are diversified);
* ``context``           ``final`` expanded to unique parents within the question type's
                        token budget (``AgentNodes.build_context``).

Each hit carries ``g``: the indexes of the golds it satisfies. Metrics are computed on
documents (``unit == "doc"``: hits collapsed to their first occurrence per document) or
on passages (``unit == "passage"``: books_v1 / complex_v1, where the gold is a page span).

Per question and stage: Recall@K (share of golds found), Precision@K (relevant items in
the first K / K), Hit@K (any gold), Success@K (every gold for ``needs_all`` questions,
else Hit), MRR and nDCG@10 with binary gains (an item gains only for golds not seen
earlier; the ideal ranking has min(golds, 10) relevant items).

Negatives (``answerable`` false, i.e. no gold document in the index): how many reranked
passages clear the score cutoffs on their own (``above_cut``; the app then keeps at
least ``rerank_min_keep`` regardless) and the best rerank score.
"""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any, Callable, Iterable

KS = (1, 3, 5, 10, 20)
STAGES = ("dense", "bm25", "pool", "reranked", "final", "context")


def collapse(hits: list[dict], unit: str) -> list[dict]:
    """Documents in first-occurrence order (``unit == "doc"``), or the passages as they are."""
    if unit != "doc":
        return hits
    seen: set[str] = set()
    out = []
    for hit in hits:
        if hit["d"] in seen:
            continue
        seen.add(hit["d"])
        out.append(hit)
    return out


def ranking_metrics(items: list[dict], n_gold: int, needs_all: bool, ks: Iterable[int] = KS) -> dict[str, float]:
    out: dict[str, float] = {}
    if n_gold <= 0:
        return out
    found: set[int] = set()
    first: int | None = None
    gains: list[int] = []
    cover_at: list[int] = []  # golds covered after each item
    relevant_at: list[int] = []  # relevant items among the first i
    relevant = 0
    for rank, item in enumerate(items, start=1):
        golds = set(item.get("g") or ())
        if golds and first is None:
            first = rank
        gains.append(1 if golds - found else 0)
        found |= golds
        relevant += 1 if golds else 0
        cover_at.append(len(found))
        relevant_at.append(relevant)

    def at(values: list[int], k: int) -> int:
        return values[min(k, len(values)) - 1] if values else 0

    for k in ks:
        covered = at(cover_at, k)
        out[f"recall@{k}"] = covered / n_gold
        out[f"precision@{k}"] = at(relevant_at, k) / k
        out[f"hit@{k}"] = float(covered > 0)
        out[f"success@{k}"] = float(covered >= n_gold) if needs_all else float(covered > 0)
    out["recall@all"] = (cover_at[-1] if cover_at else 0) / n_gold
    out["success@all"] = (float((cover_at[-1] if cover_at else 0) >= n_gold) if needs_all
                          else float(bool(cover_at and cover_at[-1])))
    out["mrr"] = 1.0 / first if first else 0.0
    dcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(gains[:10], start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(n_gold, 10) + 1))
    out["ndcg@10"] = dcg / ideal
    out["items"] = float(len(items))
    out["kept_precision"] = (relevant_at[-1] / len(items)) if items else 0.0
    return out


def above_cut(scores: list[float], min_score: float, relative: float) -> int:
    """Passages the score cutoffs keep on their own (select_hits before min_keep and caps)."""
    if not scores:
        return 0
    cutoff = max(min_score, relative * max(scores))
    return sum(score >= cutoff for score in scores)


def question_metrics(record: dict, policy: dict[str, float], stages: Iterable[str] = STAGES) -> dict[str, Any]:
    unit = record.get("unit", "doc")
    out: dict[str, Any] = {}
    for stage in stages:
        hits = record["stages"].get(stage)
        if hits is None:
            continue
        items = collapse(hits, unit)
        if record["answerable"]:
            out[stage] = ranking_metrics(items, record["n_gold"], record["needs_all"])
    scores = [hit["r"] for hit in record["stages"].get("reranked", []) if hit.get("r") is not None]
    out["neg"] = {
        "top_score": max(scores) if scores else 0.0,
        "above_cut": above_cut(scores, policy["rerank_min_score"], policy["rerank_relative_score"]),
        "returned": len(record["stages"].get("final", [])),
        "returned_docs": len(collapse(record["stages"].get("final", []), "doc")),
        "context": len(record["stages"].get("context", [])),
    }
    gold_scores = [hit["r"] for hit in record["stages"].get("reranked", []) if hit.get("g") and hit.get("r") is not None]
    out["neg"]["top_gold_score"] = max(gold_scores) if gold_scores else None
    return out


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return float("nan")
    values = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(values)))
    return values[min(rank, len(values)) - 1]


def auc(positives: list[float], negatives: list[float]) -> float | None:
    """P(score of a random positive > a random negative), ties counting one half."""
    if not positives or not negatives:
        return None
    negatives = sorted(negatives)
    total = 0.0
    import bisect

    for value in positives:
        below = bisect.bisect_left(negatives, value)
        equal = bisect.bisect_right(negatives, value) - below
        total += below + 0.5 * equal
    return total / (len(positives) * len(negatives))


def summarize(records: list[dict], policy: dict[str, float], stages: Iterable[str] = STAGES) -> dict[str, Any]:
    """Means over answerable questions per stage; negatives and latency over their own questions."""
    stages = tuple(stages)
    per = [(r, question_metrics(r, policy, stages)) for r in records]
    answerable = [(r, m) for r, m in per if r["answerable"]]
    negatives = [(r, m) for r, m in per if not r["answerable"]]
    out: dict[str, Any] = {"questions": len(records), "answerable": len(answerable), "negatives": len(negatives)}
    for stage in stages:
        rows = [m[stage] for _, m in answerable if stage in m]
        if rows:
            out[stage] = {key: statistics.fmean(row[key] for row in rows) for key in rows[0]}
    if negatives:
        neg = [m["neg"] for _, m in negatives]
        tops = [n["top_score"] for n in neg]
        out["negative"] = {
            "returned": statistics.fmean(n["returned"] for n in neg),
            "returned_docs": statistics.fmean(n["returned_docs"] for n in neg),
            "above_cut": statistics.fmean(n["above_cut"] for n in neg),
            "any_above_cut": statistics.fmean(n["above_cut"] > 0 for n in neg),
            "context": statistics.fmean(n["context"] for n in neg),
            "top_score_p50": percentile(tops, 50),
            "top_score_p90": percentile(tops, 90),
        }
        positives = [m["neg"]["top_score"] for _, m in answerable]
        out["negative"]["auc_top_score"] = auc(positives, tops)
        # Share of negatives a stricter floor would leave empty, and what it would cost
        # answerable questions (their best gold passage falls below it).
        for floor in (0.1, 0.3, 0.5):
            out["negative"][f"empty_at_{floor}"] = statistics.fmean(t < floor for t in tops)
            golds = [m["neg"]["top_gold_score"] for _, m in answerable]
            out["negative"][f"answerable_gold_below_{floor}"] = statistics.fmean((g or 0.0) < floor for g in golds) if golds else None
    latency: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for stage, ms in (record.get("latency_ms") or {}).items():
            if ms is not None:
                latency[stage].append(ms)
    out["latency_ms"] = {stage: {"p50": percentile(v, 50), "p90": percentile(v, 90), "p99": percentile(v, 99),
                                 "mean": statistics.fmean(v), "n": len(v)} for stage, v in sorted(latency.items())}
    return out


def group_by(records: list[dict], key: Callable[[dict], Iterable[str]]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        for name in key(record):
            groups[name].append(record)
    return dict(groups)


def by_type(record: dict) -> list[str]:
    return [record["type"]]


def by_tag(record: dict) -> list[str]:
    return list(record.get("tags") or [])
