"""Tables and loss attribution from runner.py records.

For every run: metrics per question type and per tag, before the cut (the pool in rerank
order) and after it (what the retriever returns), the negatives, the per-stage latency,
and for every gold document an answerable question did not get, where it was lost:

* ``not_retrieved``   in neither route's top list (dense KNN, BM25): the passage text or
                      its representation (chunking, tokenization, embedding);
* ``rrf_cut``         in a route's top list but not in the fused pool sent to the reranker;
* ``below_threshold`` in the pool, but its rerank score is under the cutoff
                      (max(rerank_min_score, rerank_relative_score x best));
* ``type_cap``        above the cutoff but beyond the question type's ``max_keep``;
* ``per_doc_cap``     kept out by the per-document cap (with no cap it would be kept);
* ``context_budget``  returned, but its parent did not fit the question type's budget.

Version filtering (``drop_superseded``) does not apply: the isolated indexes have no
file registry, so the retriever has no file versions to compare.

Usage:
  python -I evaluation/suite/report.py crud_rag:base duretrieval:base ... [--latency latency]
Writes evaluation/suite_v1/results/<dataset>_<run>.json and prints Markdown tables.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out

import common  # noqa: E402
from common import RESULTS  # noqa: E402
import metrics  # noqa: E402
import runner  # noqa: E402

from retrieval.selection import SelectionPolicy, select_hits  # noqa: E402
from retrieval.types import SearchHit  # noqa: E402

LOSSES = ("not_retrieved", "rrf_cut", "below_threshold", "type_cap", "per_doc_cap", "context_budget")
LOSS_NAMES = {"not_retrieved": "两路都没召回", "rrf_cut": "RRF 融合截断", "below_threshold": "低于重排阈值",
              "type_cap": "题型条数上限", "per_doc_cap": "单文档上限", "context_budget": "上下文预算"}


def defective(dataset: str) -> set[str]:
    """Questions the dataset itself got wrong (adapters.DEFECT): left out of every metric."""
    return {q["id"] for q in runner.load_dataset(dataset).questions if "dataset_defect" in q["tags"]}


def load(dataset: str, run_name: str) -> tuple[list[dict], dict]:
    folder = runner.runs_dir(dataset)
    meta = json.loads((folder / f"{run_name}.meta.json").read_text(encoding="utf-8"))
    skip = defective(dataset)
    return [r for r in runner.load_records(folder / f"{run_name}.jsonl") if r["id"] not in skip], meta


def policy_of(meta: dict) -> dict:
    return meta["retrieval_settings"]


# ---------------------------------------------------------------- attribution
def _first(hits: list[dict], gold: int) -> int | None:
    return next((rank for rank, hit in enumerate(hits, start=1) if gold in (hit.get("g") or ())), None)


def attribute(record: dict, settings: dict) -> list[dict]:
    """One entry per gold the final hits miss (and per gold returned but not in the context)."""
    if not record["answerable"]:
        return []
    stages = record["stages"]
    out = []
    final_golds = {g for hit in stages["final"] for g in hit.get("g") or ()}
    context_golds = {g for hit in stages.get("context", []) for g in hit.get("g") or ()}
    reranked = stages["reranked"]
    for gold in range(record["n_gold"]):
        info = {"gold": gold, "dense_rank": _first(stages["dense"], gold), "bm25_rank": _first(stages["bm25"], gold),
                "pool_rank": _first(stages["pool"], gold), "rerank_rank": _first(reranked, gold)}
        if gold in final_golds:
            if stages.get("context") is not None and gold not in context_golds:
                out.append({**info, "loss": "context_budget"})
            continue
        if info["pool_rank"] is None:
            loss = "rrf_cut" if (info["dense_rank"] or info["bm25_rank"]) else "not_retrieved"
        else:
            scores = [hit.get("r") or 0.0 for hit in reranked]
            best = reranked[info["rerank_rank"] - 1].get("r") or 0.0
            info["score"] = round(best, 4)
            info["top_score"] = round(max(scores), 4)
            cutoff = max(settings["rerank_min_score"], settings["rerank_relative_score"] * max(scores))
            if best < cutoff:
                loss = "below_threshold"
            else:
                cap = min(record["max_keep"], settings["rerank_top_k"])
                kept = _replay(reranked, SelectionPolicy(settings["rerank_min_score"], settings["rerank_relative_score"],
                                                         min(settings["rerank_min_keep"], cap), cap, 0))
                loss = "per_doc_cap" if any(gold in (h.get("g") or ()) for h in kept) else "type_cap"
        out.append({**info, "loss": loss})
    return out


def _replay(reranked: list[dict], policy: SelectionPolicy) -> list[dict]:
    hits = []
    for row in reranked:
        hit = SearchHit(id=row["c"], score=0.0, text="", doc_id=row["d"], parent_id=row["p"], source="", heading="",
                        locator="", retrieval_method="replay", rerank_score=row.get("r") or 0.0)
        hit.row = row  # type: ignore[attr-defined]
        hits.append(hit)
    return [h.row for h in select_hits(hits, policy)]  # type: ignore[attr-defined]


# ---------------------------------------------------------------- summaries
def latency_table(records: list[dict], latency_records: list[dict] | None, embed: dict | None = None) -> dict:
    """Local stages from the single-worker offline re-run when there is one (else the live
    run); the rerank call from the live run (the successful attempt only: our client-side
    rate limiting and 429 retries are not part of the app's latency); the query embedding
    from embed_latency.py when measured (else the live run's calls that went out).

    ``total``: per question, local stages + its rerank call + the embedding P50.
    """
    local = {row["id"]: row for row in (latency_records or records)}
    stages: dict[str, list[float]] = defaultdict(list)
    for row in local.values():
        for stage in ("dense", "bm25", "fuse", "search", "select", "context"):
            stages[stage].append(row["latency_ms"][stage])
    for row in records:
        live = row.get("live") or {}
        if live.get("embed_calls") and not embed:
            stages["embed_network"].append(live["embed_ms"])
        if live.get("rerank_calls"):
            stages["rerank_network"].append(live["rerank_ms"])
            stages["rerank_docs"].append(live["rerank_docs"])
    out = {stage: {"p50": metrics.percentile(v, 50), "p90": metrics.percentile(v, 90), "p99": metrics.percentile(v, 99),
                   "mean": round(statistics.fmean(v), 2), "n": len(v)} for stage, v in stages.items() if v}
    if embed:
        out["embed_network"] = embed
    embed_p50 = out.get("embed_network", {}).get("p50", 0.0)
    totals = []
    for row in records:
        live = row.get("live") or {}
        own = local.get(row["id"], row)["latency_ms"]
        if live.get("rerank_calls"):
            totals.append(embed_p50 + live["rerank_ms"] + own["search"] + own["select"] + own["context"])
    if totals:
        out["total_live"] = {"p50": metrics.percentile(totals, 50), "p90": metrics.percentile(totals, 90),
                             "p99": metrics.percentile(totals, 99), "mean": round(statistics.fmean(totals), 2), "n": len(totals)}
    return out


def summarize_run(dataset: str, run_name: str, latency_tag: str = "") -> dict:
    records, meta = load(dataset, run_name)
    settings = policy_of(meta)
    latency_records = None
    if latency_tag:
        path = runner.runs_dir(dataset) / f"{run_name}_{latency_tag}.jsonl"
        if path.is_file():
            latency_records = runner.load_records(path)
    embed_path = RESULTS / "embed_latency.json"
    embed = json.loads(embed_path.read_text(encoding="utf-8")).get(dataset) if embed_path.is_file() else None
    out = {"dataset": dataset, "run": run_name, "provenance": meta, "excluded_defective": len(defective(dataset)),
           "overall": metrics.summarize(records, settings)}
    out["by_type"] = {k: metrics.summarize(v, settings) for k, v in sorted(metrics.group_by(records, metrics.by_type).items())}
    out["by_tag"] = {k: metrics.summarize(v, settings) for k, v in sorted(metrics.group_by(records, metrics.by_tag).items())}
    out["by_profile"] = {k: metrics.summarize(v, settings) for k, v in sorted(metrics.group_by(records, lambda r: [r["profile"]]).items())}
    out["latency"] = latency_table(records, latency_records, embed)
    losses = {r["id"]: attribute(r, settings) for r in records}
    out["losses"] = {
        "overall": dict(Counter(item["loss"] for items in losses.values() for item in items)),
        "by_type": {t: dict(Counter(item["loss"] for r in rs for item in losses[r["id"]]))
                    for t, rs in metrics.group_by(records, metrics.by_type).items()},
        "golds_answerable": sum(r["n_gold"] for r in records if r["answerable"]),
    }
    # Where the app would send each question first (with its file search service).
    queries = {q["id"]: q["query"] for q in runner.load_dataset(dataset).questions}
    routes = {r["id"]: runner.route_only(queries[r["id"]], file_search=True) for r in records}
    out["routes"] = dict(Counter(routes.values()))
    out["routes_by_type"] = {t: dict(Counter(routes[r["id"]] for r in rs))
                             for t, rs in metrics.group_by(records, metrics.by_type).items()}
    out["routes_by_tag"] = {t: dict(Counter(routes[r["id"]] for r in rs))
                            for t, rs in metrics.group_by(records, metrics.by_tag).items()}
    out["profiles"] = dict(Counter(r["profile"] for r in records))
    spans = [r for r in records if r["answerable"] and r.get("n_spans")]
    if spans:
        def span_recall(rows, stage):
            return statistics.fmean(len({t for h in r["stages"][stage] for t in h.get("t") or ()}) / r["n_spans"] for r in rows)
        out["evidence_sentences"] = {"questions": len(spans), "final": span_recall(spans, "final"),
                                     "context": span_recall(spans, "context")}
    return out


def write(summary: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{summary['dataset']}_{summary['run']}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------- Markdown
def f3(value) -> str:
    return "–" if value is None else f"{value:.3f}"


def quality_rows(groups: dict[str, dict], k: int = 5, labels: dict[str, str] | None = None) -> list[str]:
    """Before the cut: the whole pool in rerank order (R@all = pool recall). After: the kept
    passages (R = recall of what is kept, P = share of kept passages that are relevant,
    success = every gold for multi-document questions, any gold otherwise)."""
    lines = [f"| 分组 | 可答题数 | 切前 R@{k} | 切前 R@10 | 切前 R@all（池） | 切前 P@{k} | 切前 MRR | 切前 nDCG@10 | 切后 R | 切后 P | 切后 MRR | 切后 nDCG@10 | 切后成功 | 切后条数 | 上下文召回 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in groups.items():
        if not s.get("answerable"):
            continue
        before, after, ctx = s.get("reranked", {}), s.get("final", {}), s.get("context", {})
        lines.append(
            f"| {(labels or {}).get(name, name)} | {s['answerable']} | {f3(before.get(f'recall@{k}'))} | {f3(before.get('recall@10'))} | "
            f"{f3(before.get('recall@all'))} | {f3(before.get(f'precision@{k}'))} | {f3(before.get('mrr'))} | {f3(before.get('ndcg@10'))} | "
            f"{f3(after.get('recall@all'))} | {f3(after.get('kept_precision'))} | {f3(after.get('mrr'))} | "
            f"{f3(after.get('ndcg@10'))} | {f3(after.get('success@all'))} | {after.get('items', 0):.1f} | {f3(ctx.get('recall@all'))} |")
    return lines


def negative_rows(groups: dict[str, dict]) -> list[str]:
    lines = ["| 分组 | 负例题数 | 切后返回条数 | 其中独立过阈值 | ≥1 条过阈值 | 进上下文段数 | 最高分 P50 / P90 | AUC（可答 vs 负例 最高分） |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in groups.items():
        neg = s.get("negative")
        if not neg:
            continue
        lines.append(f"| {name} | {s['negatives']} | {neg['returned']:.2f} | {neg['above_cut']:.2f} | {neg['any_above_cut']:.0%} | "
                     f"{neg['context']:.2f} | {neg['top_score_p50']:.3f} / {neg['top_score_p90']:.3f} | {f3(neg['auc_top_score'])} |")
    return lines


def latency_rows(table: dict) -> list[str]:
    names = {"embed_network": "查询向量（网络）", "dense": "向量检索", "bm25": "BM25（含分词）", "fuse": "RRF 融合",
             "search": "检索合计", "rerank_network": "重排（网络）", "select": "阈值/截断", "context": "父块扩展",
             "total_live": "合计（单题，估计）"}
    lines = ["| 阶段 | P50 ms | P90 ms | P99 ms | 次数 |", "|---|---:|---:|---:|---:|"]
    for key, label in names.items():
        if key in table:
            row = table[key]
            lines.append(f"| {label} | {row['p50']:.1f} | {row['p90']:.1f} | {row['p99']:.1f} | {row['n']} |")
    return lines


def loss_rows(summary: dict) -> list[str]:
    lines = ["| 类型 | " + " | ".join(LOSS_NAMES[l] for l in LOSSES) + " |", "|---|" + "---:|" * len(LOSSES)]
    for name, counts in sorted(summary["losses"]["by_type"].items()):
        if counts:
            lines.append(f"| {name} | " + " | ".join(str(counts.get(l, 0)) for l in LOSSES) + " |")
    total = summary["losses"]["overall"]
    lines.append("| 合计 | " + " | ".join(str(total.get(l, 0)) for l in LOSSES) + " |")
    return lines


def overview_rows(summaries: list[dict]) -> list[str]:
    """One row per dataset: the headline numbers before and after the cut, negatives, latency."""
    lines = ["| 集合 | 可答 / 负例 | 切前 R@5 | 切前 MRR | 切前 nDCG@10 | 池召回 | 切后 R | 切后 P | 切后成功 | 上下文召回 | 负例返回条数 | 负例 ≥1 条过阈值 | AUC | 检索本地 P50 ms | 重排 P50 ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        o = s["overall"]
        before, after, ctx, neg = o.get("reranked", {}), o.get("final", {}), o.get("context", {}), o.get("negative") or {}
        lat = s["latency"]
        local = sum(lat.get(stage, {}).get("p50", 0) for stage in ("search", "select", "context"))
        lines.append(
            f"| {s['dataset']} | {o['answerable']} / {o['negatives']} | {f3(before.get('recall@5'))} | {f3(before.get('mrr'))} | "
            f"{f3(before.get('ndcg@10'))} | {f3(before.get('recall@all'))} | {f3(after.get('recall@all'))} | {f3(after.get('kept_precision'))} | "
            f"{f3(after.get('success@all'))} | {f3(ctx.get('recall@all'))} | "
            f"{neg.get('returned', float('nan')):.2f} | {neg.get('any_above_cut', float('nan')):.0%} | {f3(neg.get('auc_top_score'))} | "
            f"{local:.0f} | {lat.get('rerank_network', {}).get('p50', float('nan')):.0f} |")
    return lines


def markdown(summary: dict) -> str:
    out = [f"### {summary['dataset']} / {summary['run']}", ""]
    out += ["按题型：", ""] + quality_rows({"全部": summary["overall"], **summary["by_type"]}) + [""]
    if summary["by_tag"]:
        out += ["按标签：", ""] + quality_rows(summary["by_tag"]) + [""]
    neg = negative_rows({"全部": summary["overall"], **summary["by_type"]})
    if len(neg) > 2:
        out += ["负例：", ""] + neg + [""]
    out += ["丢失位置（缺失的金标文档数）：", ""] + loss_rows(summary) + [""]
    out += ["耗时：", ""] + latency_rows(summary["latency"]) + [""]
    return "\n".join(out)


def compare(dataset: str, base: str, other: str) -> dict:
    """Two runs on the questions they share (e.g. the raw query vs the planner's rewrite)."""
    a, meta = load(dataset, base)
    b, _ = load(dataset, other)
    ids = {r["id"] for r in b} & {r["id"] for r in a}
    a = [r for r in a if r["id"] in ids]
    b = [r for r in b if r["id"] in ids]
    settings = policy_of(meta)
    out = {"questions": len(ids)}
    for name, recs in ((base, a), (other, b)):
        s = metrics.summarize(recs, settings, stages=("pool", "reranked", "final"))
        out[name] = {"pool_recall": s.get("pool", {}).get("recall@all"), "before_mrr": s.get("reranked", {}).get("mrr"),
                     "before_ndcg@10": s.get("reranked", {}).get("ndcg@10"), "after_recall": s.get("final", {}).get("recall@all"),
                     "after_success": s.get("final", {}).get("success@all"), "answerable": s["answerable"]}
    by_id = {r["id"]: r for r in a}
    out["rewritten"] = sum(1 for r in b if r.get("query_used"))
    out["intents"] = dict(Counter((r.get("route") or {}).get("intent", "?") for r in b))
    good = lambda r: bool({g for h in r["stages"]["final"] for g in h.get("g") or ()})  # noqa: E731
    out["fixed"] = sum(1 for r in b if r["answerable"] and good(r) and not good(by_id[r["id"]]))
    out["broken"] = sum(1 for r in b if r["answerable"] and not good(r) and good(by_id[r["id"]]))
    planner_ms = [(r.get("route") or {}).get("ms") for r in b if (r.get("route") or {}).get("ms")]
    out["planner_ms_p50"] = metrics.percentile(planner_ms, 50) if planner_ms else None
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="dataset:run")
    parser.add_argument("--latency", default="", help="tag of the single-worker offline re-run")
    parser.add_argument("--compare", default="", help="another run name: compare on the shared questions")
    args = parser.parse_args()
    if args.compare:
        for spec in args.runs:
            dataset, _, run_name = spec.partition(":")
            result = compare(dataset, run_name or "base", args.compare)
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / f"{dataset}_{run_name or 'base'}_vs_{args.compare}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
            print(dataset, json.dumps(result, ensure_ascii=False))
        return
    texts, summaries = [], []
    for spec in args.runs:
        dataset, _, run_name = spec.partition(":")
        summary = summarize_run(dataset, run_name or "base", args.latency)
        write(summary)
        summaries.append(summary)
        texts.append(markdown(summary))
    text = "\n".join(["### 总览", ""] + overview_rows(summaries) + [""] + texts)
    (RESULTS / "tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
