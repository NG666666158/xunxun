"""Before / after the three retrieval fixes of suite_v1 (REPORT.md, section 5), as one table.

* **Routing** (fix 1): where the app sends each question first, with its file search service
  (``runner.route_only(file_search=True)``), under the find-file cue of the base commit and
  under the current one. A question sent to file search never reaches passage retrieval, so
  "routed success" counts it as a miss; "success" is the retriever's alone, as before.
* **Selection** (fixes 2 and 3): two runner.py runs of the same questions, ``base`` (the
  report's) and the run after the fixes, both on the cached vectors and rerank scores.
* **file_search_v1**: the find-file cue of its 32 questions before and after (file search
  itself does not read the cue; only the route to it does).

Usage:
  python -I evaluation/suite/fixes_compare.py --after fixes [--base-commit 64ac7dd] [--out PATH]
Prints Markdown; writes evaluation/suite_v1/results/fixes_compare.json (or --out).
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out

import common  # noqa: E402
from common import RESULTS, ROOT  # noqa: E402
import metrics  # noqa: E402
import report  # noqa: E402
import runner  # noqa: E402

from rag_agent.nodes import HISTORY_REFERENCE, SMALL_TALK, AgentNodes  # noqa: E402
from retrieval import file_filters  # noqa: E402

SETS = ("crud_rag", "duretrieval", "t2retrieval", "multihop_rag", "books", "complex", "benchmark")
# Question types shown on their own (the ones the fixes are about).
TYPES = {"crud_rag": ("multi_2doc", "multi_3doc"), "multihop_rag": ("inference", "comparison", "temporal")}


def old_cue(commit: str):
    """``find_file_cue`` of ``commit`` (its file_filters.py, loaded under another name)."""
    source = subprocess.run(["git", "show", f"{commit}:project/retrieval/file_filters.py"], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", check=True).stdout
    path = Path(tempfile.mkdtemp(prefix="old-cue-")) / "old_file_filters.py"
    path.write_text(source, encoding="utf-8")
    # Inside the retrieval package, so that "from .doc_cards import …" (since doc_cards_v1)
    # resolves; its helpers are today's.
    name = "retrieval.old_file_filters"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while it loads
    spec.loader.exec_module(module)
    return module.find_file_cue


def route(question: str, cue) -> str:
    """runner.route_only(file_search=True) with a given cue function."""
    if SMALL_TALK.match(question.strip()):
        return "chitchat"
    if not HISTORY_REFERENCE.search(question) and cue(question) == "strong":
        return "find_file"
    return "raw" if AgentNodes._plain_knowledge_question(question) else "planner"


def summary(records: list[dict], settings: dict, routes: dict[str, str]) -> dict:
    s = metrics.summarize(records, settings, stages=("final", "context"))
    answerable = [r for r in records if r["answerable"]]
    per = {r["id"]: metrics.question_metrics(r, settings, stages=("final", "context")) for r in answerable}
    success = {i: per[i]["final"]["success@all"] for i in per}
    out = {
        "answerable": len(answerable),
        "success": s.get("final", {}).get("success@all"),
        "routed_success": statistics.fmean(success[i] * (routes[i] != "find_file") for i in success) if success else None,
        "context_recall": s.get("context", {}).get("recall@all"),
        "context_success": s.get("context", {}).get("success@all"),
        "items": s.get("final", {}).get("items"),
        "to_file_search": sum(routes[r["id"]] == "find_file" for r in records),
        "questions": len(records),
        "profiles": dict(Counter(r["profile"] for r in records)),
    }
    if records and records[0]["unit"] == "passage":
        out["strict_hit@5"] = s.get("final", {}).get("success@5")
    negatives = [r for r in records if not r["answerable"]]
    if negatives:
        out["negatives"] = len(negatives)
        out["neg_returned"] = statistics.fmean(len(r["stages"]["final"]) for r in negatives)
        out["neg_empty"] = statistics.fmean(not r["stages"]["final"] for r in negatives)
        out["neg_context"] = statistics.fmean(len(r["stages"].get("context") or []) for r in negatives)
    return out


def compare_set(dataset: str, after: str, before_cue, base: str = "base") -> dict:
    a, meta_a = report.load(dataset, base)
    b, meta_b = report.load(dataset, after)
    ids = {r["id"] for r in a} & {r["id"] for r in b}
    a = [r for r in a if r["id"] in ids]
    b = [r for r in b if r["id"] in ids]
    queries = {q["id"]: q["query"] for q in runner.load_dataset(dataset).questions}
    routes_before = {i: route(queries[i], before_cue) for i in ids}
    routes_after = {i: route(queries[i], file_filters.find_file_cue) for i in ids}
    out = {"before": summary(a, meta_a["retrieval_settings"], routes_before),
           "after": summary(b, meta_b["retrieval_settings"], routes_after),
           "after_settings": {k: meta_b["retrieval_settings"].get(k) for k in ("rerank_per_document", "rerank_no_evidence_below")},
           "after_commit": meta_b["git"]}
    by_type = {}
    for kind in TYPES.get(dataset, ()):
        rows = {}
        for name, recs, meta, routes in (("before", a, meta_a, routes_before), ("after", b, meta_b, routes_after)):
            sub = [r for r in recs if r["type"] == kind]
            rows[name] = summary(sub, meta["retrieval_settings"], routes)
        by_type[kind] = rows
    out["by_type"] = by_type
    # Answerable questions whose retrieval got worse / better (any gold kept, or all for needs_all).
    good = lambda r, s: metrics.question_metrics(r, s, stages=("final",))["final"]["success@all"]  # noqa: E731
    before_by = {r["id"]: r for r in a}
    changed = Counter()
    for r in b:
        if not r["answerable"]:
            continue
        x, y = good(before_by[r["id"]], meta_a["retrieval_settings"]), good(r, meta_b["retrieval_settings"])
        if y > x:
            changed["fixed"] += 1
        elif y < x:
            changed["broken"] += 1
            changed["broken_no_evidence" if not r["stages"]["final"] else "broken_other"] += 1
    out["changed"] = dict(changed)
    return out


def file_search_set(before_cue) -> dict:
    rows = []
    for line in (ROOT / "evaluation" / "file_search_v1" / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            rows.append({"id": q["id"], "before": before_cue(q["question"]), "after": file_filters.find_file_cue(q["question"]),
                         "route_before": route(q["question"], before_cue),
                         "route_after": route(q["question"], file_filters.find_file_cue)})
    return {"questions": len(rows), "strong_before": sum(r["before"] == "strong" for r in rows),
            "strong_after": sum(r["after"] == "strong" for r in rows),
            "same_cue": sum(r["before"] == r["after"] for r in rows),
            "same_route": sum(r["route_before"] == r["route_after"] for r in rows), "rows": rows}


def f3(value) -> str:
    return "–" if value is None else f"{value:.3f}"


def markdown(result: dict) -> str:
    lines = ["| 集合 | 可答 | 送去找文件（前 → 后） | 检索成功（前 → 后） | 计入路由的成功（前 → 后） | 进上下文召回（前 → 后） | 每题条数（前 → 后） | 负例空结果（前 → 后） | 多部分题 | 变好 / 变差 |",
             "|---|---:|---|---|---|---|---|---|---:|---|"]
    for dataset in SETS:
        if dataset not in result:
            continue
        r = result[dataset]
        x, y = r["before"], r["after"]
        hit = (f"；严格命中@5 {f3(x.get('strict_hit@5'))} → {f3(y.get('strict_hit@5'))}" if "strict_hit@5" in x else "")
        neg = f"{x['neg_empty']:.0%} → {y['neg_empty']:.0%}" if "negatives" in x else "–"
        changed = r["changed"]
        lines.append(
            f"| {dataset} | {x['answerable']} | {x['to_file_search']} → {y['to_file_search']} | {f3(x['success'])} → {f3(y['success'])}{hit} | "
            f"{f3(x['routed_success'])} → {f3(y['routed_success'])} | {f3(x['context_recall'])} → {f3(y['context_recall'])} | "
            f"{x['items']:.2f} → {y['items']:.2f} | {neg} | {y['profiles'].get('multi', 0)} | "
            f"{changed.get('fixed', 0)} / {changed.get('broken', 0)} |")
    out = ["\n".join(lines), ""]
    for dataset in SETS:
        for kind, rows in (result.get(dataset, {}).get("by_type") or {}).items():
            x, y = rows["before"], rows["after"]
            out.append(f"- {dataset} / {kind}（{x['answerable']} 题）：检索成功 {f3(x['success'])} → {f3(y['success'])}，"
                       f"计入路由 {f3(x['routed_success'])} → {f3(y['routed_success'])}，每题条数 {x['items']:.2f} → {y['items']:.2f}")
    fs = result.get("file_search_v1")
    if fs:
        out += ["", f"file_search_v1（{fs['questions']} 题）：强信号 {fs['strong_before']} → {fs['strong_after']}；"
                    f"找文件信号相同 {fs['same_cue']} / {fs['questions']}；首轮路由相同 {fs['same_route']} / {fs['questions']}"]
    return "\n".join(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--after", default="fixes")
    parser.add_argument("--base", default="base")
    parser.add_argument("--base-commit", default="64ac7dd", help="the commit whose find-file cue is the 'before'")
    parser.add_argument("--sets", default=",".join(SETS))
    parser.add_argument("--out", type=Path, default=RESULTS / "fixes_compare.json")
    args = parser.parse_args()
    before_cue = old_cue(args.base_commit)
    result = {dataset: compare_set(dataset, args.after, before_cue, args.base) for dataset in args.sets.split(",")}
    result["file_search_v1"] = file_search_set(before_cue)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
    print(markdown(result))


if __name__ == "__main__":
    main()
