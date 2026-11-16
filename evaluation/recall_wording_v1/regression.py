"""The existing retrieval regression (5,756 questions over seven prepared sets) with this
branch's code, compared question by question with a reference run.

  python evaluation/recall_wording_v1/regression.py run --name step2 [--offline] [--set field=value ...] [--datasets ...]
  python evaluation/recall_wording_v1/regression.py compare --name step2 [--reference rw_base]

The prepared sets, indexes and the model-output cache are the ones of evaluation/suite
(third-party data, downloaded locally and never committed): a copy under
work/regression/ (suite/, legacy_indexes/, tree/ = a checkout with the older evaluation
folders the runner reads). ``run`` overlays this worktree's project/ and evaluation/suite on
that tree and runs evaluation/suite/runner.py per set. Without --offline, pairs the cache
lacks are scored live (a changed pool needs them).

``compare`` counts, per set: questions whose kept passages changed, questions that keep more /
fewer gold items, and the share that keeps what it needs (every gold item when the set needs
all of them, at least one otherwise). Results: results/regression_<name>.json (counts only).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = Path(os.environ.get("RW_WORK_DIR", HERE / "work")) / "regression"
# Parents and parsed files of the prepared indexes stay where the suite built them.
SOURCE = Path(os.environ.get("RW_SUITE_SOURCE", r"D:\github\rag-eval-suite"))
SETS = ("books", "complex", "benchmark", "crud_rag", "multihop_rag", "duretrieval", "t2retrieval")
LEGACY = SETS[:3]


def overlay() -> Path:
    tree = WORK / "tree"
    ignore = shutil.ignore_patterns("__pycache__", ".env", "*.pyc", "node_modules")
    shutil.copytree(ROOT / "project", tree / "project", dirs_exist_ok=True, ignore=ignore)
    shutil.copytree(ROOT / "evaluation" / "suite", tree / "evaluation" / "suite", dirs_exist_ok=True, ignore=ignore)
    return tree


def run(name: str, datasets: list[str], offline: bool, options: list[str], workers: int) -> None:
    tree = overlay()
    env = dict(os.environ, EVAL_WORK_DIR=str(WORK / "suite"), RT_WORK_DIR=str(WORK / "legacy_indexes"), PYTHONIOENCODING="utf-8")
    if offline:
        env["SILICONFLOW_API_KEY"] = "offline-evaluation"
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        original = (SOURCE / "evaluation/retrieval_tuning_v1/work" / f"{dataset}-data" if dataset in LEGACY
                    else SOURCE / "evaluation/suite_v1/work" / dataset / "index")
        command = [sys.executable, "-I", str(tree / "evaluation/suite/runner.py"), dataset, "--name", name,
                   "--workers", str(workers), "--restart",
                   "--set-option", f"parent_store_dir={original / 'parents'}", "--set-option", f"parsed_dir={original / 'parsed'}"]
        if offline:
            command.append("--offline")
        for option in options:
            command += ["--set-option", option]
        started = time.perf_counter()
        with (logs / f"{dataset}_{name}.log").open("w", encoding="utf-8") as out:
            code = subprocess.run(command, cwd=tree, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
        record = WORK / "suite" / dataset / "runs" / f"{name}.jsonl"
        done = sum(1 for line in record.open(encoding="utf-8") if line.strip()) if record.exists() else 0
        print(json.dumps({"dataset": dataset, "name": name, "exit": code, "records": done,
                          "seconds": round(time.perf_counter() - started, 1)}), flush=True)


def rows(dataset: str, name: str) -> dict[str, dict]:
    path = WORK / "suite" / dataset / "runs" / f"{name}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return {row["id"]: row for row in (json.loads(line) for line in handle if line.strip())}


def covered(row: dict) -> int:
    return len({gold for hit in row["stages"]["final"] for gold in hit.get("g") or ()})


def enough(row: dict) -> bool:
    if not row.get("answerable", True) or not row.get("n_gold"):
        return False
    return covered(row) >= row["n_gold"] if row.get("needs_all") else covered(row) >= 1


def compare(name: str, reference: str, datasets: list[str]) -> dict:
    out = []
    for dataset in datasets:
        old, new = rows(dataset, reference), rows(dataset, name)
        ids = [i for i in old if i in new]
        signature = lambda row: [(hit.get("c"), hit.get("d")) for hit in row["stages"]["final"]]  # noqa: E731
        answerable = [i for i in ids if old[i].get("answerable", True) and old[i].get("n_gold")]
        unanswerable = [i for i in ids if i not in set(answerable)]
        tokens = lambda row: sum(call.get("tokens", 0) for call in (row.get("parts") or {}).get("rerank_calls", []))  # noqa: E731
        out.append({
            "dataset": dataset, "questions": len(ids), "missing": len(old) - len(ids),
            "changed": sum(signature(old[i]) != signature(new[i]) for i in ids),
            "more_gold": sum(covered(new[i]) > covered(old[i]) for i in answerable),
            "fewer_gold": sum(covered(new[i]) < covered(old[i]) for i in answerable),
            "enough_before": round(sum(enough(old[i]) for i in answerable) / max(1, len(answerable)), 4),
            "enough_after": round(sum(enough(new[i]) for i in answerable) / max(1, len(answerable)), 4),
            "unanswerable": len(unanswerable),
            "unanswerable_empty_before": sum(not old[i]["stages"]["final"] for i in unanswerable),
            "unanswerable_empty_after": sum(not new[i]["stages"]["final"] for i in unanswerable),
            "rerank_tokens_before": round(sum(tokens(old[i]) for i in ids) / max(1, len(ids))),
            "rerank_tokens_after": round(sum(tokens(new[i]) for i in ids) / max(1, len(ids))),
            "live_rerank_docs": sum((new[i].get("live") or {}).get("rerank_docs", 0) for i in ids),
        })
    total = {"questions": sum(r["questions"] for r in out), "changed": sum(r["changed"] for r in out),
             "more_gold": sum(r["more_gold"] for r in out), "fewer_gold": sum(r["fewer_gold"] for r in out)}
    result = {"name": name, "reference": reference, "sets": out, "total": total}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"regression_{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in out:
        print(json.dumps(row, ensure_ascii=False))
    print(json.dumps(total))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "compare"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--reference", default="rw_base")
    parser.add_argument("--datasets", nargs="+", choices=SETS, default=list(SETS))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--set", action="append", default=[], dest="options")
    args = parser.parse_args()
    if args.action == "run":
        run(args.name, args.datasets, args.offline, args.options, args.workers)
    else:
        compare(args.name, args.reference, args.datasets)


if __name__ == "__main__":
    main()
