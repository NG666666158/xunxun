"""Live retrieval time of survey questions, main's defaults against the given settings.

  python evaluation/survey_recall_v2/latency.py --name files [--set field=value ...]

12 survey questions (8 of the tuning set, 4 held out), each asked once per configuration,
alternating which goes first; an empty model-output cache per configuration, so every
embedding and rerank request is live. Reports the median and slowest wall time of the
retriever, the passages scored and the estimated rerank tokens per question.
Results: results/latency_<name>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shared  # noqa: E402
from shared import harness  # noqa: E402
from cache import Cache  # noqa: E402

from rag_agent.answering import classify_question, retrieval_limits  # noqa: E402
from retrieval import multipart  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--pause", type=float, default=4.0)
    parser.add_argument("--into", default="", help="another round's folder for results/ (survey_recall_v3)")
    parser.add_argument("--extra", default="", help="a .jsonl of more questions: every third, at most four, join the sample")
    args = parser.parse_args()
    after = harness.parse_overrides(args.overrides)
    sample = shared.surveys()[::3][:8] + shared.read_rows(shared.HERE / "holdout.jsonl")[::4][:4]
    if args.extra:
        sample += shared.read_rows(Path(args.extra))[::3][:4]
    temp = Path(tempfile.mkdtemp(prefix="sv2-latency-"))
    opened = {}
    for name, overrides in (("before", {}), ("after", after)):
        harness.cache = lambda name=name: Cache(temp / f"{name}.sqlite3")  # empty: every call is live
        opened[name] = harness.open_retriever("base", **overrides)
    rows: dict[str, list[dict]] = {"before": [], "after": []}
    for number, question in enumerate(sample):
        for name in (("before", "after") if number % 2 == 0 else ("after", "before")):
            retriever, recorder, counter, store, cache = opened[name]
            wide = bool(getattr(retriever.settings, "survey_wide", False))
            profile = classify_question(question["question"], True, wide, retriever.survey_places() if wide else frozenset())
            started = time.perf_counter()
            retriever.retrieve(question["question"], mode="hybrid_rerank", **retrieval_limits(profile))
            seconds = time.perf_counter() - started
            calls = multipart.last_trace().rerank_calls
            rows[name].append({"id": question["id"], "seconds": round(seconds, 3), "requests": len(calls),
                               "scored": sum(c["docs"] for c in calls), "tokens": sum(c["tokens"] for c in calls)})
            time.sleep(args.pause)
    out: dict = {"name": args.name, "after": after, "questions": len(sample)}
    for name, items in rows.items():
        out[name] = {"median_s": round(statistics.median(r["seconds"] for r in items), 2),
                     "max_s": round(max(r["seconds"] for r in items), 2),
                     "rerank_requests": round(statistics.fmean(r["requests"] for r in items), 1),
                     "scored_mean": round(statistics.fmean(r["scored"] for r in items), 1),
                     "rerank_tokens_mean": round(statistics.fmean(r["tokens"] for r in items))}
    out["rows"] = rows
    shared.write_json((Path(args.into) / "results" if args.into else shared.RESULTS) / f"latency_{args.name}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
