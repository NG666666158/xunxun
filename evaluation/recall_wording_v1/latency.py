"""Live retrieval time per question, before and after: nothing cached, every model call real.

  python evaluation/recall_wording_v1/latency.py [--index base] [--set field=value ...]

A fixed sample (12 named fact questions, 6 un-named, 6 survey), each asked once with the
defaults of main and once with the given settings, alternating which goes first; an empty
model-output cache per configuration, one question at a time. Reports the median and the
slowest of the retriever's wall time (query embedding, searches, reranking), and the passages
scored. Results: results/latency.json.
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

import harness  # noqa: E402
from cache import Cache  # noqa: E402

from rag_agent.answering import classify_question, retrieval_limits  # noqa: E402

MAIN = {"wording_expansion": False, "wording_rescore": False, "candidate_per_document": 0, "survey_route": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="base")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--pause", type=float, default=3.0)
    args = parser.parse_args()
    after = harness.parse_overrides(args.overrides)
    questions = harness.load_questions()
    sample = [q for q in questions if q["type"] == "fact"][::5][:12] + [q for q in questions if q["type"] == "generic"][::5][:6]
    sample += [q for q in questions if q["type"] == "survey"][::4][:6]
    temp = Path(tempfile.mkdtemp(prefix="rw-latency-"))
    opened = {}
    for name, overrides in (("before", MAIN), ("after", after)):
        harness.cache = lambda name=name: Cache(temp / f"{name}.sqlite3")  # an empty cache: every call is live
        opened[name] = harness.open_retriever(args.index, **overrides)
    rows = {"before": [], "after": []}
    for number, question in enumerate(sample):
        for name in (("before", "after") if number % 2 == 0 else ("after", "before")):
            retriever, recorder, counter, store, cache = opened[name]
            recorder.reset()
            profile = classify_question(question["question"], bool(getattr(retriever.settings, "survey_route", False)))
            started = time.perf_counter()
            retriever.retrieve(question["question"], mode="hybrid_rerank", **retrieval_limits(profile))
            rows[name].append({"id": question["id"], "type": question["type"], "seconds": round(time.perf_counter() - started, 3),
                               "scored": recorder.inner.live.items})
            time.sleep(args.pause)
    out: dict = {"index": args.index, "after": after, "questions": len(sample)}
    for name, items in rows.items():
        scored = [b["scored"] - a["scored"] for a, b in zip([{"scored": 0}] + items, items)]
        for kind in ("all", "fact", "generic", "survey"):
            chosen = [(row, n) for row, n in zip(items, scored) if kind in ("all", row["type"])]
            out[f"{name}_{kind}"] = {"median_s": round(statistics.median(r["seconds"] for r, _ in chosen), 2),
                                     "max_s": round(max(r["seconds"] for r, _ in chosen), 2),
                                     "scored_mean": round(statistics.fmean(n for _, n in chosen), 1)}
    (harness.RESULTS / "latency.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
