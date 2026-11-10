"""Query-embedding latency: one live bge-m3 call per question, one at a time, no cache.

The live runs embed most queries from the cache (they were embedded in one batch first,
to keep the runs off the embedding rate limit while indexes were being built), so the
network time of the query embedding is measured here on a fixed sample per dataset.

Usage:
  python -I evaluation/suite/embed_latency.py crud_rag duretrieval t2retrieval multihop_rag [--n 100]
Writes evaluation/suite_v1/results/embed_latency.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out

import common  # noqa: E402
from common import RESULTS  # noqa: E402
import metrics  # noqa: E402
import runner  # noqa: E402

from services.model_clients import SiliconFlowEmbeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    client = SiliconFlowEmbeddings(common.settings())
    path = RESULTS / "embed_latency.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for name in args.datasets:
        questions = runner.load_dataset(name).questions
        sample = random.Random(f"latency:{name}").sample(questions, min(args.n, len(questions)))
        client._embed_batch(["warm-up"])  # connection set up, not measured
        times = []
        for q in sample:
            started = time.perf_counter()
            client._embed_batch([q["query"]])
            times.append((time.perf_counter() - started) * 1000)
        out[name] = {"n": len(times), "p50": metrics.percentile(times, 50), "p90": metrics.percentile(times, 90),
                     "p99": metrics.percentile(times, 99), "mean": sum(times) / len(times)}
        print(name, {k: round(v, 1) for k, v in out[name].items()}, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
