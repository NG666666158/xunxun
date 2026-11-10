"""The offline CI tier: the whole retrieval evaluation on a small fixed slice, without
network or keys.

fixtures/ci/ holds
* ``corpus.jsonl`` / ``questions.jsonl``: 24 short documents and 16 questions written
  for this suite (Chinese and English; single and multi-document, digits and model
  numbers, colloquial wording, unanswerable questions). No third-party text.
* ``vectors.npz`` / ``scores.json``: the bge-m3 vectors and bge-reranker-v2-m3 scores of
  exactly the texts and pairs the run needs (cache.py keys).
* ``expected.json``: the metrics of every stage and each question's kept documents.

``check`` rebuilds the index from the corpus through the ingestion pipeline (index.py),
runs every question through the retriever with the caches only (runner.py --offline),
and compares the outcome with expected.json: any change to chunking, tokenization,
fusion, the cutoffs, the caps or parent expansion that moves a result shows up here.
After an intended change, ``refresh`` (needs the SiliconFlow key) recomputes the caches
and expected.json; ``update`` rewrites expected.json from the cached fixtures alone, for a
change that moves results without new texts to embed or score (a cutoff, a cap, routing).
Review the diff of expected.json like any other change.

Usage:
  python -I evaluation/suite/ci_tier.py check [--work DIR]
  python -I evaluation/suite/ci_tier.py refresh
  python -I evaluation/suite/ci_tier.py update
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out

import common  # noqa: E402
from common import HERE, index_doc_id, sha256_file  # noqa: E402

FIXTURE = HERE / "fixtures" / "ci"
_ROOT: Path | None = None
# runner.py imports this module by name; run as a script, it must be the same module.
sys.modules.setdefault("ci_tier", sys.modules[__name__])


def work_root() -> Path:
    global _ROOT
    if _ROOT is None:
        _ROOT = Path(os.environ.get("EVAL_CI_WORK") or tempfile.mkdtemp(prefix="eval-ci-"))
    return _ROOT


def set_root(path: Path) -> None:
    global _ROOT
    _ROOT = Path(path)


def prepare(root: Path) -> None:
    prepared = root / "prepared"
    if root.exists():
        shutil.rmtree(root)
    prepared.mkdir(parents=True)
    for name in ("questions.jsonl", "corpus.jsonl"):
        shutil.copyfile(FIXTURE / name, prepared / name)
    manifest = {"dataset": "ci", "seed": None, "source": "evaluation/suite/fixtures/ci (written for this suite)",
                "questions_sha256": sha256_file(prepared / "questions.jsonl"),
                "corpus_sha256": sha256_file(prepared / "corpus.jsonl")}
    (prepared / "PREPARED.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def evaluate(offline: bool, cache=None) -> dict:
    """Build, run and summarize the slice (in ``work_root()``)."""
    import index
    import metrics
    import runner
    from cache import Cache

    root = work_root()
    prepare(root)
    own = cache is None
    if own:
        cache = Cache(root / "cache.sqlite3")
        config = common.settings()
        cache.load_export(FIXTURE, config.embedding_model, config.rerank_model)
    index.build("ci", offline=offline, cache=cache, target=root / "index", prepared=root / "prepared")
    path = runner.run("ci", "ci", workers=1, offline=offline, cache=cache, restart=True, quiet=True)
    records = runner.load_records(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if own:
        cache.close()
    questions = common.read_jsonl(FIXTURE / "questions.jsonl")
    if len(records) != len(questions):
        raise SystemExit(f"{len(questions) - len(records)} questions failed (see the output above)")
    names = {index_doc_id("ci", row["doc_id"]): row["doc_id"] for row in common.read_jsonl(FIXTURE / "corpus.jsonl")}
    summary = metrics.summarize(records, meta["retrieval_settings"])
    summary.pop("latency_ms", None)
    return _rounded({
        "summary": summary,
        "kept": {r["id"]: [names[h["d"]] for h in r["stages"]["final"]] for r in records},
        "pool": {r["id"]: [names[h["d"]] for h in r["stages"]["pool"][:10]] for r in records},
        "context": {r["id"]: [names[h["d"]] for h in r["stages"]["context"]] for r in records},
        "profiles": {r["id"]: r["profile"] for r in records},
    })


def diff(expected, actual, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            if key not in actual or key not in expected:
                out.append(f"{path}/{key}: {'missing' if key not in actual else 'unexpected'}")
            else:
                out += diff(expected[key], actual[key], f"{path}/{key}")
        return out
    return [] if expected == actual else [f"{path}: expected {expected!r}, got {actual!r}"]


def check() -> int:
    expected = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    actual = evaluate(offline=True)
    problems = diff(expected, actual)
    if problems:
        print("offline retrieval evaluation changed:")
        for line in problems[:40]:
            print("  " + line)
        return 1
    s = actual["summary"]
    print(f"offline retrieval evaluation unchanged: {s['questions']} questions, final recall@5 "
          f"{s['final']['recall@5']:.3f}, MRR {s['final']['mrr']:.3f}, negatives returned {s['negative']['returned']:.1f}")
    return 0


def refresh() -> None:
    """Recompute vectors, scores and expected.json with the live models (needs the key)."""
    from cache import Cache

    root = work_root()
    cache_file = Path(tempfile.mkdtemp(prefix="eval-ci-cache-")) / "cache.sqlite3"
    cache = Cache(cache_file)
    evaluate(offline=False, cache=cache)
    vectors = {k for (k,) in cache.conn.execute("SELECT key FROM vectors")}
    scores = {k for (k,) in cache.conn.execute("SELECT key FROM scores")}
    cache.export(FIXTURE, vectors, scores)
    cache.close()
    set_root(root)
    result = evaluate(offline=True)
    (FIXTURE / "expected.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"refreshed: {len(vectors)} vectors, {len(scores)} rerank scores")


def update() -> None:
    """Rewrite expected.json from the cached fixtures (offline: a missing vector or score fails)."""
    result = evaluate(offline=True)
    (FIXTURE / "expected.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                                           newline="\n")
    print("expected.json rewritten from the cached vectors and scores")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "refresh", "update"])
    parser.add_argument("--work", help="work folder (default: a new temporary folder)")
    args = parser.parse_args()
    if args.work:
        set_root(Path(args.work) / "ci")
    if args.command == "refresh":
        refresh()
        return 0
    if args.command == "update":
        update()
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main())
