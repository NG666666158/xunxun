"""Build an isolated index for a prepared dataset through the app's ingestion pipeline.

Each corpus document becomes what the app's .txt parser makes of a file (one unit, the
file name as heading) and is committed with ``IngestionPipeline._commit``: the app's
chunker (parents and children, the "【title】" prefix), its vector reuse, the store's
``replace_document`` and the parent store. No file registry is involved: tens of
thousands of passages are not added as files.

Child vectors are computed first, in large batches through the text-keyed cache
(cache.py), so the per-document commits read every vector from the cache.

Usage:
  python -I evaluation/suite/index.py crud_rag [--workers 4] [--rebuild]
Writes <work>/<dataset>/index/ (knowledge.sqlite3, parents/, parsed/, cache/) and
BUILT.json (sizes, timings, the prepared checksums).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out

import common  # noqa: E402
from common import WORK, index_doc_id, iter_jsonl  # noqa: E402
from cache import Cache, CachedEmbeddings  # noqa: E402

from db.parent_store_manager import ParentStoreManager  # noqa: E402
from ingestion.models import DocumentUnit, ParsedDocument  # noqa: E402
from ingestion.pipeline import IngestionPipeline  # noqa: E402
from retrieval.embedded_store import EmbeddedStore  # noqa: E402


def cache_path() -> Path:
    return WORK / "cache.sqlite3"


def index_dir(name: str) -> Path:
    return WORK / name / "index"


def documents(name: str, prepared: Path) -> list[ParsedDocument]:
    out = []
    for row in iter_jsonl(prepared / "corpus.jsonl"):
        source = row["source"]
        # As MultiFormatParser._parse_text: the cleaned text, the file stem as heading.
        out.append(ParsedDocument(index_doc_id(name, row["doc_id"]), source, "txt",
                                  [DocumentUnit(text=row["text"], heading=Path(source).stem)]))
    return out


def lengths(values: list[int]) -> dict:
    values = sorted(values)
    return {"n": len(values), "median": statistics.median(values), "mean": round(statistics.fmean(values), 1),
            "p90": values[int(0.9 * (len(values) - 1))], "max": values[-1]}


def embed_all(embeddings: CachedEmbeddings, texts: list[str], workers: int, batch: int = 32) -> int:
    """Fill the cache for every text not in it yet (``workers`` requests at a time)."""
    found = embeddings.cache.get_vectors(embeddings.model, texts)
    missing = list(dict.fromkeys(t for t in texts if common.text_key(embeddings.model, t) not in found))
    if not missing:
        return 0
    groups = [missing[i:i + batch * 8] for i in range(0, len(missing), batch * 8)]
    done = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        for count in pool.map(lambda group: len(embeddings.embed_documents(group)), groups):
            done += count
            if done % (batch * 80) < batch * 8:
                rate = done / max(1e-6, time.perf_counter() - started)
                print(f"  embedded {done}/{len(missing)} ({rate:.0f}/s)", flush=True)
    return len(missing)


def build(name: str, workers: int = 4, rebuild: bool = False, offline: bool = False,
          cache: Cache | None = None, target: Path | None = None, prepared: Path | None = None) -> dict:
    prepared = prepared or WORK / name / "prepared"
    target = target or index_dir(name)
    built = target / "BUILT.json"
    manifest = json.loads((prepared / "PREPARED.json").read_text(encoding="utf-8"))
    if built.is_file() and not rebuild:
        stats = json.loads(built.read_text(encoding="utf-8"))
        if stats.get("corpus_sha256") == manifest["corpus_sha256"]:
            return stats
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    config = common.settings(data_dir=target)
    common.check_models(config)
    own_cache = cache is None
    cache = cache or Cache(cache_path())
    embeddings = CachedEmbeddings(cache, config, offline=offline)
    store = EmbeddedStore(config, db_path=target / "knowledge.sqlite3")
    store.ensure_collection()
    for folder in (config.parsed_dir, config.parent_store_dir, config.cache_dir):
        Path(folder).mkdir(parents=True, exist_ok=True)
    pipeline = IngestionPipeline(store, embeddings, config, ParentStoreManager(config))
    docs = documents(name, prepared)
    started = time.perf_counter()
    children, parents, texts = [], [], []
    for document in docs:
        doc_parents, drafts = pipeline.chunker.split_parent_child(document)
        children += [len(d.text) for d in drafts]
        parents += [len(p.text) for p in doc_parents]
        texts += [d.text for d in drafts]
    embedded = embed_all(embeddings, texts, workers) if not offline else 0
    embed_seconds = time.perf_counter() - started
    for count, document in enumerate(docs, start=1):
        pipeline._commit(document.doc_id, document, None, {})
        if count % 5000 == 0:
            print(f"  committed {count}/{len(docs)}", flush=True)
    store.close()
    stats = {
        "dataset": name, "documents": len(docs), "children": lengths(children), "parents": lengths(parents),
        "embedded_now": embedded, "embed_live_ms": round(embeddings.live.ms, 1), "embed_requests": embeddings.live.calls,
        "embed_seconds": round(embed_seconds, 1), "total_seconds": round(time.perf_counter() - started, 1),
        "corpus_sha256": manifest["corpus_sha256"], "questions_sha256": manifest["questions_sha256"],
        "settings": {f: getattr(config, f) for f in common.RETRIEVAL_FIELDS},
    }
    built.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if own_cache:
        cache.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    for name in args.datasets:
        stats = build(name, args.workers, args.rebuild)
        print(name, json.dumps({k: stats[k] for k in ("documents", "children", "parents", "embedded_now", "embed_seconds", "total_seconds")}), flush=True)


if __name__ == "__main__":
    main()
