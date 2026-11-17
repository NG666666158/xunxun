"""Shared by the recall_wording_v1 scripts: the index of the synthetic library, a retriever
over it with cached model outputs, and what the reranker was sent for each question.

Model outputs (bge-m3 vectors, bge-reranker scores) go through evaluation/suite/cache.py, so
a re-run with the same pools is offline and free. The key is read from project/.env or the
environment (SILICONFLOW_API_KEY); it is never written anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUITE = ROOT / "evaluation" / "suite"
for folder in (SUITE, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import common as suite_common  # noqa: E402  (evaluation/suite/common.py; puts project/ on sys.path)
from cache import Cache, CachedEmbeddings, CachedReranker  # noqa: E402

WORK = Path(os.environ.get("RW_WORK_DIR", HERE / "work"))
CORPUS = WORK / "corpus"
RESULTS = HERE / "results"


def settings(data_dir: Path | None = None, **overrides):
    return suite_common.settings(data_dir=data_dir, **overrides)


def cache() -> Cache:
    return Cache(WORK / "cache.sqlite3")


def doc_id_of(name: str) -> str:
    return hashlib.sha256(f"recall_wording_v1\n{name}".encode("utf-8")).hexdigest()[:32]


def load_questions(path: Path | None = None) -> list[dict]:
    with (path or HERE / "questions.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_dir(name: str) -> Path:
    return WORK / "index" / name


def build_index(name: str = "base", corpus: Path | None = None, rebuild: bool = False, **overrides) -> dict:
    """Every file of the corpus through the app's parser, chunker and store (no file registry)."""
    from db.parent_store_manager import ParentStoreManager
    from ingestion.pipeline import IngestionPipeline
    from retrieval.embedded_store import EmbeddedStore
    import shutil

    target = index_dir(name)
    built = target / "BUILT.json"
    if built.is_file() and not rebuild:
        return json.loads(built.read_text(encoding="utf-8"))
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    config = settings(data_dir=target, **overrides)
    store_cache = cache()
    embeddings = CachedEmbeddings(store_cache, config)
    store = EmbeddedStore(config, db_path=target / "knowledge.sqlite3")
    store.ensure_collection()
    for folder in (config.parsed_dir, config.parent_store_dir, config.cache_dir):
        Path(folder).mkdir(parents=True, exist_ok=True)
    pipeline = IngestionPipeline(store, embeddings, config, ParentStoreManager(config))
    files = sorted(p for p in (corpus or CORPUS).iterdir() if p.is_file())
    chunks = 0
    for path in files:
        count, _ = pipeline.index_document(path, doc_id_of(path.name))
        chunks += count
    store.close()
    store_cache.close()
    stats = {"index": name, "files": len(files), "chunks": chunks, "overrides": {k: str(v) for k, v in overrides.items()}}
    built.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def all_chunks(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT id, doc_id, source, text FROM chunks ORDER BY rowid").fetchall()
    finally:
        conn.close()
    return [{"id": r[0], "doc_id": r[1], "source": r[2], "text": r[3]} for r in rows]


def squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text)


def gold_chunk_ids(chunks: list[dict], gold: dict) -> set[str]:
    marker = squeeze(gold["marker"])
    return {c["id"] for c in chunks if c["source"] == gold["file"] and marker in squeeze(c["text"])}


class RecordingReranker:
    """Passes through to the cached reranker and keeps each request of the current thread."""

    def __init__(self, inner: CachedReranker):
        self.inner = inner
        self._local = threading.local()

    def reset(self) -> None:
        self._local.calls = []

    @property
    def calls(self) -> list[dict]:
        if not hasattr(self._local, "calls"):
            self._local.calls = []
        return self._local.calls

    def rerank(self, query, documents, top_n):
        documents = list(documents)
        items = self.inner.rerank(query, documents, top_n)
        self.calls.append({"query": query, "texts": documents, "scores": {documents[i.index]: i.score for i in items}})
        return items


class CountingEmbeddings:
    def __init__(self, inner: CachedEmbeddings):
        self.inner = inner
        self.texts = 0
        self.requests = 0

    def embed_query(self, text):
        self.texts += 1
        self.requests += 1
        return self.inner.embed_query(text)

    def embed_queries(self, texts):
        self.texts += len(texts)
        self.requests += 1
        return self.inner.embed_documents(list(texts))

    def embed_documents(self, texts):
        return self.inner.embed_documents(texts)


def open_retriever(index: str | Path = "base", offline: bool = False, **overrides):
    """(retriever, recorder, embeddings counter, store, cache) over an index folder."""
    from retrieval.embedded_store import EmbeddedStore
    from retrieval.hybrid_retriever import HybridRetriever

    folder = index if isinstance(index, Path) else index_dir(index)
    config = settings(data_dir=folder, **overrides)
    store_cache = cache()
    store = EmbeddedStore(config, db_path=folder / "knowledge.sqlite3")
    recorder = RecordingReranker(CachedReranker(store_cache, config, offline=offline))
    counter = CountingEmbeddings(CachedEmbeddings(store_cache, config, offline=offline))
    retriever = HybridRetriever(store, counter, recorder, config)
    return retriever, recorder, counter, store, store_cache


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        else:
            try:
                out[key] = int(value)
            except ValueError:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
    return out
