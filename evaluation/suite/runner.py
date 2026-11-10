"""Run the app's retrieval on a dataset and record every stage of it, per question.

The question goes through ``HybridRetriever.retrieve`` exactly as the retrieve node
calls it: the question type (``classify_question``) sets ``max_keep``; inside, the
store's hybrid search (dense KNN + jieba BM25, fused by RRF), the candidate pool size
(auto-scaled by library size), the cross-encoder over the whole pool, the score cutoffs,
the per-question-type cap and the per-document cap, and comparison questions' own path.
The kept hits then go through ``AgentNodes.build_context`` (unique parents, the question
type's token budget): the parent expansion the answer model sees.

Only the model clients are replaced, by caches keyed by text (cache.py), and the store
is wrapped to time and record each route. ``--offline`` forbids network calls: a cache
miss fails the question.

Datasets: the prepared public sets (adapters.py + index.py), the repository's
``books`` / ``complex`` / ``benchmark`` sets (on the indexes retrieval_tuning_v1 measured)
and the CI fixture ``ci`` (fixtures/ci, see ci_tier.py).

Usage:
  python -I evaluation/suite/runner.py crud_rag --name base --workers 3
  python -I evaluation/suite/runner.py crud_rag --name base --offline --workers 1 --tag latency
  python -I evaluation/suite/runner.py duretrieval --name planner --rewrite planner --limit 100
Records: <work>/<dataset>/runs/<name>.jsonl (resumable). Run metadata (provenance):
<work>/<dataset>/runs/<name>.meta.json.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out

import common  # noqa: E402
from common import ROOT, WORK, index_doc_id, read_jsonl  # noqa: E402
from cache import Cache, CachedEmbeddings, CachedReranker  # noqa: E402

from rag_agent.answering import classify_question, retrieval_limits  # noqa: E402
from retrieval import embedded_store as embedded_store_module  # noqa: E402
from retrieval import multipart  # noqa: E402
from retrieval.embedded_store import EmbeddedStore  # noqa: E402
from retrieval.hybrid_retriever import HybridRetriever  # noqa: E402

LEGACY = ("books", "complex", "benchmark")


# ---------------------------------------------------------------- per-thread recording
class Recorder(threading.local):
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.ms: dict[str, float] = {}
        self.dense: list | None = None
        self.bm25: list | None = None
        self.returned: list = []  # every hit the store handed out, in order
        self.rerank_docs: list[str] | None = None
        self.rerank_scores: list[float] | None = None

    def add(self, stage: str, ms: float) -> None:
        self.ms[stage] = self.ms.get(stage, 0.0) + ms


REC = Recorder()


@contextmanager
def timed(stage: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        REC.add(stage, (time.perf_counter() - started) * 1000)


def instrument_store(store: EmbeddedStore) -> None:
    """Time the dense route, the BM25 route (query segmentation included) and keep their rankings."""
    dense, sparse = store._dense, store._sparse

    def timed_dense(conn, vector, limit):
        with timed("dense"):
            hits = dense(conn, vector, limit)
        if REC.dense is None:
            REC.dense = list(hits)
        return hits

    def timed_sparse(conn, match, limit):
        with timed("bm25"):
            hits = sparse(conn, match, limit)
        if REC.bm25 is None:
            REC.bm25 = list(hits)
        return hits

    store._dense, store._sparse = timed_dense, timed_sparse


_build_match_query = embedded_store_module.build_match_query


def _timed_match_query(query: str) -> str:
    with timed("bm25"):
        return _build_match_query(query)


def instrument_tokenizer() -> None:
    """The BM25 route's query segmentation (jieba) counts as BM25 time."""
    embedded_store_module.build_match_query = _timed_match_query


class StoreProxy:
    """What the retriever sees: the store, with its searches timed and their hits kept."""

    def __init__(self, store: EmbeddedStore):
        self._store = store

    def __getattr__(self, name):
        return getattr(self._store, name)

    def _keep(self, hits):
        REC.returned.extend(hits)
        return hits

    def search_hybrid(self, *args, **kwargs):
        with timed("search"):
            return self._keep(self._store.search_hybrid(*args, **kwargs))

    def search_in_source(self, *args, **kwargs):
        with timed("search"):
            return self._keep(self._store.search_in_source(*args, **kwargs))

    def search_dense(self, *args, **kwargs):
        with timed("search"):
            return self._keep(self._store.search_dense(*args, **kwargs))

    def search_sparse(self, *args, **kwargs):
        with timed("search"):
            return self._keep(self._store.search_sparse(*args, **kwargs))

    def chunks_of_parents(self, *args, **kwargs):
        with timed("search"):
            return self._keep(self._store.chunks_of_parents(*args, **kwargs))


class RecordingReranker:
    def __init__(self, inner: CachedReranker):
        self.inner = inner

    def rerank(self, query, documents, top_n):
        documents = list(documents)
        with timed("rerank"):
            items = self.inner.rerank(query, documents, top_n)
        scores = [0.0] * len(documents)
        for item in items:
            scores[item.index] = item.score
        if REC.rerank_docs is None:
            REC.rerank_docs, REC.rerank_scores = documents, scores
        return items


class TimedEmbeddings:
    def __init__(self, inner: CachedEmbeddings):
        self.inner = inner

    def embed_query(self, text: str):
        with timed("embed"):
            return self.inner.embed_query(text)

    def embed_queries(self, texts):
        """A question and its parts in one request (retrieval/multipart.py)."""
        with timed("embed"):
            return self.inner.embed_documents(list(texts))

    def __getattr__(self, name):
        return getattr(self.inner, name)


# ---------------------------------------------------------------- datasets
_SPACE = re.compile(r"[\W_]+", re.UNICODE)


def loose(text: str) -> str:
    return _SPACE.sub("", text).lower()


class Dataset:
    """Questions, the index to search and how a hit is judged."""

    name: str
    unit = "doc"
    data_dir: Path
    questions: list[dict]
    manifest: dict

    def flags(self, question: dict, hit: dict) -> list[int]:
        raise NotImplementedError

    def span_flags(self, question: dict, text: str) -> list[int]:
        return []

    def answerable(self, q: dict) -> bool:
        return q["answerable"]

    def needs_all(self, q: dict) -> bool:
        return q["needs_all"]

    def n_gold(self, q: dict) -> int:
        return len(q["gold_doc_ids"])

    def query(self, q: dict) -> str:
        return q["query"]

    def seed_cache(self, cache: Cache, model: str) -> None:
        pass


class Prepared(Dataset):
    def __init__(self, name: str, root: Path | None = None):
        self.name = name
        root = root or WORK / name
        self.data_dir = root / "index"
        self.questions = read_jsonl(root / "prepared" / "questions.jsonl")
        self.manifest = json.loads((root / "prepared" / "PREPARED.json").read_text(encoding="utf-8"))
        built = self.data_dir / "BUILT.json"
        if not built.is_file():
            raise SystemExit(f"{self.data_dir} is not built: python -I evaluation/suite/index.py {name}")
        self.built = json.loads(built.read_text(encoding="utf-8"))
        if self.built["corpus_sha256"] != self.manifest["corpus_sha256"]:
            raise SystemExit(f"{self.data_dir} was built from another corpus: rebuild it")
        self._gold = {q["id"]: [index_doc_id(name, d) for d in q["gold_doc_ids"]] for q in self.questions}
        self._spans = {q["id"]: [loose(p["text"]) for p in q["gold_passages"] if p.get("match") == "text"]
                       for q in self.questions}

    def flags(self, question: dict, hit: dict) -> list[int]:
        return [i for i, doc in enumerate(self._gold[question["id"]]) if doc == hit["doc_id"]]

    def span_flags(self, question: dict, text: str) -> list[int]:
        spans = self._spans[question["id"]]
        if not spans:
            return []
        body = loose(text)
        return [i for i, span in enumerate(spans) if span and span in body]


class Legacy(Dataset):
    """books_v1 / complex_v1 / benchmark_v1 on the indexes retrieval_tuning_v1 measured."""

    unit = "passage"

    def __init__(self, name: str):
        import importlib.util

        # retrieval_tuning_v1/common.py, under another name (ours is "common").
        spec = importlib.util.spec_from_file_location("rt_common", ROOT / "evaluation" / "retrieval_tuning_v1" / "common.py")
        rt_common = importlib.util.module_from_spec(spec)
        sys.modules["rt_common"] = rt_common
        spec.loader.exec_module(rt_common)

        self.name = name
        self.qa = rt_common.load(name)
        self.data_dir = self.qa.data_dir
        self.questions = [legacy_question(name, q) for q in self.qa.questions]
        self._raw = {q["id"]: q for q in self.qa.questions}
        sources = {"books": "evaluation/books_v1/questions.jsonl", "complex": "evaluation/complex_v1/questions.jsonl",
                   "benchmark": "evaluation/benchmark_v1/retrieval_test.jsonl"}
        path = ROOT / sources[name]
        self.manifest = {"dataset": name, "seed": None, "questions": len(self.questions),
                         "questions_sha256": common.sha256_file(path), "source": sources[name],
                         "index": os.path.relpath(self.data_dir, ROOT).replace(os.sep, "/")}

    def flags(self, question: dict, hit: dict) -> list[int]:
        return self.qa.gold_flags(self._raw[question["source"]["original_id"]], hit)

    def answerable(self, q):
        return self.qa.answerable(self._raw[q["source"]["original_id"]])

    def needs_all(self, q):
        return self.qa.needs_all(self._raw[q["source"]["original_id"]])

    def n_gold(self, q):
        return self.qa.gold_count(self._raw[q["source"]["original_id"]])

    def seed_cache(self, cache: Cache, model: str) -> None:
        """The query vectors the earlier reports used, so a re-run searches with the same vectors."""
        items = [(q["question"], v) for q, v in zip(self.qa.questions, self.qa.vectors, strict=True)]
        cache.put_vectors(model, items)


def legacy_question(name: str, q: dict) -> dict:
    """books / complex / benchmark questions in the suite's schema."""
    import schema

    if name == "benchmark":
        golds = [{"doc_id": "", "chunk_ids": q["relevant_child_ids"], "match": "chunk"}]
        kind = q.get("category") or q.get("type") or "single"
        return schema.question(id=f"benchmark:{q['id']}", query=q["question"], language="zh", type=str(kind),
                               tags=[], gold_doc_ids=[], gold_passages=golds, gold_answer=str(q.get("answer") or ""),
                               answerable=True, needs_all=False,
                               source={"dataset": "benchmark_v1", "original_id": q["id"]})
    golds = [{"doc_id": g.get("source", ""), "locator": str(g.get("pages", g.get("page", ""))),
              "keywords": list(g.get("keywords") or []), "match": "books_strict"} for g in q.get("gold", [])]
    kind = q.get("subtype") or q.get("type") or "single"
    answerable = q.get("type") != "refusal" and q.get("answerable", True)
    return schema.question(id=f"{name}:{q['id']}", query=q["question"], language="zh", type=str(kind), tags=[],
                           gold_doc_ids=sorted({g["doc_id"] for g in golds if g["doc_id"]}), gold_passages=golds,
                           gold_answer=str(q.get("answer") or ""), answerable=bool(answerable and golds),
                           needs_all=q.get("type") == "cross_book",
                           source={"dataset": f"{name}_v1", "original_id": q["id"]})


def load_dataset(name: str) -> Dataset:
    if name in LEGACY:
        return Legacy(name)
    if name == "ci":
        import ci_tier

        return Prepared("ci", ci_tier.work_root())
    return Prepared(name)


# ---------------------------------------------------------------- query routing (planner)
def plan_query(nodes, question: str) -> dict[str, Any]:
    """analyze_query for a first-turn question without a file search service."""
    from rag_agent.nodes import SMALL_TALK

    if SMALL_TALK.match(question.strip()):
        return {"route": "chitchat", "query": question, "intent": "chitchat", "ms": 0.0}
    if nodes._plain_knowledge_question(question):
        return {"route": "raw", "query": question[:500], "intent": "knowledge", "ms": 0.0}
    started = time.perf_counter()
    plan = nodes._plan(question, "")
    return {"route": "planner", "query": plan.rewritten_query, "intent": plan.intent,
            "ms": round((time.perf_counter() - started) * 1000, 1)}


def planner_nodes(config):
    from rag_agent.nodes import AgentNodes
    from services.model_clients import build_chat_model

    return AgentNodes(build_chat_model(config), retriever=None, settings=config)


def route_only(question: str, file_search: bool = False) -> str:
    """Where analyze_query sends a first-turn question, without calling the planner.

    ``file_search``: the app has a file search service (the desktop app always does); then a
    strong find-file cue ("在哪", "哪份", "帮我找" …) sends the question to file search and
    it never reaches passage retrieval. (The other branch of ``_obviously_find_file``, a
    short title matching a file name, needs the library and is not counted.)
    """
    from rag_agent.nodes import HISTORY_REFERENCE, SMALL_TALK, AgentNodes
    from retrieval.file_filters import find_file_cue

    if SMALL_TALK.match(question.strip()):
        return "chitchat"
    if file_search and not HISTORY_REFERENCE.search(question) and find_file_cue(question) == "strong":
        return "find_file"
    return "raw" if AgentNodes._plain_knowledge_question(question) else "planner"


# ---------------------------------------------------------------- one question
def short(hit, flags: list[int], score_key: str = "score", extra: dict | None = None) -> dict:
    row = {"c": hit.id[:16], "d": hit.doc_id, "p": hit.parent_id, "g": flags,
           "s": round(float(getattr(hit, score_key) or 0.0), 6)}
    if extra:
        row.update(extra)
    return row


def run_question(ds: Dataset, q: dict, retriever: HybridRetriever, context_builder: Callable, embeddings: CachedEmbeddings,
                 reranker: CachedReranker, query: str, profile: str) -> dict:
    REC.reset()
    embeddings.reset_thread()
    reranker.reset_thread()
    limits = retrieval_limits(profile)  # what the retrieve node passes for this question type

    def flags(hit) -> list[int]:
        return ds.flags(q, {"id": hit.id, "doc_id": hit.doc_id, "parent_id": hit.parent_id, "source": hit.source,
                            "locator": hit.locator, "heading": hit.heading, "text": hit.text})

    started = time.perf_counter()
    hits = retriever.retrieve(query, mode="hybrid_rerank", **limits)
    retrieve_ms = (time.perf_counter() - started) * 1000
    if REC.rerank_docs is None:
        raise RuntimeError("the reranker was not reached")
    trace = multipart.last_trace()
    # Map the reranked documents back to the hits the store returned (same order); candidates
    # the multi-part steps added (searched in other threads) are in the trace.
    available = list(REC.returned) + list(trace.extra_hits if trace else [])
    kinds = trace.extra_kinds if trace else {}
    pool = []
    for index, text in enumerate(REC.rerank_docs):
        position = next((i for i, h in enumerate(available) if h.text == text), None)
        if position is None:
            raise RuntimeError("a reranked passage is not among the store's hits")
        hit = available.pop(position)
        extra = {"r": round(REC.rerank_scores[index], 6)}
        if hit.id in kinds:
            extra["x"] = kinds[hit.id]
        pool.append(short(hit, flags(hit), extra=extra))
    started = time.perf_counter()
    context = context_builder(hits, profile)
    context_ms = (time.perf_counter() - started) * 1000
    final_flags = {hit.id: flags(hit) for hit in hits}
    context_rows = []
    for entry in context:
        sharing = [h for h in hits if h.parent_id == entry.get("parent_id")]
        if ds.unit == "passage":
            golds = sorted({g for h in sharing for g in final_flags[h.id]})
        else:
            golds = [i for i, doc in enumerate(ds._gold[q["id"]]) if doc == entry.get("doc_id")]  # type: ignore[attr-defined]
        context_rows.append({"c": str(entry.get("id", ""))[:16], "d": entry.get("doc_id"), "p": entry.get("parent_id"),
                             "g": golds, "t": ds.span_flags(q, str(entry.get("text", ""))),
                             "tok": entry.get("_tokens")})
    ms = dict(REC.ms)
    latency = {
        "embed": ms.get("embed", 0.0), "dense": ms.get("dense", 0.0), "bm25": ms.get("bm25", 0.0),
        "search": ms.get("search", 0.0), "rerank": ms.get("rerank", 0.0), "context": context_ms,
        "retrieve": retrieve_ms, "total": retrieve_ms + context_ms,
    }
    latency["fuse"] = max(0.0, latency["search"] - latency["dense"] - latency["bm25"])
    latency["select"] = max(0.0, retrieve_ms - latency["embed"] - latency["search"] - latency["rerank"])
    live_embed, live_rerank = embeddings.last_call(), reranker.last_call()
    return {
        "id": q["id"], "dataset": ds.name, "type": q["type"], "tags": q["tags"], "language": q["language"],
        "answerable": ds.answerable(q), "needs_all": ds.needs_all(q), "n_gold": ds.n_gold(q), "unit": ds.unit,
        "query_used": query if query != q["query"] else None, "profile": profile, "max_keep": limits["max_keep"],
        "per_document": limits.get("per_document"),
        "n_spans": len(getattr(ds, "_spans", {}).get(q["id"], [])),
        "stages": {
            "dense": [short(h, flags(h)) for h in (REC.dense or [])],
            "bm25": [short(h, flags(h)) for h in (REC.bm25 or [])],
            "pool": pool,
            "final": [short(h, final_flags[h.id], "rerank_score", {"r": h.rerank_score, "t": ds.span_flags(q, h.text)})
                      for h in hits],
            "context": context_rows,
        },
        "latency_ms": {k: round(v, 2) for k, v in latency.items()},
        "live": {"embed_ms": round(live_embed.ms, 1), "embed_calls": live_embed.calls,
                 "rerank_ms": round(live_rerank.ms, 1), "rerank_calls": live_rerank.calls, "rerank_docs": live_rerank.items},
        # What the multi-part steps did (retrieval/multipart.py); rerank_calls has every
        # request's passages and estimated input tokens, the per-part ones included.
        "parts": trace.to_dict() if trace else None,
    }


def context_builder_for(config) -> Callable:
    from db.parent_store_manager import ParentStoreManager
    from rag_agent.nodes import AgentNodes

    parents = ParentStoreManager(config) if Path(config.parent_store_dir).is_dir() else None
    estimate = AgentNodes._estimate_tokens

    def build(hits, profile: str) -> list[dict]:
        holder = SimpleNamespace(settings=config, parent_store=parents, _estimate_tokens=estimate, documents=None)
        holder._label_same_names = lambda documents: AgentNodes._label_same_names(holder, documents)
        state = {"retrieve_chunks": [h.to_dict() for h in hits], "answer_profile": profile}
        selected = AgentNodes.build_context(holder, state)["context_docs"]
        for entry in selected:
            entry["_tokens"] = estimate(str(entry.get("text", "")))
        return selected

    return build


def expand(record: dict) -> dict:
    """Adds the ``reranked`` stage (the pool in rerank order: before the cut)."""
    pool = record["stages"]["pool"]
    order = sorted(range(len(pool)), key=lambda i: (-(pool[i].get("r") or 0.0), i))
    record["stages"]["reranked"] = [pool[i] for i in order]
    return record


def load_records(path: Path, ids: list[str] | None = None) -> list[dict]:
    rows = {row["id"]: row for row in read_jsonl(path)} if path.is_file() else {}
    order = ids or list(rows)
    return [expand(rows[i]) for i in order if i in rows]


def parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(text.lower(), text)


def runs_dir(name: str) -> Path:
    if name == "ci":
        import ci_tier

        return ci_tier.work_root() / "runs"
    return WORK / name / "runs"


def run(name: str, run_name: str = "base", workers: int = 3, offline: bool = False, limit: int = 0,
        restart: bool = False, overrides: dict | None = None, rewrite: str = "", only_route: str = "", only_type: str = "",
        cache: Cache | None = None, tag: str = "", ids: list[str] | None = None, quiet: bool = False,
        fresh_cache: bool = False) -> Path:
    overrides = overrides or {}
    ds = load_dataset(name)
    config = common.settings(data_dir=ds.data_dir, **overrides)
    common.check_models(config)
    own_cache = cache is None
    if fresh_cache:
        # Network timing: every embedding and rerank call goes out (nothing is reused).
        import tempfile

        cache = Cache(Path(tempfile.mkdtemp(prefix="eval-fresh-")) / "cache.sqlite3")
    cache = cache or Cache(WORK / "cache.sqlite3")
    if not fresh_cache:
        ds.seed_cache(cache, config.embedding_model)
    embeddings = CachedEmbeddings(cache, config, offline=offline)
    reranker = CachedReranker(cache, config, offline=offline)
    store = EmbeddedStore(config, db_path=ds.data_dir / "knowledge.sqlite3")
    instrument_store(store)
    instrument_tokenizer()
    retriever = HybridRetriever(StoreProxy(store), TimedEmbeddings(embeddings), RecordingReranker(reranker), config)
    builder = context_builder_for(config)
    nodes = planner_nodes(config) if rewrite == "planner" else None

    folder = runs_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    full = run_name + (f"_{tag}" if tag else "")
    path = folder / f"{full}.jsonl"
    if restart:
        path.unlink(missing_ok=True)
    done = {row["id"] for row in read_jsonl(path)} if path.is_file() else set()
    questions = ds.questions
    if ids:
        wanted = set(ids)
        questions = [q for q in questions if q["id"] in wanted]
    if only_route:
        questions = [q for q in questions if route_only(ds.query(q)) == only_route]
    if only_type:
        questions = [q for q in questions if q["type"] in only_type.split(",")]
    if limit:
        questions = questions[:limit]
    todo = [q for q in questions if q["id"] not in done]
    meta = common.provenance(config, dataset=ds.manifest, run={
        "name": full, "offline": offline, "workers": workers, "limit": limit, "overrides": overrides,
        "rewrite": rewrite, "only_route": only_route, "only_type": only_type, "questions": len(questions)})
    (folder / f"{full}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Warm-up (not recorded, no model call): jieba's dictionary and the index pages.
    if todo:
        unit = [1.0] + [0.0] * (int(config.embedding_dimension) - 1)
        store.search_hybrid(ds.query(todo[0]), unit, limit=config.fusion_top_k)
        REC.reset()

    lock = threading.Lock()
    failures = []
    with path.open("a", encoding="utf-8", newline="\n") as handle, ThreadPoolExecutor(max(1, workers)) as pool:
        def work(q):
            try:
                query, profile, plan = ds.query(q), classify_question(ds.query(q)), None
                if nodes is not None:
                    plan = plan_query(nodes, query)
                    if plan["route"] == "planner":
                        query = plan["query"] or query
                        if profile == "fact" and query != q["query"]:
                            profile = classify_question(query)
                row = run_question(ds, q, retriever, builder, embeddings, reranker, query, profile)
                row["route"] = plan or {"route": route_only(q["query"])}
            except Exception as exc:  # left out; a rerun fills it in
                failures.append(q["id"])
                print(f"skipped {q['id']}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                return
            with lock:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

        for count, _ in enumerate(pool.map(work, todo), start=1):
            if not quiet and count % 100 == 0:
                print(f"{name} {full}: {count}/{len(todo)} (live rerank {reranker.live.calls}, embed {embeddings.live.calls})", flush=True)
    store.close()
    if own_cache:
        cache.close()
    if not quiet:
        print(f"written {path} ({len(todo) - len(failures)} new, {len(failures)} skipped)", flush=True)
    return path


def check_pair_independence(samples: int = 3) -> None:
    """Rerank scores of a pair do not depend on the other passages in the request."""
    config = common.settings()
    cache = Cache(WORK / "cache.sqlite3")
    reranker = CachedReranker(cache, config)
    query = "国家法定节假日一共有多少天"
    docs = ["根据规定，全体公民放假的节日共有11天。", "春节放假3天，农历除夕、正月初一、初二。",
            "今天天气很好，适合出去散步。", "劳动节放假1天（5月1日）。", "Python 是一种编程语言。"]
    together = reranker.score_live(query, docs)
    alone = [reranker.score_live(query, [d])[0] for d in docs]
    shuffled = reranker.score_live(query, docs[::-1])[::-1]
    worst = max(max(abs(a - b) for a, b in zip(together, alone)), max(abs(a - b) for a, b in zip(together, shuffled)))
    print("together", [round(s, 6) for s in together])
    print("alone   ", [round(s, 6) for s in alone])
    print("max difference", worst)
    embeddings = CachedEmbeddings(cache, config)
    a = embeddings._fetch(docs[:2] + [query])
    b = embeddings._fetch([query])
    import numpy as np

    print("query vector alone vs in a batch, max |diff|:", float(np.max(np.abs(np.asarray(a[2]) - np.asarray(b[0])))))
    cache.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("--name", default="base")
    parser.add_argument("--tag", default="")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--set-option", action="append", default=[], help="Settings field=value (an ablation)")
    parser.add_argument("--rewrite", choices=["", "planner"], default="")
    parser.add_argument("--only-route", choices=["", "raw", "planner"], default="")
    parser.add_argument("--only-type", default="", help="comma-separated question types")
    parser.add_argument("--check-reranker", action="store_true")
    parser.add_argument("--fresh-cache", action="store_true", help="an empty cache: time the network calls")
    args = parser.parse_args()
    if args.check_reranker:
        check_pair_independence()
        return
    overrides = {k: parse_value(v) for k, _, v in (o.partition("=") for o in args.set_option)}
    run(args.dataset, args.name, args.workers, args.offline, args.limit, args.restart, overrides,
        args.rewrite, args.only_route, args.only_type, tag=args.tag, fresh_cache=args.fresh_cache)


if __name__ == "__main__":
    main()
