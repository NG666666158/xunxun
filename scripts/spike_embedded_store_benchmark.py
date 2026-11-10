"""Spike: compare EmbeddedStore (SQLite FTS5 + sqlite-vec) with MilvusStore on benchmark_v1.

Both stores are filled with the *same* chunk records and vectors and queried with the
*same* cached query embeddings, so differences come only from the storage/search layer.

Frozen benchmark files are never modified. The corpus is regenerated into the work
directory; because .docx/.pdf/.xlsx renders embed timestamps, their doc_ids (and so
chunk ids) differ from the frozen run, so ground-truth chunk ids are re-linked by
clause code exactly like ``link_ground_truth.py`` does, into the work directory only.

Usage:
    python scripts/spike_embedded_store_benchmark.py            # full run (resumable)
    python scripts/spike_embedded_store_benchmark.py --scale 10000,50000,100000
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from benchmark_common import BENCHMARK_DIR, ROOT, read_jsonl, write_jsonl
from run_retrieval_benchmark import evaluate_question, summarize

from config import Settings  # noqa: E402
from ingestion.chunker import HeadingAwareChunker  # noqa: E402
from ingestion.parsers import MultiFormatParser  # noqa: E402
from retrieval.embedded_store import EmbeddedStore  # noqa: E402
from retrieval.milvus_store import MilvusStore  # noqa: E402
from retrieval.types import ChunkRecord  # noqa: E402
from services.model_clients import SiliconFlowEmbeddings, SiliconFlowReranker  # noqa: E402


SPIKE_DIR = ROOT / "evaluation" / "spike_embedded_store"
WORK_DIR = SPIKE_DIR / "work"
MILVUS_COLLECTION = "spike_benchmark_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="retrieval_test.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--restart", action="store_true", help="discard cached evaluation details")
    parser.add_argument("--latency-passes", type=int, default=3)
    parser.add_argument("--skip-milvus", action="store_true")
    parser.add_argument("--scale", default="", help="comma separated synthetic chunk counts for the embedded store")
    return parser.parse_args()


def build_settings() -> Settings:
    settings = Settings(
        milvus_collection=MILVUS_COLLECTION,
        data_dir=WORK_DIR / "runtime",
        upload_dir=WORK_DIR / "runtime" / "uploads",
        parsed_dir=WORK_DIR / "runtime" / "parsed",
        parent_store_dir=WORK_DIR / "runtime" / "parents",
        evaluation_dir=WORK_DIR,
        dense_top_k=24,
        sparse_top_k=24,
        fusion_top_k=24,
        rerank_top_k=5,
    )
    settings.ensure_directories()
    return settings


def embed_with_retry(embeddings: SiliconFlowEmbeddings, texts: list[str], attempts: int = 5) -> np.ndarray:
    """Batch-wise embedding; the adapter's own retry does not cover openai.APITimeoutError."""
    vectors = []
    for start in range(0, len(texts), embeddings.batch_size):
        batch = texts[start : start + embeddings.batch_size]
        for attempt in range(attempts):
            try:
                vectors.extend(embeddings._embed_batch(batch))
                break
            except Exception as exc:
                if attempt == attempts - 1:
                    raise
                print(f"embedding batch {start} failed ({type(exc).__name__}); retrying", flush=True)
                time.sleep(2 ** attempt)
    return np.asarray(vectors, dtype=np.float32)


def regenerate_corpus() -> Path:
    bench = WORK_DIR / "bench"
    if (bench / "corpus_manifest.json").is_file():
        return bench
    spec = importlib.util.spec_from_file_location("generate_benchmark", ROOT / "scripts" / "generate_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BENCHMARK_DIR = bench
    module.CORPUS_DIR = bench / "corpus"
    module.main()
    return bench


def chunk_and_embed(bench: Path, settings: Settings, embeddings: SiliconFlowEmbeddings) -> list[ChunkRecord]:
    meta_path = WORK_DIR / "chunks.jsonl"
    vector_path = WORK_DIR / "chunk_vectors.npz"
    if meta_path.is_file() and vector_path.is_file():
        rows = read_jsonl(meta_path)
        vectors = np.load(vector_path)["vectors"]
        return [ChunkRecord(**row, dense=vector.tolist()) for row, vector in zip(rows, vectors, strict=True)]

    parser = MultiFormatParser(settings)
    chunker = HeadingAwareChunker(settings)
    manifest = json.loads((bench / "corpus_manifest.json").read_text(encoding="utf-8"))
    drafts = []
    for document in manifest["documents"]:
        parsed = parser.parse(bench / "corpus" / document["source"])
        drafts.extend(chunker.split_parent_child(parsed)[1])
    started = time.perf_counter()
    vectors = embed_with_retry(embeddings, [draft.text for draft in drafts])
    print(f"embedded {len(drafts)} chunks in {time.perf_counter() - started:.1f}s", flush=True)
    rows = [
        {
            "id": d.id,
            "doc_id": d.doc_id,
            "parent_id": d.parent_id,
            "source": d.source,
            "heading": d.heading,
            "locator": d.locator,
            "text": d.text,
        }
        for d in drafts
    ]
    write_jsonl(meta_path, rows)
    np.savez_compressed(vector_path, vectors=vectors)
    return [ChunkRecord(**row, dense=vector.tolist()) for row, vector in zip(rows, vectors, strict=True)]


def relink_labels(bench: Path, records: list[ChunkRecord], dataset: str) -> tuple[list[dict], dict]:
    """Same clause-code matching as link_ground_truth.py, written to the work dir only."""
    by_source: dict[str, list[ChunkRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source].append(record)
    mapping = {}
    for fact in read_jsonl(bench / "facts.jsonl"):
        matches = [r for r in by_source.get(fact["source"], []) if fact["clause_code"] in r.text]
        mapping[fact["fact_id"]] = (sorted({r.id for r in matches}), sorted({r.parent_id for r in matches}))
        if not matches:
            raise RuntimeError(f"fact {fact['fact_id']} could not be linked")

    frozen = read_jsonl(BENCHMARK_DIR / dataset)
    regenerated = read_jsonl(bench / dataset)
    if [(r["id"], r["question"], r["relevant_fact_ids"]) for r in frozen] != [
        (r["id"], r["question"], r["relevant_fact_ids"]) for r in regenerated
    ]:
        raise RuntimeError("regenerated questions differ from the frozen dataset")

    text_formats = (".md", ".txt")
    same_ids = text_rows = 0
    rows = []
    for row in frozen:
        children, parents = set(), set()
        for fact_id in row["relevant_fact_ids"]:
            children.update(mapping[fact_id][0])
            parents.update(mapping[fact_id][1])
        if row["relevant_source"].endswith(text_formats):
            text_rows += 1
            same_ids += sorted(children) == row["relevant_child_ids"]
        rows.append({**row, "relevant_child_ids": sorted(children), "relevant_parent_ids": sorted(parents)})
    write_jsonl(WORK_DIR / f"relinked_{dataset}", rows)
    check = {
        "questions_identical_to_frozen": True,
        "text_format_rows": text_rows,
        "text_format_rows_with_identical_child_ids": same_ids,
    }
    return rows, check


def query_vectors(rows: list[dict], embeddings: SiliconFlowEmbeddings) -> np.ndarray:
    path = WORK_DIR / "query_embeddings.npz"
    ids = [row["id"] for row in rows]
    if path.is_file():
        cached = np.load(path)
        if list(cached["ids"]) == ids:
            return cached["vectors"]
    vectors = embed_with_retry(embeddings, [row["question"] for row in rows])
    np.savez_compressed(path, vectors=vectors, ids=np.asarray(ids))
    return vectors


def index_store(store, records: list[ChunkRecord]) -> float:
    store.clear()
    started = time.perf_counter()
    store.insert(records)
    return time.perf_counter() - started


def run_eval(name: str, store, rows, vectors, reranker, settings, k: int, restart: bool) -> dict:
    details_path = WORK_DIR / f"{name}_details.jsonl"
    if restart and details_path.exists():
        details_path.unlink()
    completed = {d["question_id"]: d for d in read_jsonl(details_path)} if details_path.is_file() else {}

    class Retriever:  # evaluate_question only touches retriever.reranker
        pass

    retriever = Retriever()
    retriever.reranker = reranker
    started = time.perf_counter()
    with details_path.open("a", encoding="utf-8") as handle:
        for number, (row, vector) in enumerate(zip(rows, vectors, strict=True), start=1):
            if row["id"] in completed:
                continue
            detail = evaluate_question(row, vector.tolist(), store, retriever, settings, k)
            completed[row["id"]] = detail
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
            handle.flush()
            if number % 50 == 0:
                print(f"[{name}] {number}/{len(rows)} {time.perf_counter() - started:.0f}s", flush=True)
    details = [completed[row["id"]] for row in rows]
    return summarize(details, 0.0, 1000)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def latency_profile(store, rows, vectors, settings, passes: int) -> dict:
    """Warm, in-process timings of each search route (no reranker, no embedding)."""
    timings: dict[str, list[float]] = {"dense": [], "sparse": [], "hybrid": []}
    for row, vector in zip(rows[:20], vectors[:20]):  # warm-up
        store.search_hybrid(row["question"], vector.tolist(), limit=settings.fusion_top_k)
    for _ in range(passes):
        for row, vector in zip(rows, vectors):
            values = vector.tolist()
            for route, call in (
                ("dense", lambda: store.search_dense(values, limit=settings.dense_top_k)),
                ("sparse", lambda: store.search_sparse(row["question"], limit=settings.sparse_top_k)),
                ("hybrid", lambda: store.search_hybrid(row["question"], values, limit=settings.fusion_top_k)),
            ):
                started = time.perf_counter()
                call()
                timings[route].append((time.perf_counter() - started) * 1000)
    return {
        route: {
            "calls": len(values),
            "mean_ms": round(statistics.mean(values), 3),
            "p50_ms": round(percentile(values, 50), 3),
            "p95_ms": round(percentile(values, 95), 3),
            "p99_ms": round(percentile(values, 99), 3),
        }
        for route, values in timings.items()
    }


def db_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.parent.glob(path.name + "*") if p.is_file())


def checkpoint(store: EmbeddedStore) -> None:
    store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def table_sizes(store: EmbeddedStore) -> dict:
    try:
        rows = store._conn.execute(
            "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC"
        ).fetchall()
    except Exception as exc:  # dbstat is optional in some SQLite builds
        return {"error": str(exc)}
    return {name: int(size) for name, size in rows}


def scale_test(records: list[ChunkRecord], rows, vectors, settings: Settings, sizes: list[int]) -> list[dict]:
    rng = np.random.default_rng(20261023)
    base = np.asarray([r.dense for r in records], dtype=np.float32)
    results = []
    for size in sizes:
        db_path = WORK_DIR / f"scale_{size}.sqlite3"
        for leftover in WORK_DIR.glob(db_path.name + "*"):
            leftover.unlink()
        store = EmbeddedStore(settings, db_path=db_path)
        store.ensure_collection()
        insert_seconds = 0.0
        batch = 2000
        for start in range(0, size, batch):
            synthetic = []
            for i in range(start, min(size, start + batch)):
                source = records[i % len(records)]
                vector = base[i % len(base)] + rng.normal(0, 0.02, base.shape[1]).astype(np.float32)
                vector /= np.linalg.norm(vector)
                synthetic.append(
                    ChunkRecord(
                        id=f"syn{i:08d}",
                        doc_id=f"{i // 50:024x}",
                        parent_id=f"{i // 50:024x}-p{i % 5}",
                        source=f"synthetic_{i // 50}.txt",
                        heading=source.heading,
                        locator=source.locator,
                        text=f"{source.text}\n副本编号 {i}",
                        dense=vector.tolist(),
                    )
                )
            started = time.perf_counter()
            store.insert(synthetic)
            insert_seconds += time.perf_counter() - started
        checkpoint(store)
        sample = list(range(0, len(rows), max(1, len(rows) // 100)))[:100]
        profile = latency_profile(store, [rows[i] for i in sample], vectors[sample], settings, passes=1)
        open_started = time.perf_counter()
        reopened = EmbeddedStore(settings, db_path=db_path)
        reopened.search_dense(vectors[0].tolist(), limit=24)
        cold_ms = (time.perf_counter() - open_started) * 1000
        reopened.close()
        result = {
            "chunks": size,
            "insert_seconds": round(insert_seconds, 2),
            "insert_chunks_per_second": round(size / insert_seconds, 1),
            "db_bytes": db_size(db_path),
            "open_and_first_dense_query_ms": round(cold_ms, 1),
            "latency": profile,
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
        store.close()
        for leftover in WORK_DIR.glob(db_path.name + "*"):
            leftover.unlink()
    return results


def milvus_footprint() -> dict:
    info = {}
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", "milvusdb/milvus:v2.6.15", "quay.io/coreos/etcd:v3.5.18",
             "--format", "{{index .RepoTags 0}} {{.Size}}"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        info["docker_images_bytes"] = {line.split()[0]: int(line.split()[1]) for line in out.splitlines() if line}
    except Exception as exc:
        info["docker_images_error"] = str(exc)
    return info


def main() -> None:
    args = parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    settings = build_settings()
    embeddings = SiliconFlowEmbeddings(settings)
    reranker = SiliconFlowReranker(settings)

    bench = regenerate_corpus()
    records = chunk_and_embed(bench, settings, embeddings)
    rows, link_check = relink_labels(bench, records, args.dataset)
    vectors = query_vectors(rows, embeddings)
    print(f"chunks={len(records)} questions={len(rows)} link_check={link_check}", flush=True)

    report_path = SPIKE_DIR / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}

    if args.scale:
        sizes = [int(value) for value in args.scale.split(",") if value]
        report["scale_test"] = scale_test(records, rows, vectors, settings, sizes)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    embedded_path = WORK_DIR / "embedded.sqlite3"
    embedded = EmbeddedStore(settings, db_path=embedded_path)
    embedded.ensure_collection()
    stores = {"embedded": embedded}
    if not args.skip_milvus:
        milvus = MilvusStore(settings)
        milvus.ensure_collection()
        stores["milvus"] = milvus

    report.update(
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "dataset": args.dataset,
            "questions": len(rows),
            "chunks": len(records),
            "k": args.k,
            "candidate_k": settings.fusion_top_k,
            "link_check": link_check,
            "models": {"embedding": settings.embedding_model, "reranker": settings.rerank_model},
        }
    )
    report.setdefault("stores", {})
    for name, store in stores.items():
        index_seconds = index_store(store, records)
        if name == "milvus":
            time.sleep(2)  # Bounded consistency: let the freshly flushed segment become visible
        summary = run_eval(name, store, rows, vectors, reranker, settings, args.k, args.restart)
        entry = {
            "index_seconds": round(index_seconds, 3),
            "results": summary,
            "latency_profile": latency_profile(store, rows, vectors, settings, args.latency_passes),
        }
        if name == "embedded":
            checkpoint(store)
            entry["db_bytes"] = db_size(embedded_path)
            entry["table_bytes"] = table_sizes(store)
            started = time.perf_counter()
            reopened = EmbeddedStore(settings, db_path=embedded_path)
            reopened.search_hybrid(rows[0]["question"], vectors[0].tolist())
            entry["open_and_first_hybrid_query_ms"] = round((time.perf_counter() - started) * 1000, 1)
            reopened.close()
        else:
            entry["footprint"] = milvus_footprint()
        report["stores"][name] = entry
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({name: {m: {x: entry["results"][m][x] for x in ("recall", "mrr", "hit_rate")}
                                 for m in entry["results"]}}, indent=1), flush=True)
    if "milvus" in stores:
        stores["milvus"].client.drop_collection(MILVUS_COLLECTION)


if __name__ == "__main__":
    main()
