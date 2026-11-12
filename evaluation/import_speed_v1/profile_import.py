"""Import a folder into a fresh data directory with the real library service and time every stage.

    python -I evaluation/import_speed_v1/profile_import.py <corpus folder> <new data dir> <out.json>
        [--project project] [--env project/.env] [--image-mode local] [--fake-embed 0.15,0.007]
        [--no-cards] [--no-suggestions] [--drain 0] [--set key=value ...]

The library runs as in the app (``RAGSystem.start_library`` then ``LibraryService.import_paths``,
the call behind 添加文件夹), with its workers, queue, pauses and background builders. Stage times
are summed over all threads, so they add up to more than the wall time when work overlaps.

* ``--project`` is the code to measure (``project`` of this worktree, or of another checkout);
* ``--fake-embed a,b`` replaces the embedding service with a local stand-in that answers in
  ``a + b * texts`` seconds (measured on SiliconFlow, see api_bench.py) -- the app's own share;
* ``--drain S`` keeps running up to S seconds after the files are read, for doc cards and
  suggestions, and reports how far they got.

The API key comes from SILICONFLOW_API_KEY or ``--env``; it is never printed. Nothing outside
the new data directory is written; the keyring is a null one.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any


class Stages:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.extra: dict[str, float] = defaultdict(float)

    def add(self, stage: str, seconds: float, **extra: float) -> None:
        with self.lock:
            self.seconds[stage] += seconds
            self.calls[stage] += 1
            for key, value in extra.items():
                self.extra[f"{stage}.{key}"] += value

    def report(self) -> dict[str, Any]:
        return {
            stage: {"seconds": round(self.seconds[stage], 2), "calls": self.calls[stage],
                    **{k.split(".", 1)[1]: round(v, 2) for k, v in self.extra.items() if k.startswith(stage + ".")}}
            for stage in sorted(self.seconds)
        }


STAGES = Stages()


def timed(owner: Any, name: str, stage: str, extra=None) -> None:
    original = getattr(owner, name)
    if getattr(original, "_timed", False):
        return

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        failed = None
        try:
            return original(*args, **kwargs)
        except BaseException as exc:
            failed = exc
            raise
        finally:
            info = extra(args, kwargs) if extra else {}
            if failed is not None:
                code = getattr(failed, "status_code", None) or getattr(getattr(failed, "response", None), "status_code", None)
                info["errors"] = 1
                info[f"error_{code or type(failed).__name__}"] = 1
            STAGES.add(stage, time.perf_counter() - started, **info)

    wrapper._timed = True
    setattr(owner, name, wrapper)


def load_key(env_file: Path | None) -> None:
    if os.environ.get("SILICONFLOW_API_KEY"):
        return
    if env_file and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "SILICONFLOW_API_KEY":
                os.environ["SILICONFLOW_API_KEY"] = value.strip().strip('"').strip("'")


def rss_mb() -> float:
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        info = ctypes.windll.psapi.GetProcessMemoryInfo
        info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        info(current(), ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / 1e6
    except Exception:
        return 0.0


def fake_vector(text: str, dim: int) -> list[float]:
    import numpy as np

    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    vector = np.random.default_rng(seed).standard_normal(dim).astype("float32")
    return (vector / np.linalg.norm(vector)).tolist()


# Tokens bge-m3 counts per character of Chinese news (api_bench.py: ~370 tokens per 650 characters).
TOKENS_PER_CHAR = 0.57
FAKE_BURST = [0.0]
FAKE_CARD = [0.0]


class FakeLimit:
    """SiliconFlow's per-minute token limit as measured (api_bench.py): a burst of ``burst``
    tokens, refilled at ``per_minute``; past it a request gets 429 (TPM limit reached)."""

    def __init__(self, per_minute: float, burst: float | None = None):
        self.rate = per_minute / 60.0
        self.capacity = burst or per_minute
        self.level = self.capacity
        self.at = time.monotonic()
        self.lock = threading.Lock()

    def take(self, tokens: float) -> None:
        import httpx

        with self.lock:
            now = time.monotonic()
            self.level = min(self.capacity, self.level + (now - self.at) * self.rate)
            self.at = now
            if self.level < tokens:
                STAGES.add("embed.fake_429", 0.0)
                request = httpx.Request("POST", "https://api.siliconflow.cn/v1/embeddings")
                raise httpx.HTTPStatusError(
                    "429 TPM limit reached", request=request,
                    response=httpx.Response(429, request=request, text='{"message":"TPM limit reached."}'),
                )
            self.level -= tokens


def instrument(fake_embed: tuple[float, float] | None, fake_tpm: float = 0.0, fake_rerank: float = 0.0,
               no_suggestions: bool = False) -> None:
    from db.parent_store_manager import ParentStoreManager
    from ingestion import pipeline
    from ingestion.chunker import HeadingAwareChunker
    from ingestion.parsers import MultiFormatParser
    from library import service
    from library.registry import FileRegistry
    from retrieval.doc_cards import DocCardBuilder
    from retrieval.embedded_store import EmbeddedStore
    from retrieval.file_index import FileIndexer
    from services import model_clients
    from suggestions.service import SuggestionService

    def parse_kind(args, kwargs):
        suffix = Path(str(args[1])).suffix.lower()
        return {f"files_{suffix.lstrip('.')}": 1}

    # Parsing per type: the stage is "parse" and also "parse:<suffix>".
    original_parse = MultiFormatParser.parse

    @functools.wraps(original_parse)
    def parse(self, file_path, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_parse(self, file_path, *args, **kwargs)
        finally:
            seconds = time.perf_counter() - started
            STAGES.add("parse", seconds)
            STAGES.add("parse:" + Path(str(file_path)).suffix.lower().lstrip("."), seconds)

    MultiFormatParser.parse = parse
    timed(HeadingAwareChunker, "split_parent_child", "chunk")
    timed(EmbeddedStore, "_prepare", "sql.tokenize")
    timed(EmbeddedStore, "replace_document", "sql.replace_document (incl. tokenize)")
    timed(EmbeddedStore, "upsert_file_entries", "sql.file_entries")
    timed(EmbeddedStore, "upsert_doc_card", "sql.doc_card")
    timed(EmbeddedStore, "vectors_by_text", "sql.vectors_by_text")
    for name in ("write_many", "commit_batch"):
        if hasattr(EmbeddedStore, name):
            timed(EmbeddedStore, name, f"sql.{name}")
    timed(ParentStoreManager, "save_document", "parents.save")
    timed(pipeline.IngestionPipeline, "_commit", "commit (chunk+vectors+write)")
    timed(pipeline.IngestionPipeline, "_vectors", "vectors (reuse+embed)")
    timed(pipeline.IngestionPipeline, "index_document", "index_document")
    timed(FileIndexer, "index", "file_entry")
    timed(FileRegistry, "update_file", "registry.update_file")
    timed(service.LibraryService, "pause_for_questions", "pause_for_questions")
    timed(service.LibraryService, "_index", "job (_index)")
    timed(DocCardBuilder, "build", "doc_card.build")
    timed(SuggestionService, "build_file", "suggestion.build_file")
    timed(service, "sha256_file", "sha256")
    # The JSON snapshot written per file.
    pipeline.json = SimpleNamespace(**{n: getattr(json, n) for n in ("loads", "load", "dump", "JSONDecodeError")})
    pipeline.json.dumps = lambda *a, **k: _timed_dumps(*a, **k)

    def embed_extra(args, kwargs):
        texts = args[1]
        return {"texts": len(texts), "chars": sum(len(t) for t in texts)}

    if fake_embed:
        base, per_text = fake_embed
        bucket = FakeLimit(fake_tpm, FAKE_BURST[0] or None) if fake_tpm else None

        try:  # the pacer's HTTP hooks never see the stand-in: call them as they would be
            from ingestion import embed_pacing
        except ImportError:
            embed_pacing = None

        def fake_batch(self, texts):
            if embed_pacing is not None and embed_pacing.is_paced():
                embed_pacing.PACER.acquire(embed_pacing.estimate_tokens(texts))
            time.sleep(base + per_text * len(texts))
            if bucket is not None:
                try:
                    bucket.take(sum(len(t) for t in texts) * TOKENS_PER_CHAR + 4 * len(texts))
                except Exception:
                    if embed_pacing is not None:
                        embed_pacing.PACER.limited()
                    raise
            return [fake_vector(t, int(self.settings.embedding_dimension)) for t in texts]

        model_clients.SiliconFlowEmbeddings._embed_batch = fake_batch
    if fake_rerank:
        import random as _random

        def fake_rerank_call(self, query, documents, top_n):
            time.sleep(fake_rerank)
            rng = _random.Random(hash(query) & 0xFFFF)
            scores = sorted(((rng.random(), i) for i in range(len(documents))), reverse=True)[:top_n]
            return [model_clients.RerankItem(index=i, score=s) for s, i in scores]

        model_clients.SiliconFlowReranker._rerank = fake_rerank_call
    if FAKE_CARD[0]:
        # A document card without the chat model: the call's time, inside the same slot.
        def fake_ask(self, name, opening):
            with model_clients.background_calls.slot():
                time.sleep(FAKE_CARD[0])
            return {"label": "报告", "doc_type": "报告", "subject": "", "period": "", "summary": name[:20]}, 900

        DocCardBuilder._ask = fake_ask
    if no_suggestions:
        # The home-page builder stays out of the run (it builds per indexed file otherwise).
        SuggestionService.file_indexed = lambda self, file_id: None
        SuggestionService.start = lambda self: None
    timed(model_clients.SiliconFlowEmbeddings, "_embed_batch", "embed.http", embed_extra)

    original_slot = model_clients.CallLimiter.slot

    @contextmanager
    def slot(self):
        started = time.perf_counter()
        with original_slot(self):
            STAGES.add("embed.wait_for_slot", time.perf_counter() - started)
            yield

    model_clients.CallLimiter.slot = slot


def _timed_dumps(*args, **kwargs):
    started = time.perf_counter()
    try:
        return json.dumps(*args, **kwargs)
    finally:
        STAGES.add("parsed.json_dumps", time.perf_counter() - started)


def fingerprint(store, registry, corpus: Path) -> dict[str, Any]:
    """Per file (by path under the corpus): chunk count, a hash of the chunk texts in order,
    and the vectors (rounded) of up to 3 chunks -- to compare two runs."""
    import numpy as np

    by_id = {row["file_id"]: row for row in registry.list_files()}
    out: dict[str, Any] = {}
    with store._reading() as conn:
        rows = conn.execute(
            "SELECT c.doc_id, c.id, c.text, v.embedding FROM chunks c JOIN chunks_vec v ON v.rowid = c.rowid "
            "ORDER BY c.doc_id, c.rowid"
        ).fetchall()
    for doc_id, chunk_id, text, blob in rows:
        record = by_id.get(doc_id)
        if not record:
            continue
        key = str(Path(record["path"]).resolve().relative_to(corpus.resolve()))
        item = out.setdefault(key, {"chunks": 0, "texts": hashlib.sha256(), "vectors": []})
        item["chunks"] += 1
        item["texts"].update(chunk_id.split("-", 1)[-1].encode() + b"\0" + text.encode("utf-8") + b"\1")
        if len(item["vectors"]) < 3:
            item["vectors"].append([round(float(x), 4) for x in np.frombuffer(blob, dtype=np.float32)[:64]])
    return {k: {"chunks": v["chunks"], "texts": v["texts"].hexdigest(), "vectors": v["vectors"]} for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2] / "project")
    parser.add_argument("--env", type=Path)
    parser.add_argument("--image-mode", default="local")
    parser.add_argument("--fake-embed", default="")
    parser.add_argument("--fake-tpm", type=float, default=0.0, help="with --fake-embed: 429 past this many tokens a minute")
    parser.add_argument("--fake-burst", type=float, default=0.0)
    parser.add_argument("--fake-rerank", type=float, default=0.0, help="seconds per rerank call, no API")
    parser.add_argument("--no-cards", action="store_true")
    parser.add_argument("--fake-cards", type=float, default=0.0, help="seconds per document card, no model call")
    parser.add_argument("--no-suggestions", action="store_true")
    parser.add_argument("--drain", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--set", action="append", default=[], help="settings override, e.g. index_workers=6")
    parser.add_argument("--fingerprint", action="store_true")
    parser.add_argument("--answer-at", type=float, default=0.0,
                        help="seconds into the import: hold library.answering() for --answer-for seconds")
    parser.add_argument("--answer-for", type=float, default=15.0)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    project = args.project.resolve()
    sys.path.insert(0, str(project))
    os.chdir(project)
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    os.environ["RAG_DATA_DIR"] = str(args.data_dir.resolve())
    load_key(args.env)
    args.data_dir.mkdir(parents=True, exist_ok=False)
    prefs = {"followups_enabled": False}
    if args.no_suggestions:
        prefs["suggestions_enabled"] = False
    (args.data_dir / "preferences.json").write_text(json.dumps(prefs), encoding="utf-8")

    import logging

    logging.basicConfig(level=logging.INFO, filename=str(args.data_dir / "profile.log"),
                        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s")
    fake = tuple(float(v) for v in args.fake_embed.split(",")) if args.fake_embed else None
    FAKE_BURST[0] = args.fake_burst
    FAKE_CARD[0] = args.fake_cards
    instrument(fake, args.fake_tpm, args.fake_rerank, args.no_suggestions)

    from config import Settings
    from core.rag_system import RAGSystem

    overrides: dict[str, Any] = {"watch_enabled": False}
    if args.no_cards:
        overrides["doc_cards"] = False
    for item in args.set:
        name, _, value = item.partition("=")
        overrides[name] = json.loads(value) if value[:1] in "0123456789-[{tf\"" else value
    settings = Settings(**overrides)
    settings.ensure_directories()
    system = RAGSystem(settings)
    started = time.monotonic()
    system.start_library()
    library = system.library
    ready = time.monotonic() - started
    inspect_started = time.monotonic()
    inspected = library.inspect([str(args.corpus.resolve())])
    inspect_seconds = time.monotonic() - inspect_started
    began = time.monotonic()
    result = library.import_paths([str(args.corpus.resolve())], image_mode=args.image_mode)
    scan_done = None
    answer: dict[str, Any] = {}

    def hold_answer() -> None:
        time.sleep(args.answer_at)
        def requests() -> int:
            return STAGES.calls.get("embed.http", 0)
        with library.answering():
            answer.update(start=round(time.monotonic() - began, 1), requests_at_start=requests(),
                          indexed_at_start=library.registry.status_counts().get("indexed", 0))
            time.sleep(2.0)  # requests already sent finish
            answer.update(requests_after_2s=requests())
            time.sleep(max(0.0, args.answer_for - 2.0))
            answer.update(requests_at_end=requests(), indexed_at_end=library.registry.status_counts().get("indexed", 0))
        answer["end"] = round(time.monotonic() - began, 1)

    if args.answer_at:
        threading.Thread(target=hold_answer, daemon=True).start()
    progress: list[list[float]] = []
    first_indexed = None
    # [seconds, tokens sent so far by library jobs, pacer's tokens a minute] every 10 s.
    tokens_seen: list[list[float]] = []

    def _pacer_tokens() -> float:
        try:
            from ingestion import embed_pacing as pacing
            return pacing.PACER.stats["tokens"]
        except ImportError:
            return 0.0

    def _pacer_rate() -> float:
        try:
            from ingestion import embed_pacing as pacing
            return pacing.PACER.rate
        except ImportError:
            return 0.0
    peak_rss = 0.0
    while True:
        time.sleep(0.5)
        status = library.status()
        counts = status["counts"]
        indexed = counts.get("indexed", 0)
        elapsed = time.monotonic() - began
        if indexed and first_indexed is None:
            first_indexed = elapsed
        if scan_done is None and not status["scanning"]:
            scan_done = elapsed
        peak_rss = max(peak_rss, rss_mb())
        if not progress or progress[-1][1] != indexed or elapsed - progress[-1][0] >= 10:
            progress.append([round(elapsed, 1), indexed, status["queued"], status["running"]])
        if elapsed - (tokens_seen[-1][0] if tokens_seen else -10) >= 10:
            tokens_seen.append([round(elapsed), round(_pacer_tokens()), round(_pacer_rate())])
        if elapsed > 3 and not status["busy"]:
            break
        if elapsed > args.timeout:
            break
    wall = time.monotonic() - began
    during = {
        "suggestion_builds": STAGES.calls.get("suggestion.build_file", 0),
        "doc_cards_built": system.doc_cards.stats["built"] if system.doc_cards else 0,
    }
    counts = library.registry.status_counts()
    drained: dict[str, Any] = {}
    if args.drain:
        drain_start = time.monotonic()
        while time.monotonic() - drain_start < args.drain:
            pending_cards = system.doc_cards.pending() if system.doc_cards else 0
            building = system.suggestions.building if system.suggestions else False
            if not pending_cards and not building:
                break
            time.sleep(2)
        drained = {"seconds": round(time.monotonic() - drain_start, 1)}
    cards = dict(system.doc_cards.stats) if system.doc_cards else {}
    cards["pending"] = system.doc_cards.pending() if system.doc_cards else 0
    failed = [
        {"name": row["name"], "error": (row.get("error") or "")[:200]}
        for row in library.registry.list_files(statuses=("failed",))
    ]
    store = system.store
    log_text = (args.data_dir / "profile.log").read_text(encoding="utf-8", errors="ignore")
    locked = log_text.count("database is locked")
    try:
        from ingestion import embed_pacing as _pacing
        pacer = _pacing.PACER.snapshot()
    except ImportError:
        pacer = None
    report = {
        "pacer": pacer, "tokens_seen": tokens_seen, "answering": answer, "during_import": during, "database_locked_warnings": locked,
        "suggestions_built": STAGES.calls.get("suggestion.build_file", 0),
        "corpus": str(args.corpus), "project": str(project), "fake_embed": fake, "fake_tpm": args.fake_tpm,
        "fake_rerank": args.fake_rerank, "overrides": overrides,
        "image_mode": args.image_mode, "suggestions": not args.no_suggestions,
        "startup_seconds": round(ready, 2), "wall_seconds": round(wall, 1),
        "first_indexed_seconds": first_indexed and round(first_indexed, 1),
        "inspect_seconds": round(inspect_seconds, 2), "estimate": inspected.get("estimate"),
        "scan_done_seconds": scan_done and round(scan_done, 1),
        "files": counts, "chunks": store.count(), "failed": failed[:20],
        "stages": STAGES.report(), "doc_cards": cards, "drain": drained,
        "progress": progress, "peak_rss_mb": round(peak_rss), "import_result_added": result.get("added"),
    }
    for share in (0.5, 0.9, 1.0):
        total = counts.get("indexed", 0)
        target = share * total
        report[f"t_{int(share * 100)}pct_indexed"] = next((p[0] for p in progress if p[1] >= target), None)
    if args.fingerprint:
        report["fingerprint"] = fingerprint(store, library.registry, args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("progress", "stages", "fingerprint")}
    print(json.dumps(summary, ensure_ascii=False))
    print(json.dumps(report["stages"], ensure_ascii=False, indent=0))
    library.stop()
    os._exit(0)


if __name__ == "__main__":
    main()
