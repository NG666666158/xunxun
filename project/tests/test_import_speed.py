"""Importing a big library: embedding kept to the account's token limit, a busy service waited
out per batch, optional work after the text, bounded memory, and an estimate that matches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from db.parent_store_manager import ParentStoreManager
from ingestion import embed_pacing
from ingestion.embed_pacing import EmbedPacer, estimate_tokens
from ingestion.models import ParentChunk
from library import estimate
from library.registry import FileRegistry
from library.service import BULK_FILES, LibraryService


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.slept = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.slept += seconds


def _pacer(clock, **kwargs):
    return EmbedPacer(clock=clock, sleep=clock.sleep, **kwargs)


# ---------------------------------------------------------------- pacing


def test_tokens_are_estimated_per_script():
    assert estimate_tokens(["中" * 100]) == pytest.approx(62)
    assert estimate_tokens(["a" * 100]) == pytest.approx(32)
    assert estimate_tokens([]) == 0


def test_a_rested_bucket_goes_at_once_then_the_rate_holds():
    clock = Clock()
    # 1,000 tokens a second, 30,000 at once; the ceiling keeps it from growing meanwhile.
    pacer = _pacer(clock, per_minute=60_000, burst=30_000, ceiling=60_000)
    for _ in range(30):
        assert pacer.acquire(1_000) == 0  # the full bucket
    sent = 30_000
    while clock.now < 1000 + 60:
        pacer.acquire(1_000)
        sent += 1_000
    assert 30_000 + 59_000 <= sent <= 30_000 + 61_000
    # Rested again (a few minutes without requests): the whole bucket is back.
    clock.now += 600
    assert pacer.available_burst() == pytest.approx(30_000)


def test_a_429_cuts_the_rate_once_per_burst_and_holds_the_requests():
    clock = Clock()
    pacer = _pacer(clock, per_minute=500_000, burst=0)
    for _ in range(4):  # four requests in flight all come back 429
        pacer.limited()
    assert pacer.rate == pytest.approx(400_000)
    assert pacer.snapshot()["limited"] == 4 and pacer.snapshot()["cuts"] == 1
    assert pacer.acquire(10) >= EmbedPacer.HOLD_SECONDS
    clock.now += 10
    pacer.limited()
    assert pacer.rate == pytest.approx(320_000)


def test_the_rate_grows_only_while_the_library_uses_it():
    clock = Clock()
    pacer = _pacer(clock, per_minute=100_000, burst=0, ceiling=150_000)
    clock.now += 600  # idle: no growth without demand
    pacer.acquire(1)
    assert pacer.rate == pytest.approx(100_000)
    while clock.now < 1600 + 600:
        pacer.acquire(5_000)
    assert pacer.rate == pytest.approx(150_000)  # grown to the ceiling, not past it


def test_a_rate_the_library_barely_uses_does_not_grow():
    clock = Clock()
    pacer = _pacer(clock, per_minute=100_000, burst=0)
    for _ in range(60):  # a fifth of the rate, for ten minutes
        pacer.acquire(3_500)
        clock.now += 10
    assert pacer.rate == pytest.approx(100_000)


def test_only_a_proven_rate_is_kept_for_the_next_start(tmp_path):
    memory = tmp_path / "cache" / "embed_rate.json"
    clock = Clock()
    first = _pacer(clock, per_minute=100_000, burst=0, ceiling=1_000_000)
    first.remember(memory)
    while clock.now < 1000 + 300:  # five busy minutes: faster, but maybe on the service's bucket
        first.acquire(5_000)
    assert first.rate > 150_000 and first.known == 100_000 and not memory.exists()
    while clock.now < 1000 + 900:  # ten more without a 429: the account's own
        first.acquire(5_000)
    kept = json.loads(memory.read_text(encoding="utf-8"))["tokens_per_minute"]
    assert first.known > 150_000 and kept == pytest.approx(first.known, abs=1)
    first.limited()  # a 429: the cut rate is what the account takes
    assert json.loads(memory.read_text(encoding="utf-8"))["tokens_per_minute"] == pytest.approx(first.rate, abs=1)
    second = _pacer(Clock(), per_minute=100_000, burst=0)
    second.remember(memory)
    assert second.rate == second.known == pytest.approx(first.rate, abs=1)
    memory.write_text("not json", encoding="utf-8")
    third = _pacer(Clock(), per_minute=100_000)
    third.remember(memory)
    assert third.rate == third.known == 100_000


def test_only_library_requests_wait_and_any_429_slows_them(monkeypatch):
    clock = Clock()
    pacer = _pacer(clock, per_minute=60_000, burst=0)
    calls = []
    monkeypatch.setattr(pacer, "acquire", lambda tokens: calls.append(tokens) or 0.0)
    statuses = iter([200, 200, 429])

    def handler(request):
        return httpx.Response(next(statuses), json={"data": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert embed_pacing.install(client, pacer) is True
    assert embed_pacing.install(client, pacer) is False  # once per client
    client.post("https://api.example/v1/embeddings", json={"input": ["中文" * 10]})  # a question's
    assert calls == []
    with embed_pacing.paced():
        client.post("https://api.example/v1/embeddings", json={"input": ["中文" * 10, "abc"]})
        assert calls == [pytest.approx(estimate_tokens(["中文" * 10, "abc"]))]
        client.post("https://api.example/v1/embeddings", json={"input": "x"})
    assert pacer.snapshot()["limited"] == 1
    assert not embed_pacing.is_paced()


# ---------------------------------------------------------------- a busy service, per batch


pytest.importorskip("sqlite_vec")
pytest.importorskip("jieba")

from config import Settings  # noqa: E402
from ingestion.pipeline import IngestionPipeline  # noqa: E402
from retrieval.embedded_store import EmbeddedStore  # noqa: E402


def _busy():
    request = httpx.Request("POST", "https://api.example/v1/embeddings")
    return httpx.HTTPStatusError("429", request=request, response=httpx.Response(429, request=request))


class FlakyEmbeddings:
    """Fails the given request numbers (1-based) with ``error``; records every request."""

    def __init__(self, failing=(), error=_busy):
        self.failing = set(failing)
        self.error = error
        self.requests: list[int] = []

    def embed_documents(self, texts):
        self.requests.append(len(texts))
        if len(self.requests) in self.failing:
            raise self.error()
        return [[byte / 255 for byte in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]


def _pipeline(tmp_path, embeddings):
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", embedding_dimension=8,
                        parent_chunk_size=400, child_chunk_size=120, child_chunk_overlap=20)
    settings.ensure_directories()
    store = EmbeddedStore(settings)
    store.ensure_collection()
    pipeline = IngestionPipeline(store, embeddings, settings, parent_store=ParentStoreManager(settings))
    pipeline.embed_batch = 8
    pipeline.embed_patience = (0.0, 0.0)
    return pipeline, store


def _long_file(tmp_path):
    path = tmp_path / "汇编.md"
    path.write_text("# 汇编\n\n" + "\n\n".join(f"第{i}条 出差住宿按城市等级执行，编号 R-{i:03d}。" for i in range(60)),
                    encoding="utf-8")
    return path


def test_a_batch_refused_twice_is_tried_again_without_losing_the_others(tmp_path):
    embeddings = FlakyEmbeddings(failing={3, 4})
    pipeline, store = _pipeline(tmp_path, embeddings)
    pauses = []
    chunks, _ = pipeline.index_document(_long_file(tmp_path), "a" * 32, "汇编.md", pause=lambda: pauses.append(1))
    batches = -(-chunks // 8)
    assert len(embeddings.requests) == batches + 2  # only the refused batch was sent again
    assert store.count() == chunks and pauses
    store.close()


def test_a_service_that_stays_busy_or_a_refused_key_still_fails_the_file(tmp_path):
    pipeline, store = _pipeline(tmp_path, FlakyEmbeddings(failing={1, 2, 3}))
    with pytest.raises(httpx.HTTPStatusError):
        pipeline.index_document(_long_file(tmp_path), "a" * 32, "汇编.md")
    refused = FlakyEmbeddings(failing={1}, error=lambda: ValueError("401 bad key"))
    pipeline.embeddings = refused
    with pytest.raises(ValueError):
        pipeline.index_document(_long_file(tmp_path), "b" * 32, "汇编.md")
    assert refused.requests == [8]  # not waited on
    assert store.count() == 0
    store.close()


# ---------------------------------------------------------------- optional work after the text


def _library(tmp_path):
    settings = SimpleNamespace(
        allowed_extensions=(".md",), max_local_file_mb=1, max_local_text_mb=1, data_dir=tmp_path,
        upload_dir=tmp_path / "uploads", parsed_dir=tmp_path / "parsed", parent_store_dir=tmp_path / "parents",
        index_yield_max_seconds=5,
    )
    return LibraryService(settings, FileRegistry(tmp_path / "registry.sqlite3"), store=None,
                          parent_store=None, indexer=None)


def test_document_cards_and_suggestions_wait_for_a_big_batch(tmp_path):
    library = _library(tmp_path)
    for n in range(BULK_FILES + 5):
        library._queue.add(f"{n:032x}", cost=1.0)
    assert library.importing()
    done = threading.Event()
    threading.Thread(target=lambda: (library.pause_for_background(), done.set()), daemon=True).start()
    assert not done.wait(0.5)
    for n in range(10):
        library._queue.discard(f"{n:032x}")
    with library._wake:
        library._wake.notify_all()
    assert done.wait(10)
    # A few changed files (the usual case) hold nothing up.
    started = time.monotonic()
    library.pause_for_background()
    assert time.monotonic() - started < 0.5
    library.stop()
    library.registry.close()


def test_suggestions_switched_off_make_no_searches(tmp_path):
    from suggestions import privacy
    from suggestions.service import SuggestionService
    from suggestions.store import SuggestionStore

    settings = SimpleNamespace(data_dir=tmp_path, parsed_dir=tmp_path / "parsed")
    settings.parsed_dir.mkdir()
    registry = FileRegistry(tmp_path / "knowledge.sqlite3")
    store = SuggestionStore(tmp_path / "knowledge.sqlite3")
    searched = []
    retriever = SimpleNamespace(embeddings=None, retrieve=lambda q, **k: searched.append(q) or [])
    service = SuggestionService(settings, registry, store, retriever=retriever, file_search=None)
    record = registry.insert_file(str(tmp_path / "资料" / "电池报告.pdf"), root_id=None, status="indexed",
                                  indexed_sha256="v1")
    (settings.parsed_dir / f"{record['file_id']}.json").write_text(
        json.dumps({"source": record["name"], "units": [{"heading": "技术参数", "text": "循环寿命"}]},
                   ensure_ascii=False), encoding="utf-8")
    privacy.save_prefs(settings, enabled=False)
    assert service.build_file(record["file_id"]) == 0 and searched == []
    privacy.save_prefs(settings, enabled=True)
    service.sync()
    service.run_pending()
    assert searched  # switched back on: built
    store.close()
    registry.close()


# ---------------------------------------------------------------- memory


def test_parents_of_indexed_files_do_not_stay_in_memory(tmp_path):
    settings = SimpleNamespace(parent_store_dir=tmp_path / "parents")
    parents = ParentStoreManager(settings)
    parents.cache_documents = 3
    for n in range(10):
        doc_id = f"{n:032x}"
        parents.save_document(doc_id, [ParentChunk(f"{doc_id}-p0", doc_id, "a.md", "", "", "t")])
    assert len(parents._cache) == 0
    for n in range(10):
        assert parents.get(f"{n:032x}-p0")["text"] == "t"
    assert list(parents._cache) == [f"{n:032x}" for n in (7, 8, 9)]


# ---------------------------------------------------------------- the estimate


def test_the_estimate_is_the_slowest_stage_not_the_sum():
    # The folder from the report: 9,347 files, ~100 million characters (what the old estimate
    # assumed when it said 7.5 hours).
    big = estimate.batch_seconds(9347, 100_000_000, 0.0, tokens_per_minute=450_000, workers=4)
    assert big["bound"] == "embedding"
    assert 2.5 * 3600 < big["seconds"] < 3.3 * 3600
    # Many small files: requests, not tokens, are the limit.
    small = estimate.batch_seconds(5000, 2_000_000, 0.0, tokens_per_minute=4_000_000, workers=4)
    assert small["bound"] == "requests"
    # Images are read beside the text: the longer of the two counts, not both.
    with_images = estimate.batch_seconds(400, 3_700_000, 125.0, tokens_per_minute=450_000)
    text_only = estimate.batch_seconds(400, 3_700_000, 0.0, tokens_per_minute=450_000)
    assert with_images["seconds"] == text_only["seconds"]
    # A rested account sends its first ~1 million tokens at once.
    rested = estimate.batch_seconds(400, 3_700_000, 0.0, tokens_per_minute=450_000, burst_tokens=1_000_000)
    assert rested["seconds"] < text_only["seconds"] - 100
    assert estimate.batch_seconds(0, 0, 0.0, tokens_per_minute=450_000)["seconds"] == 0


# ---------------------------------------------------------------- recognition beside the text


def test_the_queue_counts_its_jobs_by_kind():
    from library.jobs import LIGHT, LONG, OCR, JobQueue

    queue = JobQueue()
    queue.add("a", 0.5)
    queue.add("b", 9.0, OCR)
    queue.add("c", 60.0, LONG)
    assert queue.has((OCR, LONG)) and queue.kinds() == {LIGHT: 1, OCR: 1, LONG: 1}
    queue.add("b", 9.0, OCR, urgent=True)  # moved forward: still one job
    assert queue.kinds()[OCR] == 1
    assert queue.pop(lambda job: job.kind == OCR).file_id == "b"
    queue.discard("c")
    assert not queue.has((OCR, LONG)) and queue.kinds() == {LIGHT: 1}
    assert queue.pop().file_id == "a" and not queue.kinds()
