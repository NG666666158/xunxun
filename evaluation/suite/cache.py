"""Embedding and rerank caches, keyed by text, and model clients that read through them.

* ``VectorStore``: bge-m3 vectors by sha1(model, text) (queries and passages alike: the
  app embeds both with the same call).
* ``ScoreStore``: bge-reranker-v2-m3 scores by sha1(model, query, passage). The
  cross-encoder scores each (query, passage) pair on its own, so a pool can be scored
  from cached pairs plus one request for the new ones (``check_pair_independence`` in
  runner.py verifies this against the live API).

Both live in one SQLite file. ``CachedEmbeddings`` / ``CachedReranker`` stand in for the
app's SiliconFlow clients. With ``offline=True`` a cache miss is an error instead of a
request: re-runs and the CI tier never touch the network.

Fixtures (the CI tier) export a cache to a small JSON + npz pair (``export`` / ``load_export``).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Sequence

import numpy as np

from common import text_key


class CacheMiss(RuntimeError):
    pass


class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, model TEXT NOT NULL, score REAL NOT NULL);
            """
        )

    # ---- vectors
    def get_vectors(self, model: str, texts: Sequence[str]) -> dict[str, np.ndarray]:
        keys = list(dict.fromkeys(text_key(model, t) for t in texts))
        found: dict[str, np.ndarray] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                batch = keys[start:start + 500]
                marks = ",".join("?" * len(batch))
                for key, blob in self.conn.execute(f"SELECT key, vector FROM vectors WHERE key IN ({marks})", batch):
                    found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_vectors(self, model: str, items: Sequence[tuple[str, Sequence[float]]]) -> None:
        rows = [(text_key(model, t), model, np.asarray(v, dtype=np.float32).tobytes()) for t, v in items]
        with self._lock:
            self.conn.execute("BEGIN")
            self.conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)", rows)
            self.conn.execute("COMMIT")

    # ---- rerank scores
    @staticmethod
    def pair_key(model: str, query: str, text: str) -> str:
        return text_key(model, f"{query}\n\x1f\n{text}")

    def get_scores(self, model: str, query: str, texts: Sequence[str]) -> dict[str, float]:
        keys = list(dict.fromkeys(self.pair_key(model, query, t) for t in texts))
        found: dict[str, float] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                batch = keys[start:start + 500]
                marks = ",".join("?" * len(batch))
                for key, score in self.conn.execute(f"SELECT key, score FROM scores WHERE key IN ({marks})", batch):
                    found[key] = float(score)
        return found

    def put_scores(self, model: str, query: str, items: Sequence[tuple[str, float]]) -> None:
        rows = [(self.pair_key(model, query, t), model, float(s)) for t, s in items]
        with self._lock:
            self.conn.execute("BEGIN")
            self.conn.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?, ?)", rows)
            self.conn.execute("COMMIT")

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {table: self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("vectors", "scores")}

    def close(self) -> None:
        self.conn.close()

    # ---- fixtures: a cache subset as files that can be committed
    def export(self, folder: Path, vector_keys: set[str], score_keys: set[str]) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        with self._lock:
            vectors = [(k, b) for k, b in self.conn.execute("SELECT key, vector FROM vectors") if k in vector_keys]
            scores = {k: s for k, s in self.conn.execute("SELECT key, score FROM scores") if k in score_keys}
        missing = (vector_keys - {k for k, _ in vectors}) | (score_keys - set(scores))
        if missing:
            raise CacheMiss(f"{len(missing)} entries are not in the cache")
        vectors.sort()
        np.savez_compressed(folder / "vectors.npz", keys=np.array([k for k, _ in vectors]),
                            vectors=np.stack([np.frombuffer(b, dtype=np.float32) for _, b in vectors]))
        (folder / "scores.json").write_text(json.dumps(dict(sorted(scores.items())), indent=0), encoding="utf-8")

    def load_export(self, folder: Path, model_vectors: str, model_scores: str) -> None:
        data = np.load(folder / "vectors.npz")
        rows = [(str(k), model_vectors, np.asarray(v, dtype=np.float32).tobytes()) for k, v in zip(data["keys"], data["vectors"])]
        scores = json.loads((folder / "scores.json").read_text(encoding="utf-8"))
        with self._lock:
            self.conn.execute("BEGIN")
            self.conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)", rows)
            self.conn.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?, ?)",
                                  [(k, model_scores, float(s)) for k, s in scores.items()])
            self.conn.execute("COMMIT")


class TokenBudget:
    """Client-side tokens-per-minute budget for live reranker calls (SiliconFlow rejects
    requests over the account's TPM with 429; waiting on our side keeps the measured call
    latency free of rejected attempts). EVAL_RERANK_TPM, default 400k."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._lock = threading.Lock()
        self._spent: list[tuple[float, int]] = []
        self.ratio = 1.0  # actual / estimated tokens, learned from responses

    def estimate(self, query: str, documents: Sequence[str]) -> int:
        def tokens(text: str) -> float:
            cjk = sum(1 for ch in text if "㐀" <= ch <= "鿿")
            return cjk + (len(text) - cjk) / 3.5

        return int(sum(tokens(d) + tokens(query) + 4 for d in documents) * self.ratio) + 1

    def acquire(self, tokens: int) -> None:
        tokens = min(tokens, self.per_minute)
        while True:
            with self._lock:
                now = time.monotonic()
                self._spent = [(t, n) for t, n in self._spent if now - t < 60.0]
                if sum(n for _, n in self._spent) + tokens <= self.per_minute:
                    self._spent.append((now, tokens))
                    return
                wait = 60.0 - (now - self._spent[0][0]) + 0.05
            time.sleep(max(0.05, wait))

    def settle(self, estimated: int, actual: int) -> None:
        with self._lock:
            self.ratio = 0.8 * self.ratio + 0.2 * (actual / max(1, estimated / self.ratio))
            if self._spent:
                t, n = self._spent[-1]
                self._spent[-1] = (t, max(0, n + actual - estimated))


RERANK_BUDGET = TokenBudget(int(os.environ.get("EVAL_RERANK_TPM", "400000")))


@dataclass
class CallLog:
    """Live (network) calls made through a cached client, for latency and cost."""

    ms: float = 0.0
    calls: int = 0
    items: int = 0


class CachedEmbeddings:
    """``SiliconFlowEmbeddings`` through the cache (embed_query / embed_documents / batches)."""

    def __init__(self, cache: Cache, settings, offline: bool = False, batch_size: int = 32):
        self.cache = cache
        self.model = settings.embedding_model
        self.settings = settings
        self.offline = offline
        self.batch_size = batch_size
        self._inner = None
        self._local = threading.local()
        self.live = CallLog()

    @property
    def inner(self):
        if self._inner is None:
            from services.model_clients import SiliconFlowEmbeddings

            self._inner = SiliconFlowEmbeddings(self.settings, batch_size=self.batch_size)
        return self._inner

    def last_call(self) -> CallLog:
        """Live calls made by this thread since the last ``reset_thread``."""
        if not hasattr(self._local, "log"):
            self._local.log = CallLog()
        return self._local.log

    def reset_thread(self) -> None:
        self._local.log = CallLog()

    def _fetch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            for attempt in range(8):
                started = time.perf_counter()
                try:
                    out = self.inner._embed_batch(batch)
                    break
                except Exception as exc:
                    if attempt == 7:
                        raise
                    print(f"embedding retry {attempt + 1} ({type(exc).__name__}: {str(exc)[:80]})", flush=True)
                    time.sleep(min(60, 2 * 2 ** attempt))
            ms = (time.perf_counter() - started) * 1000
            for log in (self.live, self.last_call()):
                log.ms += ms
                log.calls += 1
                log.items += len(batch)
            vectors.extend(out)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        found = self.cache.get_vectors(self.model, texts)
        missing = list(dict.fromkeys(t for t in texts if text_key(self.model, t) not in found))
        if missing:
            if self.offline:
                raise CacheMiss(f"{len(missing)} texts have no cached vector (offline)")
            fresh = self._fetch(missing)
            self.cache.put_vectors(self.model, list(zip(missing, fresh, strict=True)))
            found.update({text_key(self.model, t): np.asarray(v, dtype=np.float32) for t, v in zip(missing, fresh)})
        return [found[text_key(self.model, t)].tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed_batch(self, texts):  # used by helpers that batch themselves
        return self.embed_documents(list(texts))


class CachedReranker:
    """``SiliconFlowReranker`` through the cache: cached pairs are reused, the rest are scored
    in one request. Rate limits are waited out; a failure raises (the app would fall back to
    the RRF order, which is not what is measured)."""

    def __init__(self, cache: Cache, settings, offline: bool = False):
        self.cache = cache
        self.model = settings.rerank_model
        self.settings = settings
        self.offline = offline
        self._inner = None
        self._local = threading.local()
        self.live = CallLog()

    @property
    def inner(self):
        if self._inner is None:
            from services.model_clients import SiliconFlowReranker

            self._inner = SiliconFlowReranker(self.settings)
        return self._inner

    def last_call(self) -> CallLog:
        if not hasattr(self._local, "log"):
            self._local.log = CallLog()
        return self._local.log

    def reset_thread(self) -> None:
        self._local.log = CallLog()

    def score_live(self, query: str, documents: list[str]) -> list[float]:
        estimate = RERANK_BUDGET.estimate(query, documents)
        for attempt in range(12):
            RERANK_BUDGET.acquire(estimate)
            started = time.perf_counter()
            try:
                response = self.inner.client.post(
                    self.inner.url,
                    auth=self.inner.auth,
                    json={"model": self.model, "query": query, "documents": documents,
                          "top_n": len(documents), "return_documents": False},
                )
                response.raise_for_status()
                break
            except Exception as exc:
                if attempt == 11:
                    raise
                print(f"rerank retry {attempt + 1} ({type(exc).__name__}: {str(exc)[:80]})", flush=True)
                time.sleep(min(60, 5 * 2 ** min(attempt, 3)))
        ms = (time.perf_counter() - started) * 1000
        used = ((response.json().get("meta") or {}).get("tokens") or {}).get("input_tokens")
        if used:
            RERANK_BUDGET.settle(estimate, int(used))
        for log in (self.live, self.last_call()):
            log.ms += ms
            log.calls += 1
            log.items += len(documents)
        scores = [0.0] * len(documents)
        seen = set()
        for row in response.json().get("results", []):
            scores[int(row["index"])] = float(row["relevance_score"])
            seen.add(int(row["index"]))
        if len(seen) != len(documents):
            raise RuntimeError(f"reranker scored {len(seen)} of {len(documents)} passages")
        return scores

    def scores(self, query: str, documents: Sequence[str]) -> list[float]:
        documents = list(documents)
        found = self.cache.get_scores(self.model, query, documents)
        missing = list(dict.fromkeys(d for d in documents if Cache.pair_key(self.model, query, d) not in found))
        if missing:
            if self.offline:
                raise CacheMiss(f"{len(missing)} passages have no cached rerank score (offline)")
            fresh = self.score_live(query, missing)
            self.cache.put_scores(self.model, query, list(zip(missing, fresh, strict=True)))
            found.update({Cache.pair_key(self.model, query, d): s for d, s in zip(missing, fresh)})
        return [found[Cache.pair_key(self.model, query, d)] for d in documents]

    def rerank(self, query: str, documents: Sequence[str], top_n: int):
        from services.model_clients import RerankItem

        documents = list(documents)
        if not documents:
            return []
        scores = self.scores(query, documents)
        # The API returns results best first; equal scores keep the input order.
        order = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
        return [RerankItem(index=i, score=scores[i]) for i in order[: min(top_n, len(documents))]]
