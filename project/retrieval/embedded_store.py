"""Embedded (serverless) chunk store: SQLite FTS5 for BM25 + sqlite-vec for dense search.

Default chunk store (``VECTOR_BACKEND=embedded``). It mirrors the public interface of
``MilvusStore`` so the retriever and ingestion pipeline run without Docker/Milvus. Everything lives in
one SQLite file, so a chunk's text, BM25 postings and vector are written and deleted
in the same transaction.

Chinese text is segmented with jieba before it reaches FTS5; the FTS5 ``unicode61``
tokenizer then only splits on the spaces we insert (and on punctuation, which matches
how jieba separates codes such as ``SEC-05-ZX11``).

Writes go through one connection, one transaction at a time. Searches and other reads use
read-only connections of their own: under WAL a reader sees the last committed state and
never waits for a write, so re-indexing a large file does not hold up an answer.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from config import Settings, get_settings
from db.read_pool import ReaderPool
from .types import ChunkRecord, SearchHit, text_key

_TOKEN_CHARS = re.compile(r"[0-9A-Za-z㐀-鿿豈-﫿]")
# 32 hex: registry file_id (UUID); 24 hex: legacy name+content hash.
_DOC_ID = re.compile(r"[a-f0-9]{32}|[a-f0-9]{24}")
_CHUNK_COLUMNS = "c.id, c.doc_id, c.parent_id, c.source, c.heading, c.locator, c.text"

_jieba_lock = threading.Lock()
_jieba = None
_jieba_cache_dir: Path | None = None


def set_jieba_cache_dir(path: str | Path | None) -> None:
    """Keep jieba's ~9 MB dictionary cache in the app data dir instead of the system temp dir.

    Only takes effect before the first tokenization initializes jieba.
    """
    global _jieba_cache_dir
    _jieba_cache_dir = Path(path) if path else None


def _segmenter():
    global _jieba
    if _jieba is None:
        with _jieba_lock:
            if _jieba is None:
                import jieba

                jieba.setLogLevel(60)
                if _jieba_cache_dir is not None:
                    _jieba_cache_dir.mkdir(parents=True, exist_ok=True)
                    jieba.dt.tmp_dir = str(_jieba_cache_dir)
                jieba.initialize()
                _jieba = jieba
    return _jieba


def tokenize(text: str) -> list[str]:
    """jieba search-mode segmentation, dropping whitespace/punctuation-only tokens."""
    return [
        token.lower()
        for token in _segmenter().lcut_for_search(text)
        if _TOKEN_CHARS.search(token)
    ]


def build_match_query(query: str) -> str:
    """OR every distinct query token, mirroring Milvus BM25's sum over query terms."""
    tokens = dict.fromkeys(tokenize(query))
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


class EmbeddedStore:
    """Drop-in replacement for ``MilvusStore`` backed by a single SQLite file."""

    backend = "embedded"

    # Read-only connections, opened as reads overlap (an answer while the suggestion builder
    # or an indexer reads). Each keeps its own page cache (up to ``cache_mb``).
    read_connections = 2
    # The write connection only touches the pages it changes.
    write_cache_mb = 64
    # Address space mapped by each read connection (SQLite caps it at its compile-time maximum).
    read_mmap_mb = 2048

    def __init__(
        self,
        settings: Settings | None = None,
        db_path: str | Path | None = None,
        cache_mb: int = 512,
    ):
        self.settings = settings or get_settings()
        self.cache_mb = cache_mb
        self.collection_name = self.settings.milvus_collection
        self.db_path = Path(db_path) if db_path else Path(self.settings.data_dir) / f"{self.collection_name}.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_dir := getattr(self.settings, "cache_dir", None):
            set_jieba_cache_dir(cache_dir)
        # Held for every use of the write connection, transactions included.
        self._lock = threading.RLock()
        self._conn = self._connect()
        self._readers = ReaderPool(lambda: self._connect(read_only=True), self.read_connections)

    def _connect(self, read_only: bool = False) -> sqlite3.Connection:
        import sqlite_vec

        conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        if read_only:
            conn.execute("PRAGMA query_only=ON")
            # Brute-force KNN rereads every vector page per query. With the default 2 MB page
            # cache that is ~5x slower (spike scale test). The cache only grows as pages are
            # read, so this is an upper bound (~4 KB/chunk).
            conn.execute(f"PRAGMA cache_size=-{int(self.cache_mb) * 1024}")
            # A commit on any other connection (indexing, a registry update) empties this
            # cache. Pages already checkpointed into the database file are then mapped
            # instead of copied in again: a KNN right after a commit took 58 ms without the
            # map and 20 ms with it (3,900 chunks; 8 ms warm either way).
            conn.execute(f"PRAGMA mmap_size={self.read_mmap_mb * 1024 * 1024}")
            return conn
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA cache_size=-{min(int(self.cache_mb), self.write_cache_mb) * 1024}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _reading(self, snapshot: bool = False) -> Iterator[sqlite3.Connection]:
        """A read-only connection for one read. ``snapshot``: its statements all see the
        same committed state (a read transaction), e.g. the two routes of a hybrid search."""
        with self._readers.connection() as conn:
            if not snapshot:
                yield conn
                return
            conn.execute("BEGIN")
            try:
                yield conn
            finally:
                if conn.in_transaction:
                    conn.execute("COMMIT")

    def close(self) -> None:
        self._readers.close()
        with self._lock:
            self._conn.close()

    def ensure_collection(self) -> None:
        dim = int(self.settings.embedding_dimension)
        with self._lock:
            self._conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    rowid INTEGER PRIMARY KEY,
                    id TEXT NOT NULL UNIQUE,
                    doc_id TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    heading TEXT NOT NULL,
                    locator TEXT NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    tokens, tokenize = 'unicode61 remove_diacritics 0'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                    embedding float[{dim}] distance_metric=cosine
                );
                CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

                -- One row per file (name + short summary) for "find a file" questions.
                -- Kept apart from chunks so answers never retrieve or cite a summary.
                CREATE TABLE IF NOT EXISTS file_entries (
                    rowid INTEGER PRIMARY KEY,
                    doc_id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    version INTEGER NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS file_entries_fts USING fts5(
                    tokens, tokenize = 'unicode61 remove_diacritics 0'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS file_entries_vec USING vec0(
                    embedding float[{dim}] distance_metric=cosine
                );

                -- What kind of document each file is (retrieval/doc_cards.py), written by the
                -- chat model from the file name and opening; matched as a field, not searched.
                CREATE TABLE IF NOT EXISTS doc_cards (
                    doc_id TEXT PRIMARY KEY,
                    doc_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    period TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    tokens INTEGER NOT NULL DEFAULT 0,
                    model TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS doc_cards_type ON doc_cards(doc_type);
                """
            )
            row = self._conn.execute("SELECT value FROM store_meta WHERE key = 'embedding_dimension'").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO store_meta(key, value) VALUES ('embedding_dimension', ?)", (str(dim),)
                )
            elif int(row[0]) != dim:
                raise ValueError(f"Store was built with dimension {row[0]}, configured {dim}")

    def insert(self, records: Iterable[ChunkRecord]) -> int:
        rows = self._prepare(records)
        if not rows:
            return 0
        with self._lock, self._transaction():
            self._insert_rows(rows)
        return len(rows)

    def replace_document(self, doc_id: str, records: Iterable[ChunkRecord]) -> int:
        """``delete_document`` then ``insert``, in one transaction: a search sees the old
        version or the new one, never the document missing."""
        if not _DOC_ID.fullmatch(doc_id):
            raise ValueError("Invalid document id")
        rows = self._prepare(records)
        with self._lock, self._transaction():
            # The card stays: it costs a model call, and the card builder asks again only when
            # the file's name or opening changed (its input hash).
            self._delete_document(doc_id, keep_card=True)
            self._insert_rows(rows)
        return len(rows)

    def _prepare(self, records: Iterable[ChunkRecord]) -> list[tuple[dict[str, Any], str, bytes]]:
        """Rows with their FTS tokens and vector bytes, worked out before the write starts."""
        dim = int(self.settings.embedding_dimension)
        prepared = []
        for record in records:
            row = asdict(record)
            vector = np.asarray(row["dense"], dtype=np.float32)
            if vector.shape != (dim,):
                raise ValueError(f"Embedding dimension mismatch: configured {dim}, received {vector.shape}")
            prepared.append((row, " ".join(tokenize(row["text"])), vector.tobytes()))
        return prepared

    def _insert_rows(self, rows: list[tuple[dict[str, Any], str, bytes]]) -> None:
        for row, token_text, vector in rows:
            existing = self._conn.execute("SELECT rowid FROM chunks WHERE id = ?", (row["id"],)).fetchone()
            if existing:
                self._delete_rowids([existing[0]])
            cursor = self._conn.execute(
                "INSERT INTO chunks(id, doc_id, parent_id, source, heading, locator, text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (row["id"], row["doc_id"], row["parent_id"], row["source"], row["heading"], row["locator"], row["text"]),
            )
            rowid = cursor.lastrowid
            self._conn.execute("INSERT INTO chunks_fts(rowid, tokens) VALUES (?, ?)", (rowid, token_text))
            self._conn.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (rowid, vector))

    def delete_document(self, doc_id: str) -> None:
        if not _DOC_ID.fullmatch(doc_id):
            raise ValueError("Invalid document id")
        with self._lock, self._transaction():
            self._delete_document(doc_id)

    def _delete_document(self, doc_id: str, keep_card: bool = False) -> None:
        rowids = [row[0] for row in self._conn.execute("SELECT rowid FROM chunks WHERE doc_id = ?", (doc_id,))]
        self._delete_rowids(rowids)
        self._delete_file_entry(doc_id)
        if not keep_card:
            self._conn.execute("DELETE FROM doc_cards WHERE doc_id = ?", (doc_id,))

    def purge_document(self, doc_id: str) -> None:
        """``delete_document`` for good: the deleted rows are overwritten in the file, not just
        unlinked. Words can still sit in the full-text index's old segments and in free pages
        until ``scrub`` runs (measured in design/PRIVACY.md)."""
        if not _DOC_ID.fullmatch(doc_id):
            raise ValueError("Invalid document id")
        with self._lock:
            self._conn.execute("PRAGMA secure_delete=ON")
            try:
                self.delete_document(doc_id)
            finally:
                self._conn.execute("PRAGMA secure_delete=OFF")

    def scrub(self) -> None:
        """Leave no trace of purged documents in the file: merge the full-text indexes (their
        old segments still hold deleted words), rebuild the file (VACUUM writes only live
        pages) and empty the write-ahead log. About a second for 4,000 chunks."""
        with self._lock:
            for table in ("chunks_fts", "file_entries_fts"):
                self._conn.execute(f"INSERT INTO {table}({table}) VALUES('optimize')")
            self._conn.execute("VACUUM")
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def clear(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                DROP TABLE IF EXISTS doc_cards;
                DROP TABLE IF EXISTS file_entries_vec;
                DROP TABLE IF EXISTS file_entries_fts;
                DROP TABLE IF EXISTS file_entries;
                DROP TABLE IF EXISTS chunks_vec;
                DROP TABLE IF EXISTS chunks_fts;
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS store_meta;
                """
            )
            self._conn.execute("VACUUM")
        self.ensure_collection()

    def list_documents(self) -> list[dict[str, str]]:
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT DISTINCT doc_id, source FROM chunks WHERE doc_id != '' ORDER BY doc_id, source"
            ).fetchall()
        return [{"doc_id": doc_id, "source": source} for doc_id, source in rows]

    def has_document(self, doc_id: str) -> bool:
        if not _DOC_ID.fullmatch(doc_id):
            return False
        with self._reading() as conn:
            return conn.execute("SELECT 1 FROM chunks WHERE doc_id = ? LIMIT 1", (doc_id,)).fetchone() is not None

    def rename_source(self, doc_id: str, source: str) -> int:
        """Point a document's chunks at a renamed file without re-embedding them."""
        if not _DOC_ID.fullmatch(doc_id):
            raise ValueError("Invalid document id")
        with self._lock, self._transaction():
            # The file name is part of the entry text: version 0 marks it for rebuilding.
            self._conn.execute("UPDATE file_entries SET source = ?, version = 0 WHERE doc_id = ?", (source, doc_id))
            return self._conn.execute("UPDATE chunks SET source = ? WHERE doc_id = ?", (source, doc_id)).rowcount

    def reassign_document(self, old_doc_id: str, new_doc_id: str) -> int:
        """Move a document's chunks to a new id (legacy hash id -> registry file_id).

        Parent ids embed the doc id (``<doc_id>-p<n>``), so they are rewritten too.
        """
        if not (_DOC_ID.fullmatch(old_doc_id) and _DOC_ID.fullmatch(new_doc_id)):
            raise ValueError("Invalid document id")
        with self._lock, self._transaction():
            self._delete_file_entry(new_doc_id)
            self._conn.execute("UPDATE file_entries SET doc_id = ? WHERE doc_id = ?", (new_doc_id, old_doc_id))
            self._conn.execute("DELETE FROM doc_cards WHERE doc_id = ?", (new_doc_id,))
            self._conn.execute("UPDATE doc_cards SET doc_id = ? WHERE doc_id = ?", (new_doc_id, old_doc_id))
            return self._conn.execute(
                "UPDATE chunks SET doc_id = ?, parent_id = ? || substr(parent_id, ?) WHERE doc_id = ?",
                (new_doc_id, new_doc_id, len(old_doc_id) + 1, old_doc_id),
            ).rowcount

    def vectors_by_text(self, doc_id: str) -> dict[str, list[float]]:
        """``text_key(text) -> vector`` of a document's chunks: re-indexing an edited file
        embeds only the chunks whose text changed."""
        if not _DOC_ID.fullmatch(doc_id):
            return {}
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT c.text, v.embedding FROM chunks c JOIN chunks_vec v ON v.rowid = c.rowid WHERE c.doc_id = ?",
                (doc_id,),
            ).fetchall()
        return {text_key(text): np.frombuffer(blob, dtype=np.float32).tolist() for text, blob in rows}

    def document_ids_for_source(self, source: str) -> set[str]:
        with self._reading() as conn:
            return {row[0] for row in conn.execute("SELECT DISTINCT doc_id FROM chunks WHERE source = ?", (source,))}

    def chunk_counts(self) -> dict[str, int]:
        with self._reading() as conn:
            return dict(conn.execute("SELECT doc_id, COUNT(*) FROM chunks GROUP BY doc_id").fetchall())

    def count(self) -> int:
        with self._reading() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def search_dense(self, vector: list[float], limit: int | None = None) -> list[SearchHit]:
        with self._reading() as conn:
            return self._dense(conn, vector, limit)

    def search_sparse(self, query: str, limit: int | None = None) -> list[SearchHit]:
        match = build_match_query(query)
        if not match:
            return []
        with self._reading() as conn:
            return self._sparse(conn, match, limit)

    def _dense(self, conn: sqlite3.Connection, vector: list[float], limit: int | None) -> list[SearchHit]:
        query = np.asarray(vector, dtype=np.float32).tobytes()
        rows = conn.execute(
            f"""
            WITH knn AS (
                SELECT rowid, distance FROM chunks_vec
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT {_CHUNK_COLUMNS}, knn.distance
            FROM knn JOIN chunks c ON c.rowid = knn.rowid
            ORDER BY knn.distance
            """,
            (query, int(limit or self.settings.dense_top_k)),
        ).fetchall()
        # Report cosine similarity like Milvus' COSINE metric (higher is better).
        return [self._to_hit(row[:-1], 1.0 - float(row[-1]), "dense") for row in rows]

    def _sparse(self, conn: sqlite3.Connection, match: str, limit: int | None) -> list[SearchHit]:
        if not match:
            return []
        rows = conn.execute(
            f"""
            WITH ranked AS (
                SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
            )
            SELECT {_CHUNK_COLUMNS}, ranked.score
            FROM ranked JOIN chunks c ON c.rowid = ranked.rowid
            ORDER BY ranked.score
            """,
            (match, int(limit or self.settings.sparse_top_k)),
        ).fetchall()
        # FTS5 bm25() is negated so that smaller is better; flip it back.
        return [self._to_hit(row[:-1], -float(row[-1]), "bm25") for row in rows]

    def search_hybrid(
        self,
        query: str,
        vector: list[float],
        limit: int | None = None,
        doc_ids: Iterable[str] | None = None,
    ) -> list[SearchHit]:
        """RRF of dense and BM25; ``doc_ids`` restricts the search to those documents.

        Each route returns at least ``limit`` hits, so a larger pool reaches deeper into both.
        """
        limit = limit or self.settings.fusion_top_k
        if doc_ids is not None:
            return self._search_hybrid_in(query, vector, sorted(set(doc_ids)), limit)
        with self._reading(snapshot=True) as conn:
            dense_hits = self._dense(conn, vector, max(self.settings.dense_top_k, limit))
            # Segmented after the dense route, as before: right after start-up the first
            # question's KNN runs while the warm-up is still loading jieba's dictionary.
            sparse_hits = self._sparse(conn, build_match_query(query), max(self.settings.sparse_top_k, limit))
        by_id: dict[str, SearchHit] = {}
        scores: dict[str, float] = {}
        for ranking in (dense_hits, sparse_hits):
            for rank, hit in enumerate(ranking, start=1):
                by_id.setdefault(hit.id, hit)
                scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (self.settings.rrf_k + rank)
        fused = sorted(by_id.values(), key=lambda hit: scores[hit.id], reverse=True)
        for hit in fused:
            hit.score = scores[hit.id]
            hit.retrieval_method = "hybrid_rrf"
        return fused[:limit]

    def search_in_source(self, query: str, vector: list[float], source: str, limit: int = 5) -> list[SearchHit]:
        """Hybrid search restricted to one document (exact scan of its vectors).

        Used when a question names a document that a global search does not reach.
        """
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        with self._reading(snapshot=True) as conn:
            dense = conn.execute(
                f"""
                SELECT {_CHUNK_COLUMNS}, vec_distance_cosine(v.embedding, ?) AS distance
                FROM chunks c JOIN chunks_vec v ON v.rowid = c.rowid
                WHERE c.source = ?
                ORDER BY distance LIMIT ?
                """,
                (blob, source, self.settings.dense_top_k),
            ).fetchall()
            match = build_match_query(query)
            sparse = conn.execute(
                f"""
                SELECT {_CHUNK_COLUMNS}, bm25(chunks_fts) AS score
                FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ? AND c.source = ?
                ORDER BY score LIMIT ?
                """,
                (match, source, self.settings.sparse_top_k),
            ).fetchall() if match else []
        by_id: dict[str, SearchHit] = {}
        scores: dict[str, float] = {}
        for rows, method in ((dense, "dense"), (sparse, "bm25")):
            for rank, row in enumerate(rows, start=1):
                hit = by_id.setdefault(row[0], self._to_hit(row[:-1], 0.0, method))
                scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (self.settings.rrf_k + rank)
        fused = sorted(by_id.values(), key=lambda hit: scores[hit.id], reverse=True)
        for hit in fused:
            hit.score = scores[hit.id]
            hit.retrieval_method = "hybrid_rrf_source"
        return fused[:limit]

    def chunks_of_parents(self, parent_ids: Iterable[str], limit: int = 200) -> list[SearchHit]:
        """The chunks of the given parents, in index order (neighbouring passages of a hit)."""
        wanted = list(dict.fromkeys(parent_ids))
        # Parent ids are "<doc_id>-p<n>": the documents narrow the scan to their chunks (indexed).
        docs = json.dumps(sorted({p.rpartition("-p")[0] for p in wanted if "-p" in p}))
        with self._reading() as conn:
            rows = conn.execute(
                f"""
                SELECT {_CHUNK_COLUMNS} FROM chunks c
                WHERE c.doc_id IN (SELECT value FROM json_each(?))
                  AND c.parent_id IN (SELECT value FROM json_each(?))
                ORDER BY c.rowid LIMIT ?
                """,
                (docs, json.dumps(wanted), int(limit)),
            ).fetchall()
        return [self._to_hit(row, 0.0, "neighbor") for row in rows]

    # Exact scans stay fast up to this many chunks (sqlite-vec brute force, ~1-2 ms per 1k).
    exact_scan_limit = 20000

    def _search_hybrid_in(self, query: str, vector: list[float], doc_ids: list[str], limit: int) -> list[SearchHit]:
        if not doc_ids:
            return []
        ids = json.dumps(doc_ids)
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        match = build_match_query(query)
        dense_k, sparse_k = max(self.settings.dense_top_k, limit), max(self.settings.sparse_top_k, limit)
        in_docs = "c.doc_id IN (SELECT value FROM json_each(?))"
        with self._reading(snapshot=True) as conn:
            size = conn.execute(f"SELECT COUNT(*) FROM chunks c WHERE {in_docs}", (ids,)).fetchone()[0]
            if size <= self.exact_scan_limit:
                dense = conn.execute(
                    f"""
                    SELECT {_CHUNK_COLUMNS}, vec_distance_cosine(v.embedding, ?) AS distance
                    FROM chunks c JOIN chunks_vec v ON v.rowid = c.rowid
                    WHERE {in_docs}
                    ORDER BY distance LIMIT ?
                    """,
                    (blob, ids, dense_k),
                ).fetchall()
            else:
                # Most of the library: a wider global KNN, then keep the wanted documents.
                wanted = set(doc_ids)
                dense = [
                    row
                    for row in conn.execute(
                        f"""
                        WITH knn AS (SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ?)
                        SELECT {_CHUNK_COLUMNS}, knn.distance FROM knn JOIN chunks c ON c.rowid = knn.rowid
                        ORDER BY knn.distance
                        """,
                        (blob, 4 * dense_k),
                    )
                    if row[1] in wanted
                ][:dense_k]
            sparse = conn.execute(
                f"""
                SELECT {_CHUNK_COLUMNS}, bm25(chunks_fts) AS score
                FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ? AND {in_docs}
                ORDER BY score LIMIT ?
                """,
                (match, ids, sparse_k),
            ).fetchall() if match else []
        return self._fuse(((dense, "dense"), (sparse, "bm25")), limit, "hybrid_rrf_scoped")

    def _fuse(self, routes, limit: int, method: str) -> list[SearchHit]:
        by_id: dict[str, SearchHit] = {}
        scores: dict[str, float] = {}
        for rows, route in routes:
            for rank, row in enumerate(rows, start=1):
                hit = by_id.setdefault(row[0], self._to_hit(row[:-1], 0.0, route))
                scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (self.settings.rrf_k + rank)
        fused = sorted(by_id.values(), key=lambda hit: scores[hit.id], reverse=True)
        for hit in fused:
            hit.score = scores[hit.id]
            hit.retrieval_method = method
        return fused[:limit]

    # ---------------- file-level entries ----------------

    def upsert_file_entries(self, entries: Iterable[tuple[str, str, str, int, list[float]]]) -> int:
        """Write ``(doc_id, source, text, version, vector)`` rows, replacing older ones."""
        rows = list(entries)
        if not rows:
            return 0
        tokens = [" ".join(tokenize(text)) for _, _, text, _, _ in rows]
        dim = int(self.settings.embedding_dimension)
        with self._lock, self._transaction():
            for (doc_id, source, text, version, vector), token_text in zip(rows, tokens, strict=True):
                array = np.asarray(vector, dtype=np.float32)
                if array.shape != (dim,):
                    raise ValueError(f"Embedding dimension mismatch: configured {dim}, received {array.shape}")
                self._delete_file_entry(doc_id)
                rowid = self._conn.execute(
                    "INSERT INTO file_entries(doc_id, source, text, version) VALUES (?, ?, ?, ?)",
                    (doc_id, source, text, int(version)),
                ).lastrowid
                self._conn.execute("INSERT INTO file_entries_fts(rowid, tokens) VALUES (?, ?)", (rowid, token_text))
                self._conn.execute(
                    "INSERT INTO file_entries_vec(rowid, embedding) VALUES (?, ?)", (rowid, array.tobytes())
                )
        return len(rows)

    def file_entry_versions(self) -> dict[str, int]:
        with self._reading() as conn:
            return dict(conn.execute("SELECT doc_id, version FROM file_entries").fetchall())

    def search_file_entries(
        self, query: str, vector: list[float], limit: int = 12, doc_ids: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        """Hybrid (RRF) search over file entries; exact scan, there is one row per file."""
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        match = build_match_query(query)
        scope, params = "", []
        if doc_ids is not None:
            scope, params = "f.doc_id IN (SELECT value FROM json_each(?))", [json.dumps(sorted(set(doc_ids)))]
        with self._reading(snapshot=True) as conn:
            dense = conn.execute(
                f"""
                SELECT f.doc_id, f.source, f.text, vec_distance_cosine(v.embedding, ?) AS distance
                FROM file_entries f JOIN file_entries_vec v ON v.rowid = f.rowid
                {"WHERE " + scope if scope else ""}
                ORDER BY distance LIMIT ?
                """,
                (blob, *params, limit),
            ).fetchall()
            sparse = conn.execute(
                f"""
                SELECT f.doc_id, f.source, f.text, bm25(file_entries_fts) AS score
                FROM file_entries_fts JOIN file_entries f ON f.rowid = file_entries_fts.rowid
                WHERE file_entries_fts MATCH ? {"AND " + scope if scope else ""}
                ORDER BY score LIMIT ?
                """,
                (match, *params, limit),
            ).fetchall() if match else []
        entries: dict[str, dict[str, Any]] = {}
        for rows in (dense, sparse):
            for rank, (doc_id, source, text, _) in enumerate(rows, start=1):
                entry = entries.setdefault(doc_id, {"doc_id": doc_id, "source": source, "text": text, "score": 0.0})
                entry["score"] += 1.0 / (self.settings.rrf_k + rank)
        return sorted(entries.values(), key=lambda entry: entry["score"], reverse=True)[:limit]

    def file_entry_texts(self, doc_ids: Iterable[str]) -> dict[str, str]:
        """``doc_id -> entry text`` for the given files (those that have an entry)."""
        ids = sorted(set(doc_ids))
        if not ids:
            return {}
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT doc_id, text FROM file_entries WHERE doc_id IN (SELECT value FROM json_each(?))",
                (json.dumps(ids),),
            ).fetchall()
        return dict(rows)

    # ---------------- document cards ----------------

    _CARD_FIELDS = ("doc_type", "label", "subject", "period", "summary", "input_hash", "version", "tokens", "model",
                    "created_at")

    def upsert_doc_card(self, doc_id: str, card: dict[str, Any]) -> None:
        if not _DOC_ID.fullmatch(doc_id):
            raise ValueError("Invalid document id")
        values = [card.get(field, 0 if field in ("version", "tokens") else "") for field in self._CARD_FIELDS]
        with self._lock, self._transaction():
            self._conn.execute(
                f"INSERT OR REPLACE INTO doc_cards(doc_id, {', '.join(self._CARD_FIELDS)}) "
                f"VALUES (?, {', '.join('?' * len(self._CARD_FIELDS))})",
                (doc_id, *values),
            )

    def delete_doc_card(self, doc_id: str) -> None:
        with self._lock, self._transaction():
            self._conn.execute("DELETE FROM doc_cards WHERE doc_id = ?", (doc_id,))

    def doc_card_hashes(self) -> dict[str, str]:
        with self._reading() as conn:
            return dict(conn.execute("SELECT doc_id, input_hash FROM doc_cards").fetchall())

    def doc_cards(self, doc_ids: Iterable[str] | None = None, types: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
        """Cards by ``doc_id``, optionally only of the given files and / or types."""
        clauses, params = [], []
        if doc_ids is not None:
            clauses.append("doc_id IN (SELECT value FROM json_each(?))")
            params.append(json.dumps(sorted(set(doc_ids))))
        if types is not None:
            clauses.append("doc_type IN (SELECT value FROM json_each(?))")
            params.append(json.dumps(sorted(set(types))))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._reading() as conn:
            rows = conn.execute(
                f"SELECT doc_id, {', '.join(self._CARD_FIELDS)} FROM doc_cards{where}", params
            ).fetchall()
        return {row[0]: dict(zip(self._CARD_FIELDS, row[1:])) for row in rows}

    def renormalize_doc_cards(self, normalize: Callable[[str], str]) -> int:
        """Re-map stored labels after the type vocabulary changed (no model call)."""
        with self._reading() as conn:
            rows = conn.execute("SELECT doc_id, label, doc_type FROM doc_cards").fetchall()
        changed = [(normalize(label), doc_id) for doc_id, label, doc_type in rows if normalize(label) != doc_type]
        if changed:
            with self._lock, self._transaction():
                self._conn.executemany("UPDATE doc_cards SET doc_type = ? WHERE doc_id = ?", changed)
        return len(changed)

    def _delete_file_entry(self, doc_id: str) -> None:
        row = self._conn.execute("SELECT rowid FROM file_entries WHERE doc_id = ?", (doc_id,)).fetchone()
        if row:
            self._conn.execute("DELETE FROM file_entries_vec WHERE rowid = ?", (row[0],))
            self._conn.execute("DELETE FROM file_entries_fts WHERE rowid = ?", (row[0],))
            self._conn.execute("DELETE FROM file_entries WHERE rowid = ?", (row[0],))

    def _delete_rowids(self, rowids: list[int]) -> None:
        for rowid in rowids:
            self._conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
            self._conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
            self._conn.execute("DELETE FROM chunks WHERE rowid = ?", (rowid,))

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    @staticmethod
    def _to_hit(row: tuple[Any, ...], score: float, method: str) -> SearchHit:
        chunk_id, doc_id, parent_id, source, heading, locator, text = row
        return SearchHit(
            id=str(chunk_id),
            score=score,
            text=str(text),
            doc_id=str(doc_id),
            parent_id=str(parent_id),
            source=str(source),
            heading=str(heading),
            locator=str(locator),
            retrieval_method=method,
        )
