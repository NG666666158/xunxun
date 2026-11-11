"""Did two imports of the same folder build the same index?

    python -I evaluation/import_speed_v1/compare_runs.py <data dir before> <data dir after>

The data directories are the ones profile_import.py imported into. Files are matched by path;
per file the chunks must be the same, in the same order (parent, heading, locator, text; chunk
ids hash the run's own file id so they always differ), and so must the parent passages. Vectors
are compared by cosine: the service returns slightly different floats for the same text from
one request to the next (about 1e-3), so equal means cosine >= 0.999.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

import numpy as np


def load(data_dir: Path) -> tuple[dict[str, list[tuple]], dict[str, np.ndarray], dict[str, str]]:
    conn = sqlite3.connect(f"{(data_dir / 'knowledge.sqlite3').as_uri()}?mode=ro", uri=True)
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    paths = dict(conn.execute("SELECT file_id, path FROM files"))
    chunks: dict[str, list[tuple]] = {}
    vectors: dict[str, list[np.ndarray]] = {}
    rows = conn.execute(
        "SELECT c.doc_id, c.parent_id, c.heading, c.locator, c.text, v.embedding "
        "FROM chunks c JOIN chunks_vec v ON v.rowid = c.rowid ORDER BY c.doc_id, c.rowid"
    )
    for doc_id, parent_id, heading, locator, text, blob in rows:
        key = paths.get(doc_id, doc_id)
        chunks.setdefault(key, []).append((parent_id.rpartition("-")[2], heading, locator, text))
        vectors.setdefault(key, []).append(np.frombuffer(blob, dtype=np.float32))
    parents = {}
    for doc_id, key in paths.items():
        path = data_dir / "parents" / f"{doc_id}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            parents[key] = json.dumps(
                [(pid.rpartition("-")[2], row["heading"], row["locator"], row["text"]) for pid, row in payload.items()],
                ensure_ascii=False,
            )
    return chunks, {k: np.vstack(v) for k, v in vectors.items()}, parents


def main() -> None:
    before_chunks, before_vectors, before_parents = load(Path(sys.argv[1]))
    after_chunks, after_vectors, after_parents = load(Path(sys.argv[2]))
    shared = sorted(set(before_chunks) & set(after_chunks))
    cosines = []
    for key in shared:
        a, b = before_vectors[key], after_vectors[key]
        if a.shape == b.shape:
            cosines.extend((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
    cosines = np.array(cosines)
    report = {
        "files": [len(before_chunks), len(after_chunks)],
        "only_before": len(set(before_chunks) - set(after_chunks)),
        "only_after": len(set(after_chunks) - set(before_chunks)),
        "chunks": [sum(map(len, before_chunks.values())), sum(map(len, after_chunks.values()))],
        "files_with_different_chunks": sum(1 for k in shared if before_chunks[k] != after_chunks[k]),
        "files_with_different_parents": sum(1 for k in shared if before_parents.get(k) != after_parents.get(k)),
        "vectors_compared": int(cosines.size),
        "cosine_min": round(float(cosines.min()), 6) if cosines.size else None,
        "cosine_mean": round(float(cosines.mean()), 6) if cosines.size else None,
        "vectors_below_0.999": int((cosines < 0.999).sum()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
