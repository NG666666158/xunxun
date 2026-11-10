"""Copy every chunk and its dense vector from Milvus into the embedded SQLite store.

Vectors are exported with ``query_iterator`` and written as-is, so no embedding API
call is made. Parent chunks, parsed documents and uploads already live on disk
under ``data/`` and are shared by both backends, so only the chunk index moves.

    python scripts/migrate_milvus_to_embedded.py            # upsert into the embedded store
    python scripts/migrate_milvus_to_embedded.py --clear    # rebuild the embedded store first
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

from config import get_settings  # noqa: E402
from retrieval.types import ChunkRecord  # noqa: E402


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="把 Milvus 中的 chunk 与向量迁移到嵌入式 SQLite 库")
    parser.add_argument("--db-path", type=Path, default=None, help="目标 SQLite 文件，默认取 EMBEDDED_DB_PATH")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--clear", action="store_true", help="迁移前清空目标库")
    return parser.parse_args(argv)


def migrate(source, target, batch_size: int = 512, clear: bool = False) -> dict:
    """Stream ``source.iter_chunks()`` into ``target`` and verify per-document counts."""
    target.ensure_collection()
    if clear:
        target.clear()
    expected: Counter[str] = Counter()
    sources: dict[str, str] = {}
    batch: list[ChunkRecord] = []
    for record in source.iter_chunks(batch_size=batch_size):
        expected[record.doc_id] += 1
        sources[record.doc_id] = record.source
        batch.append(record)
        if len(batch) >= batch_size:
            target.insert(batch)
            batch = []
    target.insert(batch)

    migrated = target.chunk_counts()
    mismatched = {
        doc_id: {"milvus": count, "embedded": migrated.get(doc_id, 0)}
        for doc_id, count in expected.items()
        if migrated.get(doc_id, 0) != count
    }
    return {
        "documents": [
            {"doc_id": doc_id, "source": sources[doc_id], "chunks": count}
            for doc_id, count in sorted(expected.items())
        ],
        "chunks": sum(expected.values()),
        "mismatched": mismatched,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    from retrieval.embedded_store import EmbeddedStore
    from retrieval.milvus_store import MilvusStore

    source = MilvusStore(settings)
    if not source.client.has_collection(source.collection_name):
        raise SystemExit(f"Milvus 中没有集合 {source.collection_name}")
    source.client.load_collection(source.collection_name)
    target = EmbeddedStore(settings, db_path=args.db_path or settings.embedded_db_path)
    try:
        report = migrate(source, target, batch_size=args.batch_size, clear=args.clear)
    finally:
        target.close()
    report["target"] = str(target.db_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["mismatched"]:
        raise SystemExit("迁移后 chunk 数与 Milvus 不一致")


if __name__ == "__main__":
    main()
