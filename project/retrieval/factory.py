"""Construct the chunk store selected by ``Settings.vector_backend``.

Each backend is imported only when chosen, so the embedded default never imports
pymilvus and the Milvus backend never needs sqlite-vec/jieba.
"""

from __future__ import annotations

from config import Settings, get_settings
from .types import ChunkStore


def create_store(settings: Settings | None = None) -> ChunkStore:
    settings = settings or get_settings()
    if settings.vector_backend == "milvus":
        from .milvus_store import MilvusStore

        return MilvusStore(settings)
    from .embedded_store import EmbeddedStore

    return EmbeddedStore(settings, db_path=settings.embedded_db_path)
