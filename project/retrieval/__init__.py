from .hybrid_retriever import HybridRetriever
from .types import ChunkRecord, ChunkStore, SearchHit

__all__ = ["ChunkRecord", "ChunkStore", "EmbeddedStore", "HybridRetriever", "MilvusStore", "SearchHit"]


def __getattr__(name: str):
    # Store classes load lazily: importing ``retrieval`` must not import pymilvus.
    if name == "MilvusStore":
        from .milvus_store import MilvusStore

        return MilvusStore
    if name == "EmbeddedStore":
        from .embedded_store import EmbeddedStore

        return EmbeddedStore
    raise AttributeError(name)
