"""Backend-neutral records shared by every chunk store, the retriever and ingestion.

Kept free of store imports so that the embedded backend (and a packaged desktop
build) never pulls in pymilvus/grpc just to construct a chunk or a search hit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable, Protocol


def text_key(text: str) -> str:
    """Identity of a chunk's text, for reusing its vector when a file is indexed again."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class ChunkRecord:
    id: str
    doc_id: str
    parent_id: str
    source: str
    heading: str
    locator: str
    text: str
    dense: list[float]


@dataclass
class SearchHit:
    id: str
    score: float
    text: str
    doc_id: str
    parent_id: str
    source: str
    heading: str
    locator: str
    retrieval_method: str
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChunkStore(Protocol):
    """Interface implemented by ``MilvusStore`` and ``EmbeddedStore``."""

    backend: str

    def ensure_collection(self) -> None: ...

    def insert(self, records: Iterable[ChunkRecord]) -> int: ...

    def delete_document(self, doc_id: str) -> None: ...

    def clear(self) -> None: ...

    def list_documents(self) -> list[dict[str, str]]: ...

    def has_document(self, doc_id: str) -> bool: ...

    def document_ids_for_source(self, source: str) -> set[str]: ...

    def search_dense(self, vector: list[float], limit: int | None = None) -> list[SearchHit]: ...

    def search_sparse(self, query: str, limit: int | None = None) -> list[SearchHit]: ...

    def search_hybrid(
        self, query: str, vector: list[float], limit: int | None = None, doc_ids: Iterable[str] | None = None
    ) -> list[SearchHit]: ...
