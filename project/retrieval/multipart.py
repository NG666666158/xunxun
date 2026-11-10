"""Evidence for questions of several parts, and for answers that span documents.

The cross-encoder scores each passage against the whole question, so a passage that
answers one part of "A 有哪些作用，并且 B 取得了哪些成果？" scores low and the usual cutoffs
drop it, even when it is the only passage about B (evaluation/suite_v1/REPORT.md). What
this module adds to ``HybridRetriever.retrieve``, each behind its own setting
(measured in evaluation/multidoc_v1/REPORT.md):

* the parts themselves (``question_parts``, rules only);
* ``part_retrieval``: each part searched on its own (one batched embedding request, the
  searches in parallel) and its best few candidates scored against the part, so the best
  passage of every part is kept ("always"), or only for the parts the first cut left
  uncovered ("adaptive": a rule checks whether the kept passages mention each part's
  distinctive words; at most one extra round);
* ``multipart_doc_first``: the best passage of each of the first N documents is kept before
  second passages of the same documents (no extra call);
* ``bridge_terms``: names, numbers and titles taken from the passages already kept, searched
  again with BM25 (local), new candidates scored with the rest;
* ``neighbor_parents``: passages next to the best documents' kept passages (the parent
  before and after) join the candidates.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextvars
from dataclasses import dataclass, field, replace
import re
import threading
from typing import Any, Callable, Iterable, Sequence, TypeVar

from .question_parts import key_terms, mentions
from .types import SearchHit

T = TypeVar("T")

_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="parts")


def in_parallel(jobs: Sequence[Callable[[], T]], first_here: bool = True) -> list[T | BaseException]:
    """Run ``jobs`` concurrently; results (or the exception each raised) in order.

    The first job runs in the calling thread (the evaluation records that thread's calls);
    the others in a small shared pool, each with a copy of the caller's context so the
    turn's timings keep adding up.
    """
    if not jobs:
        return []
    if threading.current_thread().name.startswith("parts"):
        # Already inside the pool (a nested step): run here, so the pool never waits on itself.
        results: list[Any] = []
        for job in jobs:
            try:
                results.append(job())
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
        return results
    futures = []
    for job in jobs[1:] if first_here else jobs:
        context = contextvars.copy_context()
        futures.append(_POOL.submit(context.run, job))
    results: list[Any] = []
    if first_here:
        try:
            results.append(jobs[0]())
        except Exception as exc:  # noqa: BLE001 - returned to the caller, which decides
            results.append(exc)
    for future in futures:
        try:
            results.append(future.result())
        except Exception as exc:  # noqa: BLE001
            results.append(exc)
    return results


def score_of(hit: SearchHit) -> float:
    return hit.rerank_score if hit.rerank_score is not None else hit.score


def distinct_terms(parts: Sequence[str], limit: int = 10) -> list[list[str]]:
    """Each part's key terms that the other parts do not share (all its terms if none)."""
    terms = [key_terms(part, limit=limit) for part in parts]
    out = []
    for i, own in enumerate(terms):
        others = {t for j, ts in enumerate(terms) if j != i for t in ts}
        mine = [t for t in own if t not in others]
        out.append(mine or own)
    return out


def uncovered_parts(parts: Sequence[str], kept: Sequence[SearchHit], share: float = 0.5) -> list[int]:
    """Indexes of the parts that no kept passage mentions (at least ``share`` of the part's
    distinctive words): the cheap coverage check before a second round."""
    if not parts:
        return []
    terms = distinct_terms(parts)
    return [i for i, words in enumerate(terms) if words and not any(mentions(hit.text, words) >= share for hit in kept)]


# ---------------------------------------------------------------- bridge terms
_QUOTED = re.compile(r"[《“「『\"]([^》”」』\"]{2,30})[》”」』\"]")
_NUMBER_UNIT = re.compile(r"\d+(?:\.\d+)?(?:%|万|亿|千|百|元|吨|人|家|个|项|件|名|岁|年|月|日|号)")
_CAPITALISED = re.compile(r"\b[A-Z][a-zA-Z0-9&.-]+(?:\s+[A-Z][a-zA-Z0-9&.-]+)*")


def bridge_terms(query: str, passages: Iterable[str], limit: int = 6) -> list[str]:
    """Names, titles and numbers in ``passages`` that the query does not contain.

    "Which company … reported by The Verge and TechCrunch" -> the company the first passages
    name; "国家发展改革委的哪项政策…" -> 《关于促进家居消费若干措施的通知》. Terms found in more
    than one passage come first.
    """
    from .embedded_store import tokenize

    low_query = query.lower()
    counts: dict[str, int] = {}
    for text in passages:
        found: set[str] = set()
        found.update(m.strip() for m in _QUOTED.findall(text))
        found.update(_NUMBER_UNIT.findall(text))
        found.update(m.strip() for m in _CAPITALISED.findall(text) if len(m) >= 3)
        # Long Chinese words (names of places, organisations, products) from the same
        # segmentation BM25 uses.
        found.update(w for w in tokenize(text[:1500]) if len(w) >= 4 and re.search(r"[㐀-鿿]", w))
        for term in found:
            if term and term.lower() not in low_query and not term.startswith("【"):
                counts[term] = counts.get(term, 0) + 1
    ranked = sorted(counts, key=lambda t: (-counts[t], -len(t)))
    return ranked[:limit]


# ---------------------------------------------------------------- trace
@dataclass
class PartsTrace:
    """What the multi-part steps did for one question (the evaluation reads it)."""

    parts: list[str] = field(default_factory=list)
    mode: str = ""
    triggered: bool = False  # the adaptive round ran
    uncovered: list[int] = field(default_factory=list)
    part_pools: list[list[str]] = field(default_factory=list)  # chunk ids per part (search order)
    part_scored: list[list[tuple[str, float]]] = field(default_factory=list)  # (chunk id, part score), best first
    extra: list[str] = field(default_factory=list)  # chunk ids added to the whole-question pool
    extra_kinds: dict[str, str] = field(default_factory=dict)
    bridge: list[str] = field(default_factory=list)
    rerank_calls: list[dict[str, Any]] = field(default_factory=list)  # {"query": "whole"|"part", "docs": n, "tokens": est}
    embed_texts: int = 0
    reserved: list[str] = field(default_factory=list)
    named: list[str] = field(default_factory=list)  # doc ids of the files the question names
    named_used: bool = False  # ... and the answer came from them
    preferred: list[str] = field(default_factory=list)  # a follow-up's previous files
    preferred_used: bool = False  # ... answered it alone (an elliptical follow-up)
    extra_hits: list[SearchHit] = field(default_factory=list, repr=False)  # not serialized

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__ if key != "extra_hits"}


_TRACE = threading.local()


def start_trace() -> PartsTrace:
    _TRACE.value = PartsTrace()
    return _TRACE.value


def last_trace() -> PartsTrace | None:
    return getattr(_TRACE, "value", None)


def estimate_tokens(query: str, documents: Sequence[str]) -> int:
    """Rerank input tokens, roughly: one per CJK character, one per 3.5 other characters."""

    def tokens(text: str) -> float:
        cjk = sum(1 for ch in text if "㐀" <= ch <= "鿿")
        return cjk + (len(text) - cjk) / 3.5

    return int(sum(tokens(d) + tokens(query) + 4 for d in documents))


def with_score(hit: SearchHit, score: float, method: str | None = None) -> SearchHit:
    return replace(hit, rerank_score=score, retrieval_method=method or hit.retrieval_method)


def add_part_hits(kept: list[SearchHit], part_hits: Sequence[Sequence[SearchHit]], quota: int, cap: int) -> list[SearchHit]:
    """``kept`` plus each part's best ``quota`` passages not already in it (at most ``cap``
    in all, the part passages before the weakest of ``kept``), best first."""
    ids = {hit.id for hit in kept}
    extra: list[SearchHit] = []
    for hits in part_hits:
        for hit in list(hits)[:quota]:
            if hit.id not in ids:
                ids.add(hit.id)
                extra.append(hit)
    if not extra:
        return kept
    room = max(0, cap - len(extra))
    merged = sorted(kept, key=score_of, reverse=True)[:room] + extra
    return sorted(merged, key=score_of, reverse=True)
