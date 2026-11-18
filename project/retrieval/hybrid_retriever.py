"""Dense, hybrid and cross-encoder-reranked retrieval pipeline."""

from __future__ import annotations

from dataclasses import replace
import logging
import re
import threading
import time
from typing import Callable, Iterable, Literal, Sequence

from config import Settings, get_settings
from core import timings
from services.model_clients import SiliconFlowEmbeddings, SiliconFlowReranker
from . import multipart, wording
from .naming import named_documents
from .question_parts import question_parts
from .survey import Survey, rule_name, select_by_files, select_survey, survey_question, survey_topics, place_names, topic_files, excerpt, answers_in
from .selection import (
    PartsPolicy,
    SelectionPolicy,
    candidate_pool_size,
    crowded,
    drop_superseded,
    no_evidence,
    select_hits,
    select_parts,
    spread_documents,
)
from .types import ChunkStore, SearchHit

logger = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "hybrid", "hybrid_rerank"]

# "两本书对 RAG 的描述有何不同" / "《A》和《B》分别怎么讲 X": answers come from several documents.
_COMPARISON = re.compile(r"对比|比较|异同|有何不同|有什么不同|有哪些不同|区别|分别|各自|两本|两份|两篇|两者|《[^》]+》[^《]*《[^》]+》")
_TITLE = re.compile(r"《[^》]{1,40}》")
# "…的描述侧重点有何不同？" -- the comparison wording itself, which says nothing about the topic.
_ASKING = re.compile(
    r"(?:的)?(?:描述|讨论|解释|阐释|阐述|介绍|论述|看法|观点|讲解|处理|做法)?(?:的)?(?:侧重点|重点|角度)?"
    r"(?:有何|有什么|有哪些|存在)?(?:不同|异同|区别|差异)(?:侧重点|之处)?[？?。]*$"
)
_FILLER = re.compile(r"^(?:请)?(?:对比|比较)?[\s和与及、,，]*|这?两(?:本书|份资料|份文档|篇文章)[中里]?")


def comparison_topic(query: str) -> str | None:
    """For a comparison across documents, the query without the document names; else None.

    Titles in the question match the front matter that mentions both books (a preface,
    a reading list) better than the chapters that actually discuss the topic.
    """
    if not _COMPARISON.search(query):
        return None
    topic = _FILLER.sub("", _TITLE.sub("", query))
    topic = _ASKING.sub("", topic).strip(" ，,、和与对在上？?。")
    return topic or query


def _compact(text: str) -> str:
    return re.sub(r"[^0-9a-z一-鿿]", "", text.lower())


def named_sources(query: str, sources: set[str]) -> list[str]:
    """Documents the query names in 《》, matched loosely against file names."""
    named = []
    for title in (_compact(t[1:-1]) for t in _TITLE.findall(query)):
        if len(title) < 2:
            continue
        for source in sources:
            name = _compact(source.rsplit(".", 1)[0])
            grams = {title[i : i + 2] for i in range(len(title) - 1)}
            overlap = sum(gram in name for gram in grams) / max(1, len(grams))
            if (title in name or name in title or overlap >= 0.6) and source not in named:
                named.append(source)
    return named


def neighbor_parent_ids(parent_id: str) -> list[str]:
    """The parents before and after ``parent_id`` ("<doc>-p7" -> "<doc>-p6", "<doc>-p8")."""
    head, sep, number = parent_id.rpartition("-p")
    if not sep or not number.isdigit():
        return []
    n = int(number)
    return [f"{head}-p{m}" for m in (n - 1, n + 1) if m >= 0]


def diversify(ranked: list[SearchHit], k: int, named: list[str] | None = None, min_ratio: float = 0.05) -> list[SearchHit]:
    """Top ``k`` by score, but containing the best hit of each named source (or the two best sources)."""
    if not ranked:
        return []
    def score(hit: SearchHit) -> float:
        return hit.rerank_score if hit.rerank_score is not None else hit.score

    best: dict[str, SearchHit] = {}
    for hit in ranked:
        best.setdefault(hit.source, hit)
    top = score(ranked[0])
    wanted = [best[source] for source in (named or []) if source in best]
    others = [hit for hit in sorted(best.values(), key=score, reverse=True) if hit not in wanted]
    leaders = wanted + [hit for hit in others[: max(0, 2 - len(wanted))] if score(hit) >= min_ratio * top]
    rest = [hit for hit in ranked if hit not in leaders]
    return sorted(leaders + rest[: max(0, k - len(leaders))], key=score, reverse=True)


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        embeddings: SiliconFlowEmbeddings,
        reranker: SiliconFlowReranker,
        settings: Settings | None = None,
        describe_document: Callable[[str], tuple[str, float] | None] | None = None,
    ):
        self.store = store
        self.embeddings = embeddings
        self.reranker = reranker
        self.settings = settings or get_settings()
        # doc_id -> (file name, modification time); set once a file registry exists, so
        # older versions of a file are dropped when its newest version is retrieved.
        self.describe_document = describe_document
        # () -> doc ids kept in the index but left out of answers (files in the Recycle Bin).
        self.hidden_documents: Callable[[], Iterable[str]] | None = None
        self._count: tuple[float, int | None] | None = None
        self._count_lock = threading.Lock()

    # The library's chunk count (for the candidate pool size) is re-read at most this often.
    count_ttl_seconds = 300.0

    def candidate_pool(self) -> int:
        """Fused candidates sent to the reranker: ``fusion_top_k``, more for large libraries."""
        settings = self.settings
        if not settings.candidate_auto_scale:
            return settings.fusion_top_k
        return candidate_pool_size(settings.fusion_top_k, self._chunk_count(), settings.candidate_pool_max)

    def _chunk_count(self) -> int | None:
        count = getattr(self.store, "count", None)
        if count is None:
            return None
        with self._count_lock:
            if self._count is None or time.monotonic() - self._count[0] > self.count_ttl_seconds:
                try:
                    self._count = (time.monotonic(), int(count()))
                except Exception as exc:
                    logger.warning("Chunk count unavailable: %s", exc)
                    self._count = (time.monotonic(), None)
            return self._count[1]

    def policy(
        self, max_keep: int | None = None, per_document: int | None = None, relative: float | None = None
    ) -> SelectionPolicy:
        """Which reranked hits to keep; ``max_keep`` (the question type's) never exceeds rerank_top_k.

        ``per_document`` (the question type's) replaces the setting rerank_per_document.
        """
        settings = self.settings
        cap = settings.rerank_top_k if max_keep is None else max(1, min(max_keep, settings.rerank_top_k))
        return SelectionPolicy(
            min_score=settings.rerank_min_score,
            relative=settings.rerank_relative_score if relative is None else relative,
            min_keep=min(settings.rerank_min_keep, cap),
            max_keep=cap,
            per_document=settings.rerank_per_document if per_document is None else per_document,
            no_evidence_below=getattr(settings, "rerank_no_evidence_below", 0.0),
        )

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "hybrid_rerank",
        top_k: int | None = None,
        query_vector: list[float] | None = None,
        doc_ids: Iterable[str] | None = None,
        max_keep: int | None = None,
        per_document: int | None = None,
        relative: float | None = None,
        prefer_doc_ids: Iterable[str] | None = None,
        prefer_only: bool = False,
    ) -> list[SearchHit]:
        """``doc_ids`` limits the search to those documents ("ask about this file/folder").
        ``prefer_doc_ids`` (a follow-up's previous files) are searched as well and their
        passages get settings.history_prior_boost on their rerank score; with ``prefer_only``
        (a follow-up that names nothing of its own, "用了哪几步制备工艺？") they answer it alone
        when they hold evidence, like a file the question names.

        After reranking, the hits that clear the score cutoffs are kept (see ``policy``),
        at most ``max_keep`` and at most ``per_document`` from one document while others
        qualify (``relative``: the fraction of the best score a hit needs, instead of the
        setting); an explicit ``top_k`` keeps exactly that many instead. An empty result
        after reranking means no evidence: even the best passage scored below
        rerank_no_evidence_below.
        """
        query = query.strip()
        if not query:
            return []
        scope = set(doc_ids) if doc_ids is not None else None
        if scope is not None and not scope:
            return []
        # "一般…能给多少补贴", "各…分别是多少": several files answer it, one line each (survey.py).
        survey = self._survey(query, mode, top_k)
        topic = comparison_topic(query) if mode == "hybrid_rerank" and survey is None else None
        if topic is not None:
            return self._retrieve_comparison(query, topic, top_k, max_keep, query_vector, scope, per_document)
        settings = self.settings
        trace = multipart.start_trace()
        if survey is not None and getattr(settings, "survey_file_first", False) and not (prefer_doc_ids and prefer_only):
            # The files about the question's topic first, then each one's answering passage.
            by_files = self._survey_by_files(query, survey, scope, max_keep, per_document, relative, trace)
            if by_files is not None:
                return by_files
        parts = self._question_parts(query, mode, top_k) if survey is None else []
        trace.parts = parts
        part_mode = getattr(settings, "part_retrieval", "off") if parts else "off"
        trace.mode = part_mode
        # Everyday wording ("能给多少补贴") searched beside the documents' own words (wording.py).
        expansion = self._wording(query, mode, top_k)
        trace.wording = expansion.to_dict() if expansion else {}
        wording_vector: list[float] | None = None
        vector = query_vector
        part_vectors: list[list[float] | None] = [None] * len(parts)
        embedding_error: Exception | None = None
        if vector is None or part_mode == "always" or expansion is not None:
            # The parts and the expanded question go in the same request as the question: no
            # extra round trip.
            texts = ([query] if vector is None else []) + (parts if part_mode == "always" else [])
            texts += [expansion.text] if expansion is not None else []
            try:
                vectors = self._embed_many(texts)
                trace.embed_texts = len(texts)
                if expansion is not None:
                    wording_vector, vectors = vectors[-1], vectors[:-1]
                if vector is None:
                    vector, vectors = vectors[0], vectors[1:]
                if part_mode == "always":
                    part_vectors = list(vectors)
            except Exception as exc:
                embedding_error = exc
                logger.warning("Embedding failed; trying BM25-only retrieval: %s", exc)

        if mode == "dense":
            if vector is None:
                raise RuntimeError("dense retrieval unavailable because embedding failed") from embedding_error
            hits = self._visible(self.store.search_dense(vector, limit=top_k or self.settings.rerank_top_k))
            return hits if scope is None else [hit for hit in hits if hit.doc_id in scope]

        candidate_limit = max(top_k or 0, self.candidate_pool())
        survey_cap = 0
        if survey is not None:
            # A wider pool with few candidates per file: every likely file gets its passages scored.
            candidate_limit = max(candidate_limit, int(getattr(settings, "survey_pool", 60)))
            survey_cap = int(getattr(settings, "survey_candidates_per_document", 3))
        keep = top_k or self.policy(max_keep, relative=relative).max_keep
        candidates: list[SearchHit] = []
        hybrid_error: Exception | None = None
        part_lists: list[list[SearchHit]] = [[] for _ in parts]
        wording_hits: list[SearchHit] = []
        if vector is not None:
            spread = mode == "hybrid_rerank" and not top_k
            jobs = [lambda: self._pool(query, vector, candidate_limit, scope, spread, trace, survey_cap)]
            if part_mode == "always":
                jobs += [
                    (lambda part=part, pv=pv: self._search_hybrid(part, pv, settings.part_search_top, scope) if pv else [])
                    for part, pv in zip(parts, part_vectors)
                ]
            if wording_vector is not None:
                if survey is not None:
                    jobs.append(
                        lambda: self._pool(expansion.text, wording_vector, candidate_limit, scope, True, None, survey_cap)
                    )
                else:
                    jobs.append(
                        lambda: self._search_hybrid(
                            expansion.text, wording_vector, getattr(settings, "wording_search_top", 40), scope
                        )
                    )
            results = multipart.in_parallel(jobs)
            if isinstance(results[0], Exception):
                hybrid_error = results[0]
                logger.warning("Store hybrid search failed; trying single routes: %s", hybrid_error)
            else:
                candidates = results[0]
            others = results[1:]
            if wording_vector is not None:
                found = others.pop()
                wording_hits = found if isinstance(found, list) else []
            if others:
                part_lists = [r if isinstance(r, list) else [] for r in others]
                trace.part_pools = [[hit.id for hit in hits] for hits in part_lists]

        if vector is None or hybrid_error is not None:
            with timings.stage("search"):
                candidates = self._degraded_retrieve(query, vector, candidate_limit)
            if scope is not None:
                candidates = [hit for hit in candidates if hit.doc_id in scope]
        if not candidates and hybrid_error is not None:
            raise RuntimeError("hybrid, dense and BM25 retrieval all failed") from hybrid_error
        candidates = self._visible(candidates)

        if mode == "hybrid" or not candidates:
            return candidates[:keep]

        # A file the question names ("答辩PPT用了哪几步…"): its best passages join the pool even
        # when a similar project's file matches the topic better.
        named = self._named_documents(query) if scope is None and not top_k else []
        trace.named = named
        if named and vector is not None:
            try:
                inside = self._visible(self._search_hybrid(query, vector, self.named_search_top, set(named)))
                seen = {hit.id for hit in candidates}
                candidates = candidates + [hit for hit in inside if hit.id not in seen]
            except Exception as exc:
                logger.warning("Search inside the named files failed: %s", exc)
        preferred = set(prefer_doc_ids or []) if scope is None and not top_k else set()
        trace.preferred = sorted(preferred)
        if preferred and vector is not None:
            try:
                inside = self._visible(self._search_hybrid(query, vector, self.named_search_top, preferred))
                seen = {hit.id for hit in candidates}
                candidates = candidates + [hit for hit in inside if hit.id not in seen]
            except Exception as exc:
                logger.warning("Search inside the previous files failed: %s", exc)

        # Candidates added to the whole question's pool (scored with it, in the same request).
        if named:
            survey = None  # a file the question names answers it; the usual paths apply
        if parts or wording_hits:
            wording_top = candidate_limit // 2 if survey is not None else getattr(settings, "wording_top", 20)
            candidates = candidates + self._expansions(query, candidates, part_lists, scope, trace, wording_hits, wording_top)
        part_jobs: list[tuple[int, list[SearchHit]]] = []
        if part_mode == "always" and getattr(settings, "part_rerank", "part") == "part":
            part_jobs = [(i, self._visible(hits)[: settings.part_rerank_top]) for i, hits in enumerate(part_lists)]
            part_jobs = [(i, hits) for i, hits in part_jobs if hits]
        try:
            # Every candidate is scored (the price is per input token either way): the
            # cutoffs and the per-document cap need the scores below the top. Each part's
            # few best candidates are scored against the part at the same time.
            jobs = [lambda: self._rerank(query, candidates, trace, "whole")]
            jobs += [(lambda i=i, hits=hits: self._rerank(parts[i], hits, trace, "part")) for i, hits in part_jobs]
            # An expanded question is also scored in the documents' own words (wording_rescore);
            # a passage keeps the better of its two scores.
            second = expansion is not None and bool(getattr(settings, "wording_rescore", False))
            if second:
                jobs.append(lambda: self._rerank(expansion.text, candidates, trace, "wording"))
            results = multipart.in_parallel(jobs)
            if isinstance(results[0], Exception):
                raise results[0]
            rankings = results[0]
            if second:
                other = results.pop()
                # Outside a survey, a question with no evidence as asked stays without: the second
                # scoring only reorders what the first found relevant.
                floor = float(getattr(settings, "rerank_no_evidence_below", 0.0) or 0.0)
                unfounded = survey is None and bool(rankings) and rankings[0].score < floor
                if not isinstance(other, Exception) and not unfounded:
                    better = {item.index: item.score for item in other}
                    rankings = sorted(
                        (type(item)(index=item.index, score=max(item.score, better.get(item.index, 0.0))) for item in rankings),
                        key=lambda item: -item.score,
                    )
        except Exception as exc:
            logger.warning("Reranker failed; returning RRF order: %s", exc)
            return candidates[:keep]

        reranked: list[SearchHit] = []
        for item in rankings:
            hit = candidates[item.index]
            hit.rerank_score = item.score
            hit.retrieval_method = "hybrid_rrf_bge_rerank"
            reranked.append(hit)
        reranked = self._current_versions(reranked)
        if top_k:
            return reranked[:top_k]
        if preferred:
            boost = getattr(settings, "history_prior_boost", 0.05)
            reranked = sorted(
                (multipart.with_score(hit, min(1.0, multipart.score_of(hit) + boost)) if hit.doc_id in preferred else hit
                 for hit in reranked),
                key=multipart.score_of, reverse=True,
            )
        policy = self.policy(max_keep, per_document, relative)
        if survey is not None and not (preferred and prefer_only):
            kept = select_survey(
                reranked, policy, survey, files=int(getattr(settings, "survey_files", 8)),
                floor=float(getattr(settings, "survey_floor", 0.01)), types=self._document_types(reranked),
            )
            trace.survey = {"ask": survey.ask, "cue": survey.cue, "files": len({hit.doc_id for hit in kept})}
            return self._excerpts(kept, survey)
        if preferred and prefer_only and not named:
            own = [hit for hit in reranked if hit.doc_id in preferred]
            if own and multipart.score_of(own[0]) >= settings.rerank_min_score:
                trace.preferred_used = True
                return select_hits(own, policy)
        named_best: list[SearchHit] = []
        if named:
            wanted = set(named)
            own = [hit for hit in reranked if hit.doc_id in wanted]
            if own and multipart.score_of(own[0]) >= settings.rerank_min_score:
                trace.named_used = True
                if not (parts or question_parts(query)):
                    # The named files answer it when they hold evidence; other files only otherwise.
                    return select_hits(own, policy)
                # A question of several parts may need other files too ("Between the article
                # X … and the report on Y …"): the named file's best passage is kept, not only it.
                named_best = own[:1]
        if not parts:
            if named_best:
                return select_parts(reranked, policy, PartsPolicy(), [named_best])
            return select_hits(reranked, policy)
        part_hits: list[list[SearchHit]] = [[] for _ in parts]
        for (i, hits), result in zip(part_jobs, results[1:]):
            if not isinstance(result, Exception):
                part_hits[i] = self._part_best(hits, result, trace, i)
        parts_policy = PartsPolicy(
            doc_first=getattr(settings, "multipart_doc_first", 0),
            doc_floor=getattr(settings, "multipart_doc_floor", 0.0),
            part_quota=getattr(settings, "part_quota", 1),
        )
        kept = select_parts(reranked, policy, parts_policy, part_hits + [named_best])
        if part_mode == "adaptive" and kept:
            # The cheap check first: does every part have a kept passage that mentions its
            # own words? Only the uncovered parts get one more round (search + scoring).
            missing = multipart.uncovered_parts(parts, kept)
            trace.uncovered = missing
            if missing:
                trace.triggered = True
                if getattr(settings, "part_rerank", "part") == "whole":
                    # The uncovered parts' new candidates are scored against the question itself
                    # and join its ranking; the documents-first cut then picks them up.
                    reranked = self._whole_round(query, [parts[i] for i in missing], reranked, scope, trace)
                else:
                    for i, hits in zip(missing, self._part_round([parts[i] for i in missing], scope, trace)):
                        part_hits[i] = hits
                kept = select_parts(reranked, policy, parts_policy, part_hits + [named_best])
        trace.reserved = [hit.id for hits in part_hits for hit in hits[: parts_policy.part_quota]]
        return kept

    # Passages searched inside the named files (added to the pool before reranking).
    named_search_top = 12
    # The library's (doc id, file name) list is re-read at most this often.
    names_ttl_seconds = 60.0

    def _documents(self) -> list[tuple[str, str]]:
        """The library's (doc id, file name) list, re-read at most every names_ttl_seconds."""
        listing = getattr(self.store, "list_documents", None)
        if listing is None:
            return []
        now = time.monotonic()
        cached = getattr(self, "_names", None)
        if cached is None or now - cached[0] > self.names_ttl_seconds:
            try:
                documents = [(row["doc_id"], row["source"]) for row in listing()]
            except Exception as exc:
                logger.warning("Document list unavailable: %s", exc)
                return []
            self._names = cached = (now, documents)
        return cached[1]

    def _named_documents(self, query: str) -> list[str]:
        """Doc ids of the files the question names (prefer_named_files; naming.named_documents)."""
        if not getattr(self.settings, "prefer_named_files", False):
            return []
        documents = self._documents()
        if not documents:
            return []
        named = named_documents(query, documents)
        # A name shared by many files ("报告") says little about which one is meant.
        return named if len(named) <= 6 else []

    def _survey(self, query: str, mode: str, top_k: int | None) -> Survey | None:
        """What kind of survey question this is (survey_route), or None."""
        if mode != "hybrid_rerank" or top_k or not getattr(self.settings, "survey_route", False):
            return None
        wide = bool(getattr(self.settings, "survey_wide", False))
        return survey_question(query, wide, self.survey_places() if wide else frozenset())

    def survey_places(self) -> frozenset[str]:
        """The places the library's file names start with (survey.place_names), for survey_wide."""
        documents = self._documents()
        key = tuple(source for _, source in documents)
        cached = getattr(self, "_places", None)
        if cached is None or cached[0] != key:
            self._places = cached = (key, place_names(key))
        return cached[1]

    def _survey_by_files(
        self, query: str, survey: Survey, scope: set[str] | None, max_keep: int | None, per_document: int | None,
        relative: float | None, trace,
    ) -> list[SearchHit] | None:
        """A survey question answered files first (survey_file_first): the question's topic
        ("考技能证书一般能领多少补贴" -> "考 技能 证书") is searched alone and its files ranked;
        each of the first survey_file_candidates files gives its survey_passages_per_file
        passages closest to what is asked; the reranker scores them against the topic (is the
        file about it?) and against the question (which passage answers it?), and
        ``select_by_files`` keeps one answer per file. None: the usual survey path instead (a
        file the question names, a failed search or scoring)."""
        settings = self.settings
        if scope is None and self._named_documents(query):
            return None
        topics = survey_topics(query) or [query]
        expansion = self._wording(query, "hybrid_rerank", None)
        asked_text = expansion.text if expansion is not None else query
        try:
            vectors = self._embed_many(topics + [asked_text])
            trace.embed_texts = len(vectors)
            limit = max(self.candidate_pool(), int(getattr(settings, "survey_pool", 60))) * self.crowd_depth
            wanted = int(getattr(settings, "survey_file_candidates", 16))
            # Each listed topic ("参展", "办展", "跨境 电商") brings its own files, in turns.
            lists = [topic_files(self._visible(self._search_hybrid(topic, vector, limit, scope)), wanted)
                     for topic, vector in zip(topics, vectors)]
            doc_ids: list[str] = []
            for rank in range(wanted):
                for found in lists:
                    if rank < len(found) and found[rank] not in doc_ids and len(doc_ids) < wanted:
                        doc_ids.append(found[rank])
            extra = int(getattr(settings, "survey_question_files", 0))
            if extra > 0:
                # Files the question's own words find (a programme the topic word does not name).
                asked_found = self._visible(self._search_hybrid(asked_text, vectors[-1], limit, scope))
                doc_ids += [d for d in topic_files(asked_found, extra + len(doc_ids)) if d not in doc_ids][:extra]
            per_file = int(getattr(settings, "survey_passages_per_file", 3))
            pool: list[SearchHit] = []
            seen: set[str] = set()
            for doc_id in doc_ids:
                for hit in self._search_hybrid(asked_text, vectors[-1], per_file, {doc_id}):
                    if hit.id not in seen:
                        seen.add(hit.id)
                        pool.append(hit)
        except Exception as exc:
            logger.warning("Survey search by files failed; using the usual survey path: %s", exc)
            return None
        pool = self._current_versions(self._visible(pool))
        if not pool:
            return []
        # Which passage of a file answers is judged on its answering sentences when
        # survey_excerpt_scoring is on (the topic on the whole passage).
        asked_pool = pool
        if getattr(settings, "survey_excerpt_scoring", False):
            asked_pool = [replace(hit, text=excerpt(hit.text, survey.ask) or hit.text) for hit in pool]
            # What was scored, for the evaluation's mapping of scored texts back to passages.
            trace.extra_hits.extend(short for short, hit in zip(asked_pool, pool) if short.text != hit.text)
        jobs = [lambda: self._rerank(query, asked_pool, trace, "whole")]
        jobs += [(lambda topic=topic: self._rerank(topic, pool, trace, "topic")) for topic in topics]
        second = asked_text != query
        if second:
            jobs.append(lambda: self._rerank(asked_text, asked_pool, trace, "wording"))
        results = multipart.in_parallel(jobs)
        failed = [result for result in results[: 1 + len(topics)] if isinstance(result, Exception)]
        if failed:
            logger.warning("Survey scoring failed; using the usual survey path: %s", failed[0])
            return None
        asked = {pool[item.index].id: item.score for item in results[0]}
        if second and not isinstance(results[-1], Exception):
            for item in results[-1]:
                key = pool[item.index].id
                asked[key] = max(asked.get(key, 0.0), item.score)
        # A passage is about the topic when it is about any one of the listed topics.
        topic_scores: dict[str, float] = {}
        for result in results[1 : 1 + len(topics)]:
            for item in result:
                key = pool[item.index].id
                topic_scores[key] = max(topic_scores.get(key, 0.0), item.score)
        kept = select_by_files(
            pool, topic_scores, asked, survey, self.policy(max_keep, per_document, relative),
            files=int(getattr(settings, "survey_files", 8)),
            topic_floor=float(getattr(settings, "survey_topic_floor", 0.05)),
            file_floor=float(getattr(settings, "survey_file_floor", 0.01)),
            topic_relative=float(getattr(settings, "survey_topic_relative", 0.05)),
            types=self._document_types(pool),
            answers=answers_in if getattr(settings, "survey_answering_files", False) else None,
            about_asked=bool(getattr(settings, "survey_about_asked", False)),
        )
        for hit in kept:
            hit.retrieval_method = "survey_files"
        kept = self._excerpts(kept, survey)
        trace.survey = {"ask": survey.ask, "cue": survey.cue, "route": "files", "topics": topics,
                        "candidate_files": len(doc_ids), "pool": len(pool), "files": len({hit.doc_id for hit in kept})}
        return kept

    def _excerpts(self, kept: list[SearchHit], survey: Survey) -> list[SearchHit]:
        """Each kept passage cut to the sentences that answer the ask (survey_excerpts,
        survey.excerpt): the answer copies one line per file instead of hunting in a long passage."""
        if not getattr(self.settings, "survey_excerpts", False):
            return kept
        out = []
        for hit in kept:
            short = excerpt(hit.text, survey.ask)
            out.append(replace(hit, text=short) if short else hit)
        return out

    def _document_types(self, hits: list[SearchHit]) -> dict[str, str]:
        """doc id -> document type: what the file's own name says (a type word, or the way
        rule-stating papers are named), else its document card."""
        from .doc_cards import name_doc_type

        types = {hit.doc_id: name_doc_type(hit.source) or ("制度规范" if rule_name(hit.source) else "") for hit in hits}
        missing = [doc_id for doc_id, kind in types.items() if not kind]
        cards = getattr(self.store, "doc_cards", None)
        if missing and cards is not None:
            try:
                for doc_id, card in cards(missing).items():
                    types[doc_id] = str(card.get("doc_type") or "")
            except Exception as exc:
                logger.warning("Document cards unavailable: %s", exc)
        return types

    def _wording(self, query: str, mode: str, top_k: int | None) -> wording.Expansion | None:
        """The question with the documents' words for its subject (wording_expansion), or None."""
        if mode != "hybrid_rerank" or top_k or not getattr(self.settings, "wording_expansion", False):
            return None
        return wording.expand(query)

    # ---------------------------------------------------------------- several parts
    def _question_parts(self, query: str, mode: str, top_k: int | None) -> list[str]:
        """The parts of a multi-part question when any multi-part step is on (else [])."""
        settings = self.settings
        if mode != "hybrid_rerank" or top_k:
            return []
        if (
            getattr(settings, "part_retrieval", "off") == "off"
            and not getattr(settings, "multipart_doc_first", 0)
            and not getattr(settings, "bridge_terms", False)
            and not getattr(settings, "neighbor_parents", 0)
        ):
            return []
        return question_parts(query)

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > 1 and hasattr(self.embeddings, "embed_queries"):
            return self.embeddings.embed_queries(texts)
        return [self.embeddings.embed_query(text) for text in texts]

    def _rerank(self, query: str, hits: list[SearchHit], trace, kind: str):
        trace.rerank_calls.append(
            {"query": kind, "docs": len(hits), "tokens": multipart.estimate_tokens(query, [h.text for h in hits])}
        )
        return self.reranker.rerank(query, [hit.text for hit in hits], top_n=len(hits))

    def _part_best(self, hits: list[SearchHit], rankings, trace, index: int) -> list[SearchHit]:
        """A part's scored candidates that clear part_min_score, best first."""
        floor = getattr(self.settings, "part_min_score", 0.1)
        scored = sorted(((hits[item.index], item.score) for item in rankings), key=lambda pair: -pair[1])
        while len(trace.part_scored) <= index:
            trace.part_scored.append([])
        trace.part_scored[index] = [(hit.id, round(score, 6)) for hit, score in scored]
        return [multipart.with_score(hit, score, "hybrid_rrf_bge_rerank_part") for hit, score in scored if score >= floor]

    def _part_search(self, parts: list[str], scope: set[str] | None, trace) -> list[list[SearchHit]]:
        """Each of ``parts`` searched on its own: one embedding request, searches in parallel."""
        settings = self.settings
        try:
            vectors = self._embed_many(parts)
            trace.embed_texts += len(parts)
        except Exception as exc:
            logger.warning("Part embedding failed: %s", exc)
            return [[] for _ in parts]
        searches = multipart.in_parallel(
            [(lambda part=part, v=v: self._search_hybrid(part, v, settings.part_search_top, scope)) for part, v in zip(parts, vectors)]
        )
        return [self._visible(r) if isinstance(r, list) else [] for r in searches]

    def _whole_round(
        self, query: str, parts: list[str], reranked: list[SearchHit], scope: set[str] | None, trace
    ) -> list[SearchHit]:
        """The adaptive round with part_rerank "whole": each part's first part_rerank_top new
        candidates, scored against the question in one request, merged into its ranking."""
        top = self.settings.part_rerank_top
        seen = {hit.id for hit in reranked}
        extra: list[SearchHit] = []
        for hits in self._part_search(parts, scope, trace):
            trace.part_pools.append([hit.id for hit in hits])
            new = [hit for hit in hits if hit.id not in seen][:top]
            seen.update(hit.id for hit in new)
            extra += new
        if not extra:
            return reranked
        try:
            rankings = self._rerank(query, extra, trace, "whole")
        except Exception as exc:
            logger.warning("Reranking the parts' candidates failed: %s", exc)
            return reranked
        scored = [multipart.with_score(extra[item.index], item.score, "hybrid_rrf_bge_rerank_part") for item in rankings]
        for hit in scored:
            trace.extra_kinds[hit.id] = "part"
        trace.extra += [hit.id for hit in scored]
        return sorted(reranked + self._current_versions(scored), key=multipart.score_of, reverse=True)

    def _part_round(self, parts: list[str], scope: set[str] | None, trace) -> list[list[SearchHit]]:
        """Search and score each of ``parts`` on its own (the adaptive second round)."""
        settings = self.settings
        lists = [hits[: settings.part_rerank_top] for hits in self._part_search(parts, scope, trace)]
        trace.part_pools += [[hit.id for hit in hits] for hits in lists]
        jobs = [(lambda part=part, hits=hits: self._rerank(part, hits, trace, "part") if hits else []) for part, hits in zip(parts, lists)]
        out = []
        for hits, result in zip(lists, multipart.in_parallel(jobs)):
            index = len(trace.part_scored)
            out.append([] if isinstance(result, Exception) or not hits else self._part_best(hits, result, trace, index))
        return out

    def _expansions(
        self, query: str, candidates: list[SearchHit], part_lists: list[list[SearchHit]], scope: set[str] | None, trace,
        wording_hits: Sequence[SearchHit] = (), wording_top: int = 20,
    ) -> list[SearchHit]:
        """Candidates scored with the whole question: the expanded question's new candidates
        (wording_expansion), each part's new candidates (part_rerank "whole"), bridge-term BM25
        hits, and the parents next to the best documents' hits."""
        settings = self.settings
        seen = {hit.id for hit in candidates}
        extra: list[SearchHit] = []

        def add(hits: list[SearchHit], kind: str, limit: int) -> None:
            added = 0
            for hit in self._visible(hits):
                if added >= limit:
                    break
                if hit.id in seen or (scope is not None and hit.doc_id not in scope):
                    continue
                seen.add(hit.id)
                extra.append(hit)
                trace.extra_kinds[hit.id] = kind
                added += 1

        if wording_hits:
            add(list(wording_hits), "wording", wording_top)
        if not trace.parts:
            trace.extra = [hit.id for hit in extra]
            trace.extra_hits = extra
            return extra
        if getattr(settings, "part_retrieval", "off") == "always" and getattr(settings, "part_rerank", "part") == "whole":
            for hits in part_lists:
                add(hits, "part", settings.part_rerank_top)
        if getattr(settings, "bridge_terms", False):
            terms = multipart.bridge_terms(query, [hit.text for hit in candidates[:5]])
            trace.bridge = terms
            if terms:
                try:
                    with timings.stage("search"):
                        hits = self.store.search_sparse(" ".join(terms), limit=settings.bridge_top * 3)
                    add(hits, "bridge", settings.bridge_top)
                except Exception as exc:
                    logger.warning("Bridge search failed: %s", exc)
        count = getattr(settings, "neighbor_parents", 0)
        lookup = getattr(self.store, "chunks_of_parents", None)
        if count and lookup is not None:
            wanted: list[str] = []
            docs: list[str] = []
            for hit in candidates:
                if hit.doc_id in docs:
                    continue
                docs.append(hit.doc_id)
                wanted += neighbor_parent_ids(hit.parent_id)
                if len(docs) >= count:
                    break
            try:
                with timings.stage("search"):
                    add(lookup(wanted), "neighbor", 4 * count)
            except Exception as exc:
                logger.warning("Neighbour lookup failed: %s", exc)
        trace.extra = [hit.id for hit in extra]
        trace.extra_hits = extra
        return extra

    def _visible(self, hits: list[SearchHit]) -> list[SearchHit]:
        if self.hidden_documents is None or not hits:
            return hits
        try:
            hidden = set(self.hidden_documents())
        except Exception as exc:
            logger.warning("Hidden documents unavailable: %s", exc)
            return hits
        return [hit for hit in hits if hit.doc_id not in hidden] if hidden else hits

    def _current_versions(self, ranked: list[SearchHit]) -> list[SearchHit]:
        if self.describe_document is None:
            return ranked
        try:
            return drop_superseded(ranked, self.describe_document)
        except Exception as exc:
            logger.warning("Version check failed; keeping every version: %s", exc)
            return ranked

    # How much deeper the search goes when a few files crowd the pool (candidate_per_document).
    crowd_depth = 4

    def _pool(
        self, query: str, vector: list[float], limit: int, scope: set[str] | None, spread: bool, trace,
        survey_cap: int = 0,
    ) -> list[SearchHit]:
        """The fused candidates for the reranker. With candidate_per_document, a pool that a
        few files fill ("three policy collections say 补贴 in every line") is searched again
        ``crowd_depth`` times deeper and cut to ``limit`` with at most that many candidates
        per file, so other files' passages get scored at all. A pool no file crowds is
        returned as it is."""
        hits = self._search_hybrid(query, vector, limit, scope)
        # A survey question has its own, smaller cap (survey_candidates_per_document).
        cap = survey_cap or int(getattr(self.settings, "candidate_per_document", 0) or 0)
        if not spread or cap <= 0 or (scope is not None and len(scope) < 2) or not crowded(hits, cap):
            return hits
        try:
            deeper = self._search_hybrid(query, vector, limit * self.crowd_depth, scope)
        except Exception as exc:
            logger.warning("Deeper search for a crowded pool failed: %s", exc)
            return hits
        spread_hits = spread_documents(self._visible(deeper), cap, limit)
        if {hit.id for hit in spread_hits} == {hit.id for hit in hits}:
            return hits  # nothing else to bring in (a library of a few files)
        if trace is not None:
            trace.crowded = {"files": len({hit.doc_id for hit in hits}), "files_after": len({hit.doc_id for hit in spread_hits})}
        return spread_hits

    def _search_hybrid(self, query: str, vector: list[float], limit: int, scope: set[str] | None) -> list[SearchHit]:
        with timings.stage("search"):
            if scope is None:
                return self.store.search_hybrid(query, vector, limit=limit)
            return self.store.search_hybrid(query, vector, limit=limit, doc_ids=scope)

    def _retrieve_comparison(
        self,
        query: str,
        topic: str,
        top_k: int | None,
        max_keep: int | None,
        query_vector: list[float] | None,
        scope: set[str] | None = None,
        per_document: int | None = None,
    ) -> list[SearchHit]:
        """Search the topic without document names, rerank every candidate, keep two sources.

        "广东省和河南省分别实施了哪些措施": with part_retrieval on, each part is searched and
        scored on its own at the same time, and its best passage is kept as well.
        """
        trace = multipart.start_trace()
        parts = self._question_parts(query, "hybrid_rerank", top_k)
        trace.parts = parts
        mode = getattr(self.settings, "part_retrieval", "off") if parts else "off"
        trace.mode = f"comparison_{mode}"
        if mode == "always":
            jobs = [
                lambda: self._compare_topic(query, topic, top_k, max_keep, query_vector, scope, per_document),
                lambda: self._part_evidence(parts, scope, trace),
            ]
            kept, part_hits = multipart.in_parallel(jobs)
            if isinstance(kept, Exception):
                raise kept
            if isinstance(part_hits, Exception):
                return kept
        else:
            kept = self._compare_topic(query, topic, top_k, max_keep, query_vector, scope, per_document)
            part_hits = [[] for _ in parts]
            if mode == "adaptive" and kept:
                # Only the parts the topic's passages leave uncovered get a round of their own.
                missing = multipart.uncovered_parts(parts, kept)
                trace.uncovered = missing
                if missing:
                    trace.triggered = True
                    for i, hits in zip(missing, self._part_round([parts[i] for i in missing], scope, trace)):
                        part_hits[i] = hits
        if not kept or not any(part_hits):
            return kept
        policy = self.policy(max_keep, per_document)
        quota = getattr(self.settings, "part_quota", 1)
        trace.reserved = [hit.id for hits in part_hits for hit in hits[:quota]]
        return multipart.add_part_hits(kept, part_hits, quota, max(len(kept), policy.max_keep) + 2)

    def _part_evidence(self, parts: list[str], scope: set[str] | None, trace) -> list[list[SearchHit]]:
        """Each part searched and its best candidates scored against it (one embedding request,
        searches and scoring in parallel): the part's passages above part_min_score."""
        return self._part_round(parts, scope, trace)

    def _compare_topic(
        self,
        query: str,
        topic: str,
        top_k: int | None,
        max_keep: int | None,
        query_vector: list[float] | None,
        scope: set[str] | None = None,
        per_document: int | None = None,
    ) -> list[SearchHit]:
        policy = self.policy(max_keep, per_document)
        k = top_k or policy.max_keep
        vector = query_vector if topic == query else None
        try:
            vector = vector or self.embeddings.embed_query(topic)
            # A wider pool, so every document the question names has a chance to be in it.
            candidates = self._search_hybrid(topic, vector, max(2 * self.candidate_pool(), k), scope)
        except Exception as exc:
            logger.warning("Comparison retrieval failed; using the plain pipeline: %s", exc)
            return self.retrieve(topic, mode="hybrid", top_k=k, doc_ids=scope)
        if not candidates:
            return []
        named = named_sources(query, {hit.source for hit in candidates})
        try:
            sources = {item["source"] for item in self.store.list_documents()} if _TITLE.search(query) else set()
        except Exception:
            sources = set()
        # Inside a scope only the scoped documents are searched; nothing is added from outside.
        missing = [] if scope is not None else [s for s in named_sources(query, sources) if s not in named]
        search_in_source = getattr(self.store, "search_in_source", None)
        if missing and search_in_source is not None:
            # A named document the topic search did not reach (another book dominates the
            # topic): search inside it and add its best passages to the pool before reranking.
            seen = {hit.id for hit in candidates}
            for source in missing:
                try:
                    with timings.stage("search"):
                        extra = [hit for hit in search_in_source(topic, vector, source, limit=4) if hit.id not in seen]
                except Exception as exc:
                    logger.warning("Searching inside %s failed: %s", source, exc)
                    continue
                candidates.extend(extra)
                if extra:
                    named.append(source)
        try:
            rankings = self._rerank(topic, candidates, multipart.last_trace() or multipart.start_trace(), "whole")
        except Exception as exc:
            logger.warning("Reranker failed; returning RRF order: %s", exc)
            return diversify(candidates, k, named)
        ranked = []
        for item in rankings:
            hit = candidates[item.index]
            hit.rerank_score = item.score
            hit.retrieval_method = "hybrid_rrf_bge_rerank_diverse"
            ranked.append(hit)
        ranked = self._current_versions(ranked)
        if not top_k:
            if no_evidence(ranked, policy):
                return []
            # As many as the cutoffs keep, with room for each named document's best passage.
            k = min(k, max(len(select_hits(ranked, policy)), len(named), 2))
        return diversify(ranked, k, named)

    def _degraded_retrieve(
        self,
        query: str,
        vector: list[float] | None,
        limit: int,
    ) -> list[SearchHit]:
        dense_hits: list[SearchHit] = []
        sparse_hits: list[SearchHit] = []
        if vector is not None:
            try:
                dense_hits = self.store.search_dense(vector, limit=limit)
            except Exception as exc:
                logger.warning("Dense fallback failed: %s", exc)
        try:
            sparse_hits = self.store.search_sparse(query, limit=limit)
        except Exception as exc:
            logger.warning("BM25 fallback failed: %s", exc)

        if dense_hits and sparse_hits:
            by_id: dict[str, SearchHit] = {}
            scores: dict[str, float] = {}
            for ranking in (dense_hits, sparse_hits):
                for rank, hit in enumerate(ranking, start=1):
                    by_id.setdefault(hit.id, hit)
                    scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (self.settings.rrf_k + rank)
            fused = sorted(by_id.values(), key=lambda hit: scores[hit.id], reverse=True)
            for hit in fused:
                hit.score = scores[hit.id]
                hit.retrieval_method = "fallback_rrf"
            return fused[:limit]
        if sparse_hits:
            for hit in sparse_hits:
                hit.retrieval_method = "bm25_degraded"
            return sparse_hits[:limit]
        if dense_hits:
            for hit in dense_hits:
                hit.retrieval_method = "dense_degraded"
            return dense_hits[:limit]
        return []
