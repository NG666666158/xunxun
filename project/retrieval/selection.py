"""What happens between the reranker and the context: how many hits to keep, and which.

* ``candidate_pool_size``: how many fused candidates go to the reranker, raised gently
  for large libraries (up to a cap).
* ``select_hits``: keep the hits whose rerank score clears an absolute floor and a
  fraction of the best score, between a minimum and a maximum count, with at most
  ``per_document`` hits from one document while other documents qualify. When even the
  best score is below ``no_evidence_below``, nothing is kept: there is no evidence.
* ``drop_superseded``: when several versions of one file are among the hits, keep only
  the newest (same folder-independent name once version marks are removed, same type).

Thresholds come from evaluation/retrieval_tuning_v1/REPORT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Callable, Iterable

from .types import SearchHit


def candidate_pool_size(base: int, chunk_count: int | None, cap: int, reference: int = 20000) -> int:
    """``base`` up to ``reference`` chunks; +25% per doubling beyond it, never above ``cap``.

    A larger library holds more near-duplicates of any topic, so a fixed pool covers
    less of it. The growth is logarithmic: 40k chunks -> 1.25x, 80k -> 1.5x.
    """
    if not chunk_count or chunk_count <= reference:
        return base
    return min(max(base, cap), math.ceil(base * (1 + 0.25 * math.log2(chunk_count / reference))))


@dataclass(frozen=True)
class SelectionPolicy:
    min_score: float  # absolute floor on the rerank score
    relative: float  # ... and at least this fraction of the best score
    min_keep: int  # kept regardless of score (the relevance gate decides about weak results)
    max_keep: int
    per_document: int = 0  # 0: no cap
    # The best score below this: no evidence, nothing kept (min_keep does not apply).
    no_evidence_below: float = 0.0


def no_evidence(ranked: list[SearchHit], policy: SelectionPolicy) -> bool:
    """True when even the best of ``ranked`` (best first) scores below the policy's floor."""
    return bool(ranked) and policy.no_evidence_below > 0 and _score(ranked[0]) < policy.no_evidence_below


def _score(hit: SearchHit) -> float:
    return hit.rerank_score if hit.rerank_score is not None else hit.score


def select_hits(ranked: list[SearchHit], policy: SelectionPolicy) -> list[SearchHit]:
    """``ranked`` is best first. Returns the kept hits, best first (none: no evidence)."""
    if not ranked or no_evidence(ranked, policy):
        return []
    top = _score(ranked[0])
    cutoff = max(policy.min_score, policy.relative * top)
    passing = [hit for hit in ranked if _score(hit) >= cutoff]
    if len(passing) < policy.min_keep:
        passing = ranked[: policy.min_keep]
    if policy.per_document <= 0:
        return passing[: policy.max_keep]
    kept: list[SearchHit] = []
    deferred: list[SearchHit] = []
    per_doc: dict[str, int] = {}
    for hit in passing:
        if per_doc.get(hit.doc_id, 0) < policy.per_document:
            kept.append(hit)
            per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
        else:
            deferred.append(hit)
        if len(kept) >= policy.max_keep:
            break
    # The cap makes room for other documents; it never leaves slots empty.
    kept += deferred[: max(0, policy.max_keep - len(kept))]
    return sorted(kept, key=_score, reverse=True)


@dataclass(frozen=True)
class PartsPolicy:
    """How a question of several parts is cut (evaluation/multidoc_v1/REPORT.md, method 1).

    A document that answers one part of the question scores low against the whole of it,
    so the usual floors drop it. These keep it while it still stands out:

    * ``doc_first``: before anything else, the best passage of up to this many documents
      whose best passage clears ``doc_floor`` (absolute) and ``doc_relative`` x the best
      score: one document per part answers more parts than a second passage of the first;
    * ``reserved``: passages kept whatever their score against the whole question (the
      best match of each part, see ``part_bests``), at most ``part_quota`` per part;
    * the rest of the slots go to the usual cut (``select_hits``).
    """

    doc_first: int = 0
    doc_floor: float = 0.0
    doc_relative: float = 0.0
    part_quota: int = 1


def select_parts(
    ranked: list[SearchHit],
    policy: SelectionPolicy,
    parts: PartsPolicy,
    part_hits: Iterable[list[SearchHit]] = (),
) -> list[SearchHit]:
    """``select_hits`` for a question of several parts.

    ``part_hits[i]`` is part i's best passages, best first, each scored by part i's own
    ranking (the reranker on the part's text, or a lexical match); the caller has already
    dropped those below the part's floor. A reserved passage that is also in ``ranked``
    keeps the better of its two scores, so the evidence of every part comes first.
    """
    if not ranked or no_evidence(ranked, policy):
        return []
    usual = select_hits(ranked, policy)
    top = _score(ranked[0])
    by_id = {hit.id: hit for hit in ranked}
    reserved: dict[str, SearchHit] = {}
    for hits in part_hits:
        for hit in list(hits)[: max(0, parts.part_quota)]:
            main = by_id.get(hit.id)
            best = max(_score(hit), _score(main)) if main is not None else _score(hit)
            current = reserved.get(hit.id)
            if current is not None and _score(current) >= best:
                continue
            base = main if main is not None else hit
            reserved[hit.id] = base if _score(base) == best else replace(base, rerank_score=best)
    if parts.doc_first > 0:
        floor = max(parts.doc_floor, parts.doc_relative * top)
        seen: set[str] = set()
        leaders = 0
        for hit in ranked:
            if hit.doc_id in seen:
                continue
            seen.add(hit.doc_id)
            if _score(hit) < floor or leaders >= parts.doc_first:
                break
            leaders += 1
            reserved.setdefault(hit.id, hit)
    if not reserved:
        return usual
    cap = max(policy.max_keep, min(len(reserved), policy.max_keep + 2))
    # Reserved passages first (the best of them if there are too many), then the usual
    # cut fills what is left, keeping its per-document cap.
    kept = sorted(reserved.values(), key=_score, reverse=True)[:cap]
    ids = {hit.id for hit in kept}
    per_doc: dict[str, int] = {}
    for hit in kept:
        per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
    deferred: list[SearchHit] = []
    for hit in usual:
        if len(kept) >= cap:
            break
        if hit.id in ids:
            continue
        if policy.per_document > 0 and per_doc.get(hit.doc_id, 0) >= policy.per_document:
            deferred.append(hit)
            continue
        kept.append(hit)
        ids.add(hit.id)
        per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
    # As in select_hits, the per-document cap never leaves slots empty.
    kept += deferred[: max(0, cap - len(kept))]
    return sorted(kept, key=_score, reverse=True)


# Version marks in file names: "v2", "V1.3", "第2版", "(1)", "副本", "终版"... Dates and
# years are not version marks: "2023年报" and "2024年报", or two weekly reports, are
# different documents.
_VERSION_MARKS = re.compile(
    r"(?<![a-z])(?:v|ver|version|rev)\s*\d+(?:\.\d+)*(?![a-z])"
    r"|第\s*[0-9一二三四五六七八九十]+\s*版"
    r"|[(（]\s*\d{1,2}\s*[)）]"
    r"|副本|拷贝|最终版|最新版|终版|定稿|修订版|修订稿|修订|草稿|初稿|新版|旧版"
    r"|(?<![a-z])(?:copy(?:\s*of)?|final|draft|latest)(?![a-z])",
    re.IGNORECASE,
)
_SEPARATORS = re.compile(r"[\s_\-.·—–、，,()（）\[\]【】]+")


def version_key(name: str) -> str | None:
    """The name with version marks removed (and its type), or None when too little is left.

    Two files share a key only when their names differ by version marks alone:
    "报销制度_v2.docx" and "报销制度（终版）.docx" do; "华南区_报销制度" and
    "华东区_报销制度" do not.
    """
    stem, dot, extension = name.rpartition(".")
    if not dot:
        stem, extension = name, ""
    core = _SEPARATORS.sub("", _VERSION_MARKS.sub(" ", stem.lower()))
    if len(core) < 4:
        return None
    return f"{core}.{extension.lower()}"


# Folders that hold old copies of files kept elsewhere: a same-named file in one of them is an
# older version, not another project's file.
_BACKUP_FOLDER = re.compile(
    r"备份|旧版|旧的|历史|归档|存档|过期|废弃|作废|副本|拷贝|(?<![a-z])(?:old|backup|bak|archive|archived|copy|v\d+)(?![a-z])",
    re.IGNORECASE,
)


def _folder(path: str) -> str:
    """The folder a registered path is in ("C:/资料/02 项目B/04 答辩PPT.pptx" -> "02 项目B")."""
    parts = [part for part in re.split(r"[\\/]+", path or "") if part]
    return parts[-2] if len(parts) >= 2 else ""


def _versions_of_each_other(a: tuple, b: tuple) -> bool:
    """Same version key; but "01 项目A/04 答辩PPT.pptx" and "02 项目B/04 答辩PPT.pptx" (the very same
    name in different folders, neither a backup folder) are different projects' files."""
    name_a, path_a = a[0], (a[2] if len(a) > 2 else "")
    name_b, path_b = b[0], (b[2] if len(b) > 2 else "")
    if name_a.strip().lower() != name_b.strip().lower():
        return True  # the names differ by version marks only: "报销制度_v1" / "报销制度_v2"
    folder_a, folder_b = _folder(path_a), _folder(path_b)
    if not folder_a or not folder_b or folder_a.lower() == folder_b.lower():
        return True
    return bool(_BACKUP_FOLDER.search(folder_a) or _BACKUP_FOLDER.search(folder_b))


def drop_superseded(
    hits: list[SearchHit],
    describe: Callable[[str], tuple | None],
    candidates: Iterable[SearchHit] | None = None,
) -> list[SearchHit]:
    """Remove hits from older versions of a file whose newest version is among the candidates.

    ``describe(doc_id)`` returns the file's name, modification time and (optionally) path
    (None when unknown). ``candidates`` defaults to ``hits``; only versions that were actually
    retrieved count, so an older version that alone holds a passage is kept. The very same
    name in two different folders is two files (two projects' "04 答辩PPT.pptx"), unless one
    of the folders is a backup / old-versions folder.
    """
    pool = list(candidates) if candidates is not None else hits
    info: dict[str, tuple | None] = {}
    groups: dict[str, list[str]] = {}
    for doc_id in dict.fromkeys(hit.doc_id for hit in pool):
        info[doc_id] = described = describe(doc_id)
        key = version_key(described[0]) if described else None
        if key is not None:
            groups.setdefault(key, []).append(doc_id)
    superseded: set[str] = set()
    for members in groups.values():
        for doc_id in members:
            mine = info[doc_id]
            if any(
                other != doc_id
                and float(info[other][1] or 0.0) > float(mine[1] or 0.0)
                and _versions_of_each_other(mine, info[other])
                for other in members
            ):
                superseded.add(doc_id)
    return [hit for hit in hits if hit.doc_id not in superseded]
