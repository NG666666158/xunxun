from retrieval.selection import (
    SelectionPolicy,
    candidate_pool_size,
    drop_superseded,
    select_hits,
    version_key,
)
from retrieval.types import SearchHit


def hit(identifier, score, doc="a"):
    return SearchHit(id=str(identifier), score=0.0, text=f"t{identifier}", doc_id=doc, parent_id=f"{doc}-p{identifier}",
                     source=f"{doc}.pdf", heading="", locator="", retrieval_method="test", rerank_score=score)


def ids(hits):
    return [h.id for h in hits]


def test_pool_size_stays_at_base_for_small_libraries_and_is_capped():
    assert candidate_pool_size(40, None, 80) == 40
    assert candidate_pool_size(40, 5000, 80) == 40
    assert candidate_pool_size(40, 20000, 80) == 40
    assert candidate_pool_size(40, 40000, 80) == 50  # one doubling: +25%
    assert candidate_pool_size(40, 80000, 80) == 60
    assert candidate_pool_size(40, 10_000_000, 80) == 80
    assert candidate_pool_size(100, 10_000_000, 80) == 100  # an explicit base above the cap wins


def test_select_keeps_scores_above_floor_and_ratio_within_bounds():
    ranked = [hit(1, 0.9), hit(2, 0.5), hit(3, 0.3), hit(4, 0.1), hit(5, 0.01)]
    # cutoff = max(0.05, 0.3 * 0.9 = 0.27)
    assert ids(select_hits(ranked, SelectionPolicy(0.05, 0.3, 1, 10))) == ["1", "2", "3"]
    assert ids(select_hits(ranked, SelectionPolicy(0.05, 0.3, 1, 2))) == ["1", "2"]  # max_keep
    # floor binds when the ratio is low
    assert ids(select_hits(ranked, SelectionPolicy(0.05, 0.0, 1, 10))) == ["1", "2", "3", "4"]


def test_select_keeps_min_keep_even_when_everything_is_weak():
    ranked = [hit(1, 0.04), hit(2, 0.03), hit(3, 0.02), hit(4, 0.01)]
    assert ids(select_hits(ranked, SelectionPolicy(0.1, 0.2, 3, 8))) == ["1", "2", "3"]
    assert select_hits([], SelectionPolicy(0.1, 0.2, 3, 8)) == []


def test_per_document_cap_makes_room_for_other_documents():
    ranked = [hit(1, 0.9, "a"), hit(2, 0.85, "a"), hit(3, 0.8, "a"), hit(4, 0.7, "b"), hit(5, 0.6, "a"), hit(6, 0.5, "c")]
    kept = select_hits(ranked, SelectionPolicy(0.0, 0.0, 1, 4, per_document=2))
    assert ids(kept) == ["1", "2", "4", "6"]


def test_per_document_cap_never_leaves_slots_empty():
    # One document only (e.g. a question scoped to one book): the cap must not starve the context.
    ranked = [hit(i, 1.0 - i / 10, "a") for i in range(6)]
    kept = select_hits(ranked, SelectionPolicy(0.0, 0.0, 1, 5, per_document=2))
    assert ids(kept) == ["0", "1", "2", "3", "4"]
    # Deferred hits still have to clear the score cutoff.
    ranked = [hit(1, 0.9, "a"), hit(2, 0.8, "a"), hit(3, 0.7, "a"), hit(4, 0.01, "b")]
    assert ids(select_hits(ranked, SelectionPolicy(0.1, 0.0, 1, 4, per_document=1))) == ["1", "2", "3"]


def test_version_key_ignores_version_marks_only():
    assert version_key("差旅报销制度_v2.docx") == version_key("差旅报销制度（终版）.docx") == version_key("差旅报销制度.docx")
    assert version_key("差旅报销制度 V1.3.docx") == version_key("差旅报销制度 - 副本.docx")
    assert version_key("员工手册(1).pdf") == version_key("员工手册.pdf")
    assert version_key("员工手册 第2版.pdf") == version_key("员工手册 第3版.pdf")
    # Different documents stay apart: regions, years, types, words that merely contain "v2".
    assert version_key("008_FIN_华南区_费用报销制度.pdf") != version_key("007_FIN_华东区_费用报销制度.pdf")
    assert version_key("2023年度报告.pdf") != version_key("2024年度报告.pdf")
    assert version_key("周报_20240301.docx") != version_key("周报_20240308.docx")
    assert version_key("报销制度.pdf") != version_key("报销制度.docx")
    assert version_key("devops手册.md") != version_key("deops手册.md")
    assert version_key("v2.txt") is None  # nothing left to compare


def test_drop_superseded_keeps_only_the_newest_retrieved_version():
    files = {"old": ("报销制度_v1.docx", 100.0), "new": ("报销制度_v2.docx", 200.0), "other": ("考勤办法.docx", 50.0)}
    describe = files.get
    hits = [hit(1, 0.9, "old"), hit(2, 0.8, "new"), hit(3, 0.7, "other"), hit(4, 0.6, "old")]
    assert ids(drop_superseded(hits, describe)) == ["2", "3"]
    # The newest version is not among the candidates: the older one is all there is, keep it.
    assert ids(drop_superseded([hit(1, 0.9, "old"), hit(3, 0.7, "other")], describe)) == ["1", "3"]
    # The candidate pool decides, not only the kept hits.
    pool = hits + [hit(5, 0.1, "new")]
    assert ids(drop_superseded([hit(1, 0.9, "old")], describe, pool)) == []
    # Unknown documents (no registry row) are never dropped.
    assert ids(drop_superseded([hit(1, 0.9, "x"), hit(2, 0.8, "y")], lambda doc_id: None)) == ["1", "2"]


def test_no_evidence_floor_keeps_nothing_instead_of_min_keep():
    weak = [hit(1, 0.09), hit(2, 0.05), hit(3, 0.01)]
    assert select_hits(weak, SelectionPolicy(0.1, 0.1, 3, 6, no_evidence_below=0.1)) == []
    # Without the floor, min_keep hands the weakest passages on as before.
    assert ids(select_hits(weak, SelectionPolicy(0.1, 0.1, 3, 6))) == ["1", "2", "3"]
    # The floor looks at the best passage only: the others follow the usual cutoffs.
    ranked = [hit(0, 0.4)] + weak
    assert ids(select_hits(ranked, SelectionPolicy(0.1, 0.1, 3, 6, no_evidence_below=0.1))) == ["0", "1", "2"]
