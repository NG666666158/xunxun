"""evaluation/suite: metrics on hand-computed cases, the schema, and the offline CI tier."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

SUITE = Path(__file__).resolve().parents[2] / "evaluation" / "suite"
if str(SUITE) not in sys.path:
    sys.path.insert(0, str(SUITE))

import metrics  # noqa: E402
import schema  # noqa: E402

POLICY = {"rerank_min_score": 0.1, "rerank_relative_score": 0.1}


def items(*golds):
    return [{"d": f"d{i}", "g": list(g)} for i, g in enumerate(golds)]


def test_ranking_metrics_single_gold():
    # The gold document is third.
    m = metrics.ranking_metrics(items((), (), (0,), ()), n_gold=1, needs_all=False)
    assert m["recall@1"] == 0 and m["recall@3"] == 1 and m["recall@5"] == 1
    assert m["precision@3"] == pytest.approx(1 / 3)
    assert m["precision@5"] == pytest.approx(1 / 5)  # missing slots count as not relevant
    assert m["mrr"] == pytest.approx(1 / 3)
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(4))
    assert m["kept_precision"] == pytest.approx(1 / 4)


def test_ranking_metrics_multi_gold_needs_all():
    # Two golds: the first at rank 1, the second at rank 4; a duplicate of gold 0 at rank 2.
    m = metrics.ranking_metrics(items((0,), (0,), (), (1,)), n_gold=2, needs_all=True)
    assert m["recall@1"] == 0.5 and m["recall@3"] == 0.5 and m["recall@5"] == 1.0
    assert m["success@3"] == 0.0 and m["success@5"] == 1.0
    assert m["hit@1"] == 1.0
    assert m["precision@3"] == pytest.approx(2 / 3)
    # The duplicate gains nothing: DCG = 1 + 1/log2(5); ideal = 1 + 1/log2(3).
    assert m["ndcg@10"] == pytest.approx((1 + 1 / math.log2(5)) / (1 + 1 / math.log2(3)))


def test_collapse_keeps_first_passage_per_document():
    hits = [{"d": "a", "g": []}, {"d": "a", "g": [0]}, {"d": "b", "g": []}]
    assert [h["d"] for h in metrics.collapse(hits, "doc")] == ["a", "b"]
    assert metrics.collapse(hits, "passage") == hits


def test_negatives_count_passages_over_the_cutoff_and_auc():
    assert metrics.above_cut([0.9, 0.2, 0.05], 0.1, 0.1) == 2
    assert metrics.above_cut([0.5, 0.06, 0.04], 0.1, 0.1) == 1
    assert metrics.above_cut([0.05, 0.04], 0.1, 0.1) == 0
    assert metrics.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert metrics.auc([0.5], [0.5]) == 0.5


def test_summarize_separates_answerable_and_negatives():
    def record(answerable, golds, scores):
        pool = [{"c": str(i), "d": f"d{i}", "p": f"d{i}-p0", "g": list(g), "r": s} for i, (g, s) in enumerate(zip(golds, scores))]
        return {"id": str(answerable), "type": "t", "tags": [], "answerable": answerable, "needs_all": False,
                "n_gold": 1 if answerable else 0, "unit": "doc",
                "stages": {"pool": pool, "reranked": pool, "final": pool[:3], "context": pool[:2]},
                "latency_ms": {"total": 10.0}}

    s = metrics.summarize([record(True, [(0,), (), ()], [0.9, 0.2, 0.01]), record(False, [(), (), ()], [0.3, 0.02, 0.01])], POLICY)
    assert s["answerable"] == 1 and s["negatives"] == 1
    assert s["final"]["recall@5"] == 1.0 and s["final"]["mrr"] == 1.0
    assert s["negative"]["returned"] == 3 and s["negative"]["above_cut"] == 1
    assert s["negative"]["auc_top_score"] == 1.0
    assert s["latency_ms"]["total"]["p50"] == 10.0


def test_percentile_is_nearest_rank():
    values = list(range(1, 101))
    assert metrics.percentile(values, 50) == 50
    assert metrics.percentile(values, 90) == 90
    assert metrics.percentile(values, 99) == 99


def test_schema_rejects_answerable_question_without_gold():
    with pytest.raises(ValueError):
        schema.question(id="x", query="q", type="single", source={"dataset": "x"})
    row = schema.question(id="x", query="q", type="null", answerable=False, source={"dataset": "x"})
    assert row["gold_doc_ids"] == [] and row["language"] == "zh"


def test_ci_fixture_follows_the_schema():
    fixture = SUITE / "fixtures" / "ci"
    corpus = [json.loads(line) for line in (fixture / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if line]
    questions = [json.loads(line) for line in (fixture / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    ids = {row["doc_id"] for row in corpus}
    for row in corpus:
        schema.validate_corpus_row(row)
    for row in questions:
        schema.validate_question(row)
        assert set(row["gold_doc_ids"]) <= ids
    assert {q["language"] for q in questions} == {"zh", "en"}
    assert any(not q["answerable"] for q in questions) and any(q["needs_all"] for q in questions)


def test_offline_ci_tier_reproduces_expected(tmp_path):
    """Index build + retrieval + metrics on the fixture, from cached model outputs only."""
    result = subprocess.run(
        [sys.executable, "-I", str(SUITE / "ci_tier.py"), "check", "--work", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "unchanged" in result.stdout
