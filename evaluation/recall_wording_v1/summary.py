"""The tables of REPORT.md from results/*.json (Markdown on stdout).

  python evaluation/recall_wording_v1/summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# (label, results file)
RETRIEVAL = [
    ("main (all switches off)", "base"),
    ("step 2: wording_expansion", "wording"),
    ("step 2: + wording_rescore", "wording_rescore"),
    ("step 3: candidate_per_document=8 alone", "cap8"),
    ("step 3: pool 80 alone", "pool80"),
    ("step 3: pool 150 alone", "pool150"),
    ("step 3: wording + cap 8 + pool 150", "wording_cap8_pool150"),
    ("step 5: survey_route alone", "survey_only"),
    ("step 5: wording + rescore + survey_route = new defaults", "final"),
    ("step 5: new defaults without rescore", "final_norescore"),
    ("step 5: new defaults, survey_pool=100", "final_pool100"),
    ("step 5: new defaults, survey_files=10", "final_files10"),
    ("step 4: section_headings alone (re-indexed)", "sec_base"),
    ("step 4: section_headings + new defaults", "sec_final"),
    ("step 4: the same, survey_candidates_per_document=5", "sec_final_cap5"),
]
REGRESSION = [
    ("step 2: wording_expansion", "rw_wording"),
    ("step 2: + wording_rescore", "rw_rescore"),
    ("step 3: candidate_per_document=8", "rw_cap8"),
    ("steps 2 + 5: new defaults", "rw_default"),
]
ANSWERS = [("main", "before"), ("new defaults", "final"), ("new defaults, survey_pool=100", "after"),
           ("new defaults, survey_pool=100, survey_files=10", "after_files10"),
           ("new defaults + section_headings (survey_pool=100, 5 per file)", "after_sections")]


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def retrieval() -> None:
    print("| Configuration | Fact: kept | Fact: rank ≤ 3 | Un-named: kept | Survey: in pool | Survey: kept | Survey: gold files kept (of 3.64) | Survey: ≥ 3 files | Passages scored | Rerank tokens |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, name in RETRIEVAL:
        s = load(name)
        if s is None:
            continue
        f, g, v, c = s["fact"], s["generic"], s["survey"], s["cost"]
        print(f"| {label} | {f['final']:.3f} | {f['top3']:.3f} | {g['final']:.3f} | {v['pool']:.3f} | {v['final']:.3f} | "
              f"{v['gold_files_kept']:.2f} | {v['three_or_more']:.2f} | {c['rerank_docs']:.0f} | {c['rerank_tokens']:,} |")


def regression() -> None:
    print("| Configuration | Questions | Rankings changed | More gold | Fewer gold | books | complex | benchmark | CRUD | MultiHop | Du | T2 |")
    print("|---|---:|---:|---:|---:|---|---|---|---|---|---|---|")
    base = load("regression_rw_base")
    if base:
        cells = " | ".join(f"{r['enough_after']:.4f}" for r in base["sets"])
        print(f"| main (reference run) | {base['total']['questions']:,} | 0 | 0 | 0 | {cells} |")
    for label, name in REGRESSION:
        s = load(f"regression_{name}")
        if s is None:
            continue
        t = s["total"]
        cells = " | ".join(f"{r['enough_after']:.4f} ({r['changed']})" for r in s["sets"])
        print(f"| {label} | {t['questions']:,} | {t['changed']} | {t['more_gold']} | {t['fewer_gold']} | {cells} |")


def answers() -> None:
    print("| Configuration | Survey: refused | Survey: gold files cited | per question | ≥ 3 files cited | Fund surveys: amounts stated | Fact guard: files cited | Fact guard: amounts stated | Median seconds (survey / fact) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, name in ANSWERS:
        s = load(f"answers_{name}")
        if s is None:
            continue
        v, fund, g = s["survey"], s["survey_fund"], s.get("fact_guard") or {}
        guard = f"{g['files_cited']} / {g['gold_files']} | {g['amounts_right']} / {g['amounts']}" if g else "- | -"
        seconds = f"{v['seconds_median']} / {g['seconds_median']}" if g else f"{v['seconds_median']} / -"
        print(f"| {label} | {v['refused']} / {v['questions']} | {v['files_cited']} / {v['gold_files']} | {v['files_cited_per_question']:.2f} | "
              f"{v['three_or_more_files']} / {v['questions']} | {fund['amounts_right']} / {fund['amounts']} | {guard} | {seconds} |")


if __name__ == "__main__":
    for title, table in (("Retrieval on the synthetic library", retrieval), ("Regression (share of answerable questions that keep what they need; changed rankings in brackets)", regression),
                         ("Answers on Qwen3-8B", answers)):
        print(f"\n### {title}\n")
        table()
    for name in ("rewrite", "latency"):
        s = load(name)
        if s:
            print(f"\n### {name}\n\n```json\n{json.dumps(s, ensure_ascii=False, indent=1)}\n```")
