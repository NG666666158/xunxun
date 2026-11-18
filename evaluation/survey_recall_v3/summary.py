"""Round three's tables, from results/*.json.

  python evaluation/survey_recall_v3/summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
V2 = Path(__file__).resolve().parent.parent / "survey_recall_v2" / "results"
STEPS = ("search", "cut", "file_passage", "refused", "floor", "passage", "files", "kept")


def load(name: str, folder: Path = RESULTS) -> dict | None:
    path = folder / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def retrieval() -> None:
    rows = [("round two (v2 defaults)", "v2", "wideq_v2", "new_v2"),
            ("+ survey_wide", None, "wideq_wide", None),
            ("+ survey_topic_related (16 candidates)", "related", "wideq_wide_related", None),
            ("  with survey_file_candidates=24", "related_c24", None, None),
            ("round three (all, survey_topic_passages=2)", "v3", "wideq_v3", "new_v3")]
    print("| Configuration | Tuning (80) | ≥ 3 | Held out (56) | ≥ 3 | Negatives: nothing kept | Cue-less (96) | ≥ 3 | New set (50) | ≥ 3 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, base, wide, new in rows:
        cells = []
        for name in ((base, base and base + "_holdout") if base else (None, None)):
            s = load(f"losses_{name}") if name else None
            cells += [f"{s['steps']['kept']}" if s else "", f"{s['three_or_more']} / {s['questions']}" if s else ""]
        s = load(f"losses_{base}_negative") if base else None
        cells.append(f"{s['kept_nothing']} / {s['questions']}" if s else "")
        for name in (wide, new):
            s = load(f"losses_{name}") if name else None
            cells += [f"{s['steps']['kept']}" if s else "", f"{s['three_or_more']} / {s['questions']}" if s else ""]
        print(f"| {label} | " + " | ".join(cells) + " |")


def steps() -> None:
    print("| Configuration | Set | " + " | ".join(STEPS) + " |")
    print("|---|---|" + "---:|" * len(STEPS))
    for name, label in (("v2", "round two"), ("v3", "round three")):
        for suffix, title in (("", "tuning"), ("_holdout", "held out")):
            s = load(f"losses_{name}{suffix}")
            if s:
                print(f"| {label} | {title} | " + " | ".join(str(s["steps"].get(k, 0)) for k in STEPS) + " |")
    for name, label in (("new_v2", "round two"), ("new_v3", "round three")):
        s = load(f"losses_{name}")
        if s:
            print(f"| {label} | new set | " + " | ".join(str(s["steps"].get(k, 0)) for k in STEPS) + " |")


ANSWERS = [("round two", "final", V2, ""), ("round three", "cand", RESULTS, ""),
           ("round three without survey_missing_files", "cand_nomissing", RESULTS, ""),
           ("round two", "final", V2, "_holdout"), ("round three", "cand", RESULTS, "_holdout"),
           ("round two", "v2_new", RESULTS, ""), ("round three", "cand_new", RESULTS, ""),
           ("round two", "v2_wide", RESULTS, ""), ("round three", "cand_wide", RESULTS, "")]
SET_NAMES = {("final", ""): "tuning", ("cand", ""): "tuning", ("cand_nomissing", ""): "tuning", ("final", "_holdout"): "held out",
             ("cand", "_holdout"): "held out", ("v2_new", ""): "new set", ("cand_new", ""): "new set",
             ("v2_wide", ""): "cue-less", ("cand_wide", ""): "cue-less"}


def answers() -> None:
    print("| Set | Configuration | Refused | Gold files cited | ≥ 3 files cited | Amounts stated | Median s |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for label, name, folder, suffix in ANSWERS:
        s = load(f"answers_{name}{suffix}", folder)
        if not s:
            continue
        v = s["survey"]
        print(f"| {SET_NAMES[(name, suffix)]} | {label} | {v['refused']} / {v['questions']} | {v['files_cited']} / {v['gold_files']} | "
              f"{v['three_or_more_files']} / {v['questions']} | {v['amounts_right']} / {v['amounts']} | {round(v['seconds_median'], 1)} |")
    print()
    print("| | Fact guard (16): amounts | Negatives (8): refused | answered with citations |")
    print("|---|---:|---:|---:|")
    # Round two's fact guard ran with its candidate before answer repair (answers_norepair).
    for label, folder, name, guard_run in (("round two", V2, "final", "norepair"), ("round three", RESULTS, "cand", "cand")):
        s, n = load(f"answers_{guard_run}", folder), load(f"answers_{name}_negative", folder)
        g = (s or {}).get("fact_guard") or {}
        guard = f"{g['amounts_right']} / {g['amounts']}" if g else "-"
        neg = (n or {}).get("negative") or {}
        print(f"| {label} | {guard} | {neg.get('refused', '-')} / {neg.get('questions', '-')} | {neg.get('answered_with_citations', '-')} |")


def costs() -> None:
    s = load("latency_v3")
    if s:
        print("| | median | slowest | rerank requests | passages scored | rerank tokens |")
        print("|---|---:|---:|---:|---:|---:|")
        for key, label in (("before", "round two"), ("after", "round three")):
            v = s[key]
            print(f"| {label} | {v['median_s']} s | {v['max_s']} s | {v['rerank_requests']} | {v['scored_mean']} | {v['rerank_tokens_mean']:,} |")
    r = load("regression_sv3_v3")
    if r:
        print()
        print("| Set | Questions | Rankings changed | More gold | Fewer gold | Keeps what it needs (before / after) | Unanswerable left empty |")
        print("|---|---:|---:|---:|---:|---|---|")
        for d in r["sets"]:
            print(f"| {d['dataset']} | {d['questions']} | {d['changed']} | {d['more_gold']} | {d['fewer_gold']} | "
                  f"{d['enough_before']} / {d['enough_after']} | {d['unanswerable_empty_before']} / {d['unanswerable_empty_after']} |")
        total = r["total"]
        print(f"| all | {total['questions']} | {total['changed']} | {total['more_gold']} | {total['fewer_gold']} | | |")


if __name__ == "__main__":
    for title, table in (("Retrieval", retrieval), ("Where gold files are lost", steps), ("Answers on Qwen3-8B", answers), ("Cost", costs)):
        print(f"\n### {title}\n")
        table()
