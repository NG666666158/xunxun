"""The tables of REPORT.md from results/*.json (Markdown on stdout).

  python evaluation/survey_recall_v2/summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

STEPS = ("search", "cut", "file_passage", "refused", "floor", "shape", "files", "passage", "kept")
# (label, losses results name)
RETRIEVAL = [
    ("main (round one's defaults)", "main"),
    ("1 survey_file_first (16 files, 3 passages each)", "files"),
    ("  1 + survey_question_files=8", "files_q8"),
    ("  1 with survey_file_candidates=24", "files_f24"),
    ("  1 with survey_passages_per_file=5", "files_k5"),
    ("  1 + the ask's words for who / time questions (removed)", "files_ask"),
    ("2a 1 + survey_excerpt_scoring", "files_xs"),
    ("2b 2a with survey_passages_per_file=5", "files_xs_k5"),
    ("2c 2b + survey_answering_files", "xs_ans"),
    ("2d 2c + survey_question_files=8 (candidate)", "xs_ans_q8"),
    ("2e 2d + survey_about_asked", "files_xs_ans_q8_about"),
]
ANSWERS = [
    ("main (round one's defaults)", "main"),
    ("1 + 3: survey_file_first, survey_keep_route", "files_keep"),
    ("candidate without survey_excerpts", "final_noexcerpt"),
    ("candidate: 1 + 2d + 3 + survey_excerpts", "final"),
]


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def losses() -> None:
    print("| Configuration | Set | " + " | ".join(STEPS) + " | ≥ 3 gold files kept | all kept |")
    print("|---|---|" + "---:|" * (len(STEPS) + 2))
    for label, name in RETRIEVAL:
        for suffix, title, questions in (("", "tuning", 22), ("_holdout", "held out", 16)):
            s = load(f"losses_{name}{suffix}")
            if s is None:
                continue
            cells = " | ".join(str(s["steps"].get(step, 0)) for step in STEPS)
            print(f"| {label} | {title} ({s['gold_files']}) | {cells} | {s['three_or_more']} / {questions} | {s['all_kept']} |")


def shortfall(name: str = "main") -> None:
    """main's questions that keep fewer than three gold files: where each missing one is lost."""
    s = load(f"losses_{name}")
    if not s or "rows" not in s:
        return
    print("| Question | Gold files | Kept | Each gold file not kept: where it is lost | Kept but not cited by the answer |")
    print("|---|---:|---:|---|---|")
    for row in s["rows"]:
        kept = [g for g in row["gold"] if g["step"] == "kept"]
        if len(kept) >= 3:
            continue
        lost = "; ".join(f"{g['file'].rsplit('.', 1)[0]}: {g['step']}" for g in row["gold"] if g["step"] != "kept") or "-"
        unused = "; ".join(g["file"].rsplit(".", 1)[0] for g in row["gold"] if g["step"] == "kept" and g.get("use") == "unused") or "-"
        print(f"| {row['id']} {row['question']} | {len(row['gold'])} | {len(kept)} | {lost} | {unused} |")


def negatives() -> None:
    print("| Configuration | Questions with nothing kept | Files kept per question |")
    print("|---|---:|---|")
    for label, name in RETRIEVAL:
        s = load(f"losses_{name}_negative")
        if s is not None:
            print(f"| {label} | {s['kept_nothing']} / {s['questions']} | {s['kept_files']} |")


def answers() -> None:
    print("| Configuration | Set | Refused | Gold files cited | ≥ 3 files cited | Amounts stated | Kept files unused | Fact guard: amounts | Median s (survey / fact) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, name in ANSWERS:
        for suffix, title in (("", "tuning"), ("_holdout", "held out")):
            s = load(f"answers_{name}{suffix}")
            if s is None:
                continue
            v, g = s["survey"], s.get("fact_guard") or {}
            guard = f"{g['amounts_right']} / {g['amounts']}" if g else "-"
            unused = s.get("kept_unused", "-")
            print(f"| {label} | {title} | {v['refused']} / {v['questions']} | {v['files_cited']} / {v['gold_files']} | "
                  f"{v['three_or_more_files']} / {v['questions']} | {v['amounts_right']} / {v['amounts']} | {unused} | {guard} | "
                  f"{v['seconds_median']} / {g.get('seconds_median', '-')} |")


if __name__ == "__main__":
    for title, table in (("main: the questions that keep fewer than three gold files", shortfall),
                         ("Where gold files are lost (retrieval)", losses), ("Questions about programmes the library lacks", negatives),
                         ("Answers on Qwen3-8B", answers)):
        print(f"\n### {title}\n")
        table()
    for path in sorted(RESULTS.glob("latency_*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n### {path.stem}\n\n```json\n{json.dumps({k: v for k, v in s.items() if k != 'rows'}, ensure_ascii=False, indent=1)}\n```")
