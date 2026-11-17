"""Shared by the survey_recall_v2 scripts.

The library, its questions, its index and the cache of model outputs are those of
evaluation/recall_wording_v1 (make_corpus.py, harness.py); this folder adds what round two
measures. Per-question records go to work/ here (not committed), summaries to results/.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "recall_wording_v1"
for folder in (V1, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import harness  # noqa: E402  (recall_wording_v1/harness.py; puts project/ on sys.path)

WORK = HERE / "work"
RESULTS = HERE / "results"


def surveys() -> list[dict]:
    return [q for q in harness.load_questions() if q["type"] == "survey"]


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
