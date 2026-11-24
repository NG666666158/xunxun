"""Shared by the survey_recall_v3 scripts: round two's helpers (survey_recall_v2/shared.py, which
loads recall_wording_v1's harness, library and cache), and this folder's work/ and results/.
(Not called shared.py: round two's module of that name is imported from here.)"""

from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "survey_recall_v2"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import shared as v2  # noqa: E402  (survey_recall_v2/shared.py)
from shared import harness, read_rows, write_rows, write_json  # noqa: E402,F401

WORK = HERE / "work"
RESULTS = HERE / "results"
surveys = v2.surveys
