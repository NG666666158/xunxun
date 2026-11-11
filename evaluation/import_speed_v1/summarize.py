"""Keep the parts of profile_import.py runs the report quotes, in results/<name>.json.

    python -I evaluation/import_speed_v1/summarize.py <run.json> [...]

Drops the fingerprint (vectors) and keeps about 40 points of the progress curve.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent / "results"
KEEP = (
    "fake_embed", "fake_tpm", "fake_rerank", "overrides", "image_mode", "suggestions", "wall_seconds",
    "first_indexed_seconds", "t_50pct_indexed", "t_90pct_indexed", "t_100pct_indexed", "inspect_seconds",
    "scan_done_seconds", "files", "chunks", "failed", "peak_rss_mb", "pacer", "tokens_seen", "answering",
    "during_import", "doc_cards", "drain", "database_locked_warnings", "suggestions_built", "estimate", "stages",
)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for raw in sys.argv[1:]:
        path = Path(raw)
        run = json.loads(path.read_text(encoding="utf-8"))
        kept = {key: run[key] for key in KEEP if key in run}
        kept["code"] = "main e5cbe1f" if "base" in Path(run.get("project", "")).parts else "claude/import-speed-v1"
        progress = run.get("progress") or []
        step = max(1, len(progress) // 40)
        kept["progress"] = progress[::step] + ([progress[-1]] if progress and (len(progress) - 1) % step else [])
        (OUT / path.name).write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
        print(path.name, kept.get("wall_seconds"))


if __name__ == "__main__":
    main()
