"""关于 › 导出诊断信息: a zip someone can send when asking for help.

It holds the logs, a short summary of the app and the per-stage timings of the last answers
(``answer_timings.json``: numbers and question types, see core/timings.py), never keys or
document text. Logs are not
trusted to be clean, because an exception's message can quote a model's reply and so a
document. Each line therefore passes an allow-list: log headlines (capped, long runs of text
removed) and traceback positions stay; everything else is dropped and counted.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
from typing import Any, Iterable
import zipfile

from core import timings

EXPORT_DIR = "diagnostics"
KEEP_EXPORTS = 3
HEADLINE_LIMIT = 240

_HEADLINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} [A-Z]+ [\w.\-]+: ")
_TRACE_FRAME = re.compile(r'^\s+File "[^"]*", line \d+(, in .*)?$')
_EXCEPTION = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt|Timeout|Refused|Failed|Busy|Damaged))\b")
_TRACE_MARK = re.compile(r"^(Traceback \(most recent call last\):|The above exception was the direct cause.*|During handling of the above exception.*)$")
_SECRET = re.compile(
    r"sk-[A-Za-z0-9_\-]{6,}"
    r"|(?i:authorization[\"']?\s*[=:]\s*[\"']?(?:bearer\s+)?)[^\s\"',}]+"
    r"|(?i:bearer)\s+[A-Za-z0-9._\-]+"
    r"|(?i:(?:api[_-]?key|x-app-token|token|password|secret)[\"']?\s*[=:]\s*[\"']?)[^\s\"',}]+"
)
# Pydantic and parsers quote what they failed on.
_QUOTED_INPUT = re.compile(r"(?i)(input_value|completion|content|text)\s*[=:]\s*.*$")
# File names rarely run this long without a break; sentences do.
_LONG_CJK = re.compile(r"[　-〿㐀-鿿＀-￯]{30,}")


def scrub_text(text: str, secrets: Iterable[str] = (), home: str = "") -> str:
    """Secrets, then the user's home folder, removed from one piece of text."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[已隐去]")
    text = _SECRET.sub("[已隐去]", text)
    if home:
        text = re.sub(re.escape(home), "%USERPROFILE%", text, flags=re.IGNORECASE)
    return text


def scrub_log(lines: Iterable[str], secrets: Iterable[str] = (), home: str = "") -> list[str]:
    """Only what the allow-list knows. Dropped lines leave a count behind."""
    kept: list[str] = []
    dropped = 0

    def flush() -> None:
        nonlocal dropped
        if dropped:
            kept.append(f"  [省略 {dropped} 行]")
            dropped = 0

    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            continue
        if _HEADLINE.match(line):
            flush()
            line = scrub_text(line, secrets, home)
            line = _QUOTED_INPUT.sub(lambda m: f"{m.group(1)}=[已隐去]", line)
            line = _LONG_CJK.sub("[文本已隐去]", line)
            kept.append(line if len(line) <= HEADLINE_LIMIT else line[:HEADLINE_LIMIT] + "…")
        elif _TRACE_FRAME.match(line) or _TRACE_MARK.match(line):
            flush()
            kept.append(scrub_text(line, secrets, home))
        elif match := _EXCEPTION.match(line):
            flush()
            kept.append(match.group(1))
        else:
            dropped += 1
    flush()
    return kept


def _summary(settings: Any, data_safety: dict[str, Any], files: dict[str, int] | None) -> dict[str, Any]:
    db = Path(settings.embedded_db_path)
    return {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "app_version": getattr(settings, "app_version", ""),
        "os": platform.platform(),
        "python": platform.python_version(),
        "models": {
            "llm": settings.llm_model,
            "embedding": settings.embedding_model,
            "rerank": settings.rerank_model,
        },
        "answer_mode": settings.answer_mode,
        "image_understanding": settings.image_understanding,
        "database_bytes": db.stat().st_size if db.exists() else 0,
        "files_by_status": files or {},
        "data_safety": {
            "state": data_safety.get("state"),
            "checked_at": data_safety.get("checked_at"),
            "check_ms": data_safety.get("check_ms"),
            "backup_note": data_safety.get("backup_note"),
            "backups": [
                {key: backup.get(key) for key in ("created_at", "reason", "app_version", "size")}
                for backup in data_safety.get("backups") or []
            ],
        },
    }


def export(settings: Any, data_safety: dict[str, Any], files: dict[str, int] | None = None) -> Path:
    """Write the zip into the data folder and return its path; older exports are removed."""
    secrets = [getattr(settings, "siliconflow_api_key", ""), getattr(settings, "app_token", "")]
    home = os.path.expanduser("~")
    folder = Path(settings.data_dir) / EXPORT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"寻寻诊断-{datetime.now():%Y%m%d-%H%M%S}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "about.json",
            scrub_text(json.dumps(_summary(settings, data_safety, files), ensure_ascii=False, indent=2), secrets, home),
        )
        # Per-stage timings of the last answers: numbers, question types and model names only.
        records = timings.load(settings.log_dir)
        if records:
            archive.writestr(
                "answer_timings.json",
                json.dumps({"overview": timings.overview(records), "recent": records}, ensure_ascii=False, indent=1),
            )
        for log in sorted(Path(settings.log_dir).glob("backend.log*")):
            try:
                with log.open(encoding="utf-8", errors="replace") as handle:
                    archive.writestr(f"logs/{log.name}", "\n".join(scrub_log(handle, secrets, home)) + "\n")
            except OSError:
                archive.writestr(f"logs/{log.name}.unreadable", "")
    for old in sorted(folder.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)[KEEP_EXPORTS:]:
        old.unlink(missing_ok=True)
    return target
