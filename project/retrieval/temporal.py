"""Local time constraints and document dates. No model calls or re-embedding.

Dates are read from the basename and explicit fields near the start of saved parsed
text, never from parent directory names or an implicit filesystem timestamp.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta
import json
from pathlib import Path, PureWindowsPath
import re
import threading

from .file_filters import find_file_cue, name_match, parse_query

DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})[年./-](\d{1,2})[月./-](\d{1,2})日?(?!\d)|(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)")
FIELD = re.compile(r"(?:会议(?:日期|时间)|报告日期|发布日期|签订日期|签署日期|检测日期|统计日期|编制日期|出具日期|开具日期|填发日期|记录日期|(?<!\w)日期)\s*[:：|\s]*\s*(" + DATE.pattern + r")")
RELATIVE = re.compile(r"今天|今日|昨天|昨日|前天|本周|这周|这个?星期|本星期|这个?礼拜|上上周|上一?周|上一?个?星期|上一?个?礼拜|本月|这个月|当月|上一?个?月|上上个月|今年|去年|前年|本年度|(?:最近|过去|近)\s*[\d一二两三四五六七八九十]+\s*(?:个)?(?:天|日|周|星期|月|年)")
TEXT = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.){0,4000})')
DOCUMENT_SUBJECT = re.compile(r"文件|文档|资料|会议|例会|纪要|报告|证明|合同|记录|清单|通知")
# A business action's date can differ from the date of the document describing it.
BUSINESS_ACTION = re.compile(r"(?:[交缴付收退].{0,6}[税费款钱]|纳税)(?!证明|凭证)|营收|营业收入|销售额|发货|到货|交付|出入库")
YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})年(?:度)?(?!\d)")


def dates(text: str) -> set[str]:
    found = set()
    for match in DATE.finditer(text):
        parts = match.groups()[:3] if match[1] else match.groups()[3:]
        try:
            found.add(datetime(*map(int, parts)).date().isoformat())
        except ValueError:
            pass
    return found


def request_time(question: str, now: datetime) -> dict:
    """Keep ordinary historical/content dates and bare '最近/最新' on their old path."""
    relative = RELATIVE.search(question)
    phrases = {m[0] for m in RELATIVE.finditer(question)}
    if len(phrases) > 1 and len({parse_query(p, now).time_range for p in phrases}) > 1:
        return {}  # Comparisons need several periods, not one hard document scope.
    explicit = DATE.search(question)
    if not relative and not (explicit and question[explicit.end():].lstrip().startswith("的")):
        return {}
    parsed = parse_query(question, now)
    if not parsed.time_range:
        return {}
    start, end = parsed.time_range
    action = BUSINESS_ACTION.search(question)
    # "本周报告中纳税情况" dates the report; "今年缴税的证明" dates payments.
    anchor = relative or explicit
    subject = DOCUMENT_SUBJECT.search(question, anchor.end())
    dates_document = subject and (action is None or subject.start() < action.start())
    document_time = parsed.time_field != "any" or find_file_cue(question) == "strong" or dates_document or (not action and (
        parsed.doc_type or parsed.file_types or DOCUMENT_SUBJECT.search(question)))
    return {"phrase": parsed.time_phrase, "start": start.date().isoformat(),
            "end": end.date().isoformat(), "field": parsed.time_field,
            "basis": "document" if document_time else "content",
            "now": now.isoformat(), "keywords": parsed.name_keywords,
            "subject": subject[0].replace("例会", "会议") if subject else ""}


def label(start: str, end: str) -> str:
    last = (datetime.fromisoformat(end) - timedelta(days=1)).date().isoformat()
    return start if start == last else f"{start} 至 {last}"


def in_range(value: str, start: str, end: str) -> bool:
    """A year-only document belongs to an encompassing year, never an invented day."""
    if len(value) == 4 and value.isdigit():
        return start <= value + "-01-01" and f"{int(value) + 1}-01-01" <= end
    return len(value) == 10 and start <= value < end


class DocumentDates:
    """Bounded date-only cache; changed parsed files/names invalidate their entries."""
    def __init__(self, parsed_dir: Path | None):
        self.directory = Path(parsed_dir) if parsed_dir else None
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def get(self, row: dict) -> str:
        doc_id = str(row["file_id"])
        name = PureWindowsPath(str(row.get("name") or "")).name
        revision = row.get("updated_at")
        key = (doc_id, name, revision, row.get("mtime"), row.get("size")) if revision else None
        if key:
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
        path = self.directory / f"{doc_id}.json" if self.directory and re.fullmatch(r"[0-9a-f]{24,64}", doc_id) else None
        try:
            stat = path.stat() if path else None
        except OSError:
            stat = None
        key = key or (doc_id, name, stat.st_mtime_ns if stat else None, stat.st_size if stat else None)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        named = dates(name)
        explicit, compiled = set(), set()
        if stat:
            try:
                # Bounded local read, including for legacy files; do not parse the whole document.
                with path.open(encoding="utf-8") as handle:
                    prefix = handle.read(16384)
                text = "\n".join(json.loads('"' + re.sub(r'\\u[0-9a-fA-F]{0,3}$', '', m[1]) + '"')
                                 for m in TEXT.finditer(prefix))[:4000]
                for match in FIELD.finditer(text):
                    (compiled if match[0].startswith("编制日期") else explicit).update(dates(match[0]))
            except (OSError, ValueError):
                pass
        # A meeting's occurrence date outranks the later drafting date. Conflicts
        # between the filename and event fields still remain unresolved.
        combined = named | explicit or compiled
        result = next(iter(combined)) if len(combined) == 1 else ""
        if not combined and not DATE.search(name):
            years = {m[1] for m in YEAR.finditer(name)}
            result = next(iter(years)) if len(years) == 1 else ""
        with self._lock:
            self._cache[key] = result
            while len(self._cache) > 50000:
                self._cache.popitem(last=False)
        return result

    def select(self, request: dict, rows: list[dict], scope: list[str]) -> dict:
        start, end = request["start"], request["end"]
        known = {}
        eligible = []
        allowed = set(scope) if scope else None
        if request.get("basis") == "content":
            # A certificate issued this year may describe last year's payments.
            # Keep documents available; the answer/checker must inspect dates in their content.
            ids = sorted(str(row["file_id"]) for row in rows
                         if (allowed is None or row["file_id"] in allowed)
                         and row.get("content_mode") != "name"
                         and row.get("status") not in ("named", "recycled", "deleted"))
            return {**request, "dates": {}, "doc_ids": ids, "fallback": False,
                    "note": f"问题中的{request['phrase']}指 {label(start, end)}，以下按资料记载的实际发生日期核对。",
                    "empty_message": "当前范围内没有可用于回答的资料。"}
        for row in rows:
            if allowed is not None and row["file_id"] not in allowed:
                continue
            if row.get("content_mode") == "name" or row.get("status") in ("named", "recycled", "deleted"):
                continue
            field = request["field"]
            if field in ("added", "modified"):
                value = row.get("created_at" if field == "added" else "mtime")
                try:
                    dt = datetime.fromtimestamp(value) if isinstance(value, (int, float)) else datetime.fromisoformat(str(value)).astimezone()
                    date = dt.date().isoformat()
                except (ValueError, TypeError, OSError, OverflowError):
                    date = ""
            else:
                date = self.get(row)
            if date:
                known[str(row["file_id"])] = date
                eligible.append(row)
        selected = {key: value for key, value in known.items() if in_range(value, start, end)}
        note = f"按{request['phrase']}（{label(start, end)}）的资料回答。"
        fallback = False
        now = datetime.fromisoformat(request["now"])
        # A narrow, explained boundary correction: never silently substitute arbitrary old files.
        if now.weekday() == 0 and request["field"] == "any" and request["phrase"] in ("本周", "这周", "这个星期", "这星期", "本星期", "这个礼拜", "这礼拜"):
            previous = (datetime.fromisoformat(start) - timedelta(days=7)).date().isoformat()
            words = [word.replace("例会", "会议") for word in request["keywords"]]
            subject = request.get("subject", "")
            def related(row):
                name = str(row.get("name", "")).replace("例会", "会议")
                return subject in name if subject else name_match(words, name) > 0
            relevant = {str(row["file_id"]): known[str(row["file_id"])] for row in eligible if related(row)}
            if subject:
                selected = {key: value for key, value in selected.items() if key in relevant}
            previous_docs = {key: value for key, value in relevant.items() if in_range(value, previous, start)}
            if not selected and previous_docs:
                selected = previous_docs
                start, end = previous, start
                fallback = True
                note = (f"今天是 {now.date().isoformat()}，星期一。没有找到本周（{label(request['start'], request['end'])}）日期可确认的资料；"
                        f"你可能把今天记成了周日，想问的是上周。以下按上周（{label(start, end)}）的相关资料回答。")
        missing = f"没有找到{request['phrase']}（{label(request['start'], request['end'])}）日期可确认的资料，无法据此回答。可以补充资料，或指定其他日期。"
        return {**request, "start": start, "end": end, "dates": selected,
                "doc_ids": sorted(selected), "fallback": fallback, "note": note,
                "empty_message": missing}
