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
# Names that begin with a time word but name a product or programme ("今日头条怎么打不开").
NOT_TIME = re.compile(r"今日(?:头条|说法|关注|资本|俄罗斯|美国|影评|水印)")


def _relative(question: str) -> list[re.Match]:
    """Time words of the question, leaving out those that only start a name."""
    names = [m.span() for m in NOT_TIME.finditer(question)]
    return [m for m in RELATIVE.finditer(question) if not any(a <= m.start() < b for a, b in names)]


# "创下了怎样的记录": a record set, not a document kept.
ACHIEVEMENT = re.compile(r"(?:创|破|刷新)[^，。？！,.?!]{0,6}$")


def _subject(question: str, pos: int = 0) -> re.Match | None:
    """The first word naming a document, from ``pos`` on."""
    for match in DOCUMENT_SUBJECT.finditer(question, pos):
        if match[0] == "记录" and ACHIEVEMENT.search(question[:match.start()]):
            continue
        return match
    return None


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
    relatives = _relative(question)
    relative = relatives[0] if relatives else None
    phrases = {m[0] for m in relatives}
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
    subject = _subject(question, anchor.end())
    dates_document = subject and (action is None or subject.start() < action.start())
    document_time = parsed.time_field != "any" or find_file_cue(question) == "strong" or dates_document or (not action and (
        parsed.doc_type or parsed.file_types or _subject(question)))
    return {"phrase": parsed.time_phrase, "start": start.date().isoformat(),
            "end": end.date().isoformat(), "field": parsed.time_field,
            "basis": "document" if document_time else "content",
            "now": now.isoformat(), "keywords": parsed.name_keywords,
            "subject": subject[0].replace("例会", "会议") if subject else "",
            # An explicit date may be the day an event in the text happened, not a document's date.
            # "2023年7月……环比上月": the range comes from the explicit date, "上月" is relative to it.
            "explicit": relative is None or not any(m[0] in parsed.time_phrase for m in relatives),
            # Payments, revenue, deliveries: dated by when they happened, not by their certificate.
            "action": bool(action)}


def answer_rules(request: dict) -> str:
    """What the answer prompt adds for a time-scoped question."""
    rules = (f"\n问题要求的时间范围：{label(request['start'], request['end'])}。"
             f"{request.get('note', '')}使用实际日期，不把上周称作本周。")
    if request.get("basis") != "content":
        return rules + "只回答上述限定时间内的资料。"
    if not request.get("action"):
        return rules + "按资料正文记载的实际日期回答；所问时间范围内没有记载时，明确说所问时间没有依据，并说明找到的实际日期。"
    # The wording 1.0.1 shipped and measured for payment questions, unchanged.
    rules += ("文档开具/生成日期、税款所属期和业务实际发生日期不可混用；交税问题按缴纳/入库日期核对。"
              "如资料只记载其他年份，明确说所问年份没有依据，并说明找到的实际年份和金额。"
              "一份凭证的合计仅是该凭证覆盖的金额，不能当成全年总额；合计与明细不能重复累加。")
    return rules


def check_rules(request: dict, question: str = "") -> str:
    """What the fact check adds when the answer must hold dates found in the text."""
    if request.get("basis") != "content":
        return ""
    rules = f"\n原问题：{question}。其中{request['phrase']}指{label(request['start'], request['end'])}。"
    if not request.get("action"):
        return rules + "核对资料记载的实际日期。"
    # The wording 1.0.1 shipped and measured for payment questions, unchanged.
    return rules + "核对业务实际发生日期，开具日期不等于业务日期。不能把其他年份金额当成本年金额、单份凭证合计当成年总额。"


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

    @staticmethod
    def _usable(row: dict, allowed: set | None, include_named: bool) -> bool:
        """Files a time-scoped turn may use: in scope, not deleted; files kept by name only
        only when the turn lists files (their name and dates may be read, never their text)."""
        if allowed is not None and row["file_id"] not in allowed:
            return False
        if row.get("status") in ("recycled", "deleted"):
            return False
        return include_named or not (row.get("content_mode") == "name" or row.get("status") == "named")

    def _by_content(self, request: dict, rows: list[dict], allowed: set | None, include_named: bool) -> dict:
        # A certificate issued this year may describe last year's payments, and a news report
        # its event's day without carrying that date itself. Keep the documents available;
        # the answer and the check inspect the dates in their text.
        start, end = request["start"], request["end"]
        ids = sorted(str(row["file_id"]) for row in rows if self._usable(row, allowed, include_named))
        return {**request, "basis": "content", "dates": {}, "doc_ids": ids, "fallback": False,
                "note": f"问题中的{request['phrase']}指 {label(start, end)}，以下按资料记载的实际发生日期核对。",
                "empty_message": "当前范围内没有可用于回答的资料。"}

    def select(self, request: dict, rows: list[dict], scope: list[str], include_named: bool = False) -> dict:
        start, end = request["start"], request["end"]
        known = {}
        eligible = []
        allowed = set(scope) if scope else None
        if request.get("basis") == "content":
            return self._by_content(request, rows, allowed, include_named)
        for row in rows:
            if not self._usable(row, allowed, include_named):
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
        if not selected and request.get("explicit"):
            # "2023年7月26日的闭门会议": no document is dated that day, but one may tell of it.
            return self._by_content(request, rows, allowed, include_named)
        missing = f"没有找到{request['phrase']}（{label(request['start'], request['end'])}）日期可确认的资料，无法据此回答。可以补充资料，或指定其他日期。"
        return {**request, "start": start, "end": end, "dates": selected,
                "doc_ids": sorted(selected), "fallback": fallback, "note": note,
                "empty_message": missing}
