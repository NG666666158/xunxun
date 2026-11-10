"""Rule-based parsing of "find a file" questions into metadata filters.

The LLM planner only copies a time phrase ("上周", "上个月"); dates are resolved here,
deterministically, against the local clock. The quick-search window uses
``parse_query`` alone (no LLM call), the chat planner merges it with its own output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
import re

from .doc_cards import detect_doc_type, type_word_at_end

# Label -> extensions. Order matters for display only.
FILE_TYPES: dict[str, tuple[str, ...]] = {
    "pdf": (".pdf",),
    "word": (".docx", ".doc"),
    "excel": (".xlsx", ".xls", ".csv"),
    "ppt": (".pptx", ".ppt"),
    "markdown": (".md", ".markdown"),
    "text": (".txt",),
    "image": (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"),
}

_TYPE_WORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<![A-Za-z])pdf(?![A-Za-z])", re.I), "pdf"),
    (re.compile(r"(?<![A-Za-z])(?:word|docx?)(?![A-Za-z])(?:文档|文件)?", re.I), "word"),
    (re.compile(r"(?<![A-Za-z])(?:excel|xlsx?|csv)(?![A-Za-z])|电子表格|表格", re.I), "excel"),
    (re.compile(r"(?<![A-Za-z])pptx?(?![A-Za-z])|幻灯片|演示文稿", re.I), "ppt"),
    (re.compile(r"(?<![A-Za-z])(?:markdown|md)(?![A-Za-z])", re.I), "markdown"),
    (re.compile(r"(?<![A-Za-z])txt(?![A-Za-z])|纯文本|文本文件", re.I), "text"),
]

# Only counted or listed as files ("多少张图片"): "讲图片压缩的论文" is about something else.
_IMAGE_WORDS = re.compile(r"图片|照片|截图")

_ADDED_WORDS = re.compile(r"(?<![增参附追外施叠])(?:新)?(?:加|放|导入|添加|收|存|丢|拖|上传|入库|新增)(?:进来|进去|进|入)?(?:了)?的")
_MODIFIED_WORDS = re.compile(r"(?:改|修改|更新|编辑)(?:过)?(?:了)?的")

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_NUM = r"(?:\d{1,3}|[一二两三四五六七八九十]{1,3}|几)"


def _number(text: str) -> int:
    if text.isdigit():
        return int(text)
    if text == "几":
        return 3
    if text == "十":
        return 10
    if "十" in text:
        head, _, tail = text.partition("十")
        return (_CN_DIGITS.get(head, 1) if head else 1) * 10 + (_CN_DIGITS.get(tail, 0) if tail else 0)
    return _CN_DIGITS.get(text, 1)


def _day(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


def _month_start(year: int, month: int) -> datetime:
    while month < 1:
        month += 12
        year -= 1
    while month > 12:
        month -= 12
        year += 1
    return datetime(year, month, 1)


def _month(year: int, month: int) -> tuple[datetime, datetime]:
    start = _month_start(year, month)
    return start, _month_start(start.year, start.month + 1)


def _unit_days(unit: str) -> int:
    return {"天": 1, "日": 1, "周": 7, "星期": 7, "礼拜": 7}.get(unit, 0)


# (pattern, resolver(match, now) -> (start, end)). Longest/most specific first.
def _time_rules():
    def today(m, now):
        return _day(now), _day(now) + timedelta(days=1)

    def days_ago(n):
        def resolve(m, now):
            start = _day(now) - timedelta(days=n)
            return start, start + timedelta(days=1)
        return resolve

    def recent(m, now):
        groups = m.groupdict()
        n = _number(groups["n"]) if groups.get("n") else 7
        unit = groups.get("u") or "天"
        end = _day(now) + timedelta(days=1)
        if unit in ("月", "个月"):
            start = _day(now) - timedelta(days=30 * n)
        elif unit == "年":
            start = _day(now) - timedelta(days=365 * n)
        else:
            start = _day(now) - timedelta(days=n * _unit_days(unit))
        return start, end

    def ago(m, now):
        n = _number(m.group("n"))
        unit = m.group("u")
        if unit in ("月", "个月"):
            return _month(now.year, now.month - n)
        if unit == "年":
            return datetime(now.year - n, 1, 1), datetime(now.year - n + 1, 1, 1)
        span = _unit_days(unit)
        if span == 1:
            start = _day(now) - timedelta(days=n)
            return start, start + timedelta(days=1)
        monday = _day(now) - timedelta(days=now.weekday()) - timedelta(days=7 * n)
        return monday, monday + timedelta(days=7)

    def week(offset):
        def resolve(m, now):
            monday = _day(now) - timedelta(days=now.weekday()) + timedelta(days=7 * offset)
            return monday, monday + timedelta(days=7)
        return resolve

    def month(offset):
        def resolve(m, now):
            return _month(now.year, now.month + offset)
        return resolve

    def year(offset):
        def resolve(m, now):
            return datetime(now.year + offset, 1, 1), datetime(now.year + offset + 1, 1, 1)
        return resolve

    def ymd(m, now):
        start = datetime(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        return start, start + timedelta(days=1)

    def y(m, now):
        value = int(m.group("y"))
        return datetime(value, 1, 1), datetime(value + 1, 1, 1)

    def md(m, now):
        mon, d = _number(m.group("m")), int(m.group("d"))
        start = datetime(now.year, mon, d)
        if start > now:
            start = datetime(now.year - 1, mon, d)
        return start, start + timedelta(days=1)

    def mo(m, now):
        mon = _number(m.group("m"))
        if not 1 <= mon <= 12:
            raise ValueError("month out of range")
        start, end = _month(now.year, mon)
        if start > now:
            start, end = _month(now.year - 1, mon)
        return start, end

    unit = r"(?P<u>天|日|周|星期|礼拜|个月|月|年)"
    return [
        (re.compile(r"(?P<y>\d{4})\s*[年\-/.]\s*(?P<m>\d{1,2})\s*[月\-/.]\s*(?P<d>\d{1,2})\s*[日号]?"), ymd),
        (re.compile(r"(?P<y>\d{4})\s*(?:年\s*(?P<m>\d{1,2})\s*月份?|[\-/.](?P<m2>\d{1,2})(?!\d))"),
         lambda m, now: _month(int(m.group("y")), int(m.group("m") or m.group("m2")))),
        (re.compile(r"(?P<y>\d{4})\s*年(?:份|度)?"), y),
        (re.compile(r"(?P<m>\d{1,2}|[一二三四五六七八九十]{1,2})\s*月\s*(?P<d>\d{1,2})\s*[日号]"), md),
        (re.compile(r"(?<![\d上下本这个])(?P<m>1[0-2]|0?[1-9]|十[一二]?|[一二三四五六七八九])\s*月份?(?![\d])"), mo),
        (re.compile(r"(?:最近|近|过去)\s*(?P<n>" + _NUM + r")?\s*(?:个)?" + unit + r"(?:内|以来|里)?"), recent),
        (re.compile(r"(?P<n>" + _NUM + r")\s*(?:个)?" + unit + r"(?:以)?前"), ago),
        (re.compile(r"最近|近期|新近"), recent),
        (re.compile(r"今天|今日"), today),
        (re.compile(r"昨天|昨日"), days_ago(1)),
        (re.compile(r"前天"), days_ago(2)),
        (re.compile(r"本周|这周|这个?星期|这个?礼拜|本星期"), week(0)),
        (re.compile(r"上上周|上上个?星期"), week(-2)),
        (re.compile(r"上一?周|上一?个?星期|上一?个?礼拜"), week(-1)),
        (re.compile(r"本月|这个月|当月"), month(0)),
        (re.compile(r"上上个?月"), month(-2)),
        (re.compile(r"上一?个?月"), month(-1)),
        (re.compile(r"今年|本年度?"), year(0)),
        (re.compile(r"去年|上一年"), year(-1)),
        (re.compile(r"前年"), year(-2)),
    ]


_TIME_RULES = _time_rules()
_TIME_HINT = re.compile(r"[年月周天号日]|星期|礼拜|最近|近期|今|昨|前天")
# 最近 said alone, with no number or unit: "最近的检测报告", not "最近三天" / "近两周".
RECENT_ALONE = re.compile(r"最近|近期|新近")
_NEWEST = re.compile(r"最新")


def resolve_time(phrase: str, now: datetime | None = None) -> tuple[datetime, datetime] | None:
    """Local ``[start, end)`` for a Chinese time phrase, or None when it isn't recognized."""
    now = now or datetime.now()
    for pattern, resolver in _TIME_RULES:
        match = pattern.search(phrase or "")
        if match:
            try:
                return resolver(match, now)
            except ValueError:
                return None
    return None


@dataclass
class FileFilters:
    time_range: tuple[datetime, datetime] | None = None
    time_phrase: str = ""
    time_field: str = "any"  # added | modified | any
    file_types: list[str] = field(default_factory=list)
    folder: str = ""
    name_keywords: list[str] = field(default_factory=list)
    topic: str = ""
    unrecognized: list[str] = field(default_factory=list)
    # The document type asked for ("找一下合同" -> 合同), matched against each file's card
    # (retrieval/doc_cards.py), and the word the question used for it.
    doc_type: str = ""
    doc_type_word: str = ""
    # "我的简历": the user's own (used only with the doc_card_owner switch).
    mine: bool = False
    # "最近的检测报告", "最新的论文": a bare 最近 / 最新 and nothing but the type. With
    # doc_type_search the files of the type are listed newest first, and 最近 is that order
    # rather than the last seven days (FileSearchService.as_searched).
    newest_first: bool = False

    @property
    def type_rest(self) -> str:
        """The topic without the type word: "" when the question names nothing but the type."""
        if not self.doc_type:
            return self.topic.strip()
        rest = self.topic
        if self.doc_type_word:
            at = rest.lower().rfind(self.doc_type_word.lower())
            if at >= 0:
                rest = rest[:at] + " " + rest[at + len(self.doc_type_word):]
        rest = _MINE.sub(" ", rest)
        return " ".join(part.strip(_EDGE_PARTICLES) for part in rest.split() if part.strip(_EDGE_PARTICLES))

    @property
    def extensions(self) -> list[str]:
        return [ext for label in self.file_types for ext in FILE_TYPES.get(label, ())]

    @property
    def has_metadata(self) -> bool:
        return bool(self.time_range or self.file_types or self.folder)

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.time_range:
            start, end = self.time_range
            # Inclusive last day reads better in the UI than an exclusive bound.
            data["time_range"] = [start.date().isoformat(), (end - timedelta(days=1)).date().isoformat()]
        return data


# Wording that asks for files, not for an answer from their content.
_DOC_NOUN = r"(?:文件|文档|资料|材料|书|报告|论文|表格|PDF|pdf|Word|word|Excel|excel|PPT|ppt)"
# What a file is called when the user asks where it is: "会议纪要在哪", "那本书在哪里？".
_DOC_KIND = (
    r"(?:文件(?!夹)|文档|资料|材料|书|手册|报告|论文|表格?|规范|制度|办法|规定|合同|协议|方案|纪要|清单|名单|"
    r"简历|课件|讲义|笔记|说明|模板|范本|通知|附件|稿|档案|照片|图片|截图|幻灯片|演示文稿|"
    r"(?<![A-Za-z])(?i:pdf|word|excel|pptx?|docx?|xlsx?|csv|markdown|md|txt)(?![A-Za-z]))"
)
# "在哪" at the very end of the question ("…在哪里？", "…在哪儿呢", "…在哪能找到").
_WHERE_AT_END = (
    r"(?:存放|放|存)?在哪(?:里|儿)?(?:呢|啊|呀|吗|啦)?"
    r"(?:(?:可以|能)?(?:找到|下载|打开|看到|查看))?\s*[?？!！。.~]*\s*$"
)
# A file asked for by where it is: a kind of document right before "在哪" at the end. Not
# "在哪一天 / 在哪个城市 / 福鼎在哪个市 / 在哪些方面": those ask about the content
# (evaluation/suite_v1/REPORT.md, fix 1).
_WHERE_IS_FILE = _DOC_KIND + r"\s*" + _WHERE_AT_END + r"|在哪(?:里|儿)?(?:可以|能)?(?:找到|下载|打开)[^，,。？?；;]{0,20}" + _DOC_KIND
# "需要提交哪些材料", "要带什么证件": the documents a procedure requires, a question about content.
_REQUIRED_DOCUMENTS = re.compile(
    r"(?:需要|要|须|需|必须|提交|准备|提供|携带|带上?|递交|上交|出具|填写|填)\s*(?:哪些|什么|啥|哪几[份种样个])"
    r"(?:文件|文档|资料|材料|证件|证明|手续|表格?)"
)
_STRONG_CUE = re.compile(
    _WHERE_IS_FILE + r"|"
    # "放在哪" but not "播放哪个 / 发放哪些 / 开放哪": verbs that only end in 放.
    r"(?<![播投发释开解排摆安堆停燃下外绽奔豪怒])放(?:在)?哪|哪(?:份|个文件|篇|本|几份|几个文件)|"
    # "什么材料 / 什么书" alone ("需要什么材料", "看什么书") ask about content: weak cues only.
    r"哪些(?:文件|文档|资料|材料|书|报告|论文|表格)|什么(?:文件|文档|资料|报告|论文|表格)|"
    r"有(?:哪些|什么|没有|无)[^，,。？?；;]{0,20}(?:" + _DOC_NOUN[3:-1] + r"|规范|制度|手册|办法)|"
    r"(?:找|搜|查找|搜索|打开)(?:一下|找|出|到)?[^，,。？?；;]{0,20}" + _DOC_NOUN + r"|"
    # "讲多 Agent 协作的书" / "介绍 DeepSeek-R1 的资料": a document described by its topic.
    r"(?:讲|介绍|关于|有关|讨论|涉及)[^，,。？?；;]{1,24}的(?:那)?(?:本|份|篇|个)?" + _DOC_NOUN + r"|"
    # "华东区讲报销的那份 Word 文档" / "规定 RD-05-ZX11 的那份文件"
    r"那(?:份|本|篇|个|几份)[^，,。？?；;]{0,12}" + _DOC_NOUN + r"|"
    r"帮我找|找一下"
)
# Counting or listing the library itself ("一共有多少个 PDF", "列出所有 Excel"), whatever else
# the question names. Counting something else ("PDF 有多少页", "有几个步骤") does not match.
_FILE_NOUN = (
    r"(?:文件(?!夹)|文档|资料|材料|表格|幻灯片|演示文稿|纯文本|图片|照片|截图|"
    r"(?<![A-Za-z])(?:pdf|word|excel|pptx?|docx?|xlsx?|csv|markdown|md|txt)(?![A-Za-z]))"
)
_HOW_MANY = r"(?:多少|几)(?:个|份|篇|张|本|件)?"
_TOTAL_WORDS = r"(?:现在|目前|当前|一共|总共|总计|合计|共|都|还)*(?:有)?"
_INVENTORY = re.compile(
    # "多少个 PDF", "几份 Word 文档", "多少个讲 KV Cache 的 PDF"
    r"(?:多少|几)(?:个|份|篇|张|本|件)?\s*(?:[^，,。？?；;的]{0,24}的\s*)?" + _FILE_NOUN + r"|"
    # "PDF 有多少个", "文件总共几份", "文档有多少"
    + _FILE_NOUN + r"\s*" + _TOTAL_WORDS + r"\s*" + _HOW_MANY + r"(?=\s*(?:[?？。!！,，]|$))|"
    # "文件总数", "文档的数量"
    + _FILE_NOUN + r"\s*的?(?:总数|数量|个数)|"
    # "列出所有 PDF", "把全部 Word 列出来", "所有 Excel 文件"
    r"(?:所有|全部)的?\s*" + _FILE_NOUN + r"|"
    r"(?:列出|列一下|列举|罗列)[^，,。？?；;]{0,12}" + _FILE_NOUN,
    re.I,
)
# Points at one thing already known: "这个表格有多少个" counts inside it, not the library.
_POINTING = re.compile(r"这份|这个(?![月星礼])|这些|这篇|这本|那份|那个|那些|那篇|那本|该|此")
_COUNT_WORDS = re.compile(
    r"(?:(?:我的|我们的|这个|当前|整个)?(?:知识库|资料库|文件库))?\s*" + _TOTAL_WORDS + r"\s*" + _HOW_MANY
    + r"|(?:我的|我们的|这个|当前|整个)?(?:知识库|资料库|文件库)|总数|数量|个数|"
    r"现在|目前|当前|一共|总共|总计|合计|共有|都有|所有|全部|列出来|列出|列一下|列举|罗列|统计|清点|是多少|"
    r"(?<![A-Za-z])(?:给我)?看看"
)
_COUNT_AHEAD = r"(?=\s*(?:一共|总共|共)?(?:有)?\s*(?:多少|几))"
_ADDED_COUNT = re.compile(
    r"(?<![增参附追外施叠])(?:新)?(?:加|放|导入|添加|收|存|丢|拖|上传|入库|新增)(?:进来|进去|进|入)?(?:了)?" + _COUNT_AHEAD
)
_MODIFIED_COUNT = re.compile(r"(?:改|修改|更新|编辑)(?:过)?(?:了)?" + _COUNT_AHEAD)
# "知识库里…": where the user is asking, never what about.
_LIBRARY_PLACE = re.compile(r"(?:我的|我们的|这个|当前|整个)?(?:知识库|资料库|文件库)(?:里面|里头|里边|里|中|内)")

# A question about content rather than a request for a file.
QUESTION_WORDS = re.compile(r"什么|怎么|怎样|如何|为什么|为何|是否|吗|呢|多少|几个|区别|原理|步骤|流程|作用|意思|[?？]")
_WEAK_CUE = re.compile(r"文件|文档|资料|材料|(?<![A-Za-z])(?:pdf|word|excel|ppt|docx|xlsx|md|markdown|txt)(?![A-Za-z])", re.I)
# Refers to something earlier in the conversation; the planner needs the history.
HISTORY_REFERENCE = re.compile(r"它|刚才|上面|前面那|上一个(?!月|星期|礼拜)|上一份|这份|该文件|这个文件|第[一二三四五六七八九十\d]+个")


def is_inventory(question: str) -> bool:
    """Is this about how many files the library holds, or which ones, rather than their content?"""
    return bool(_INVENTORY.search(question)) and not (HISTORY_REFERENCE.search(question) or _POINTING.search(question))


_WEAK_WHERE = re.compile(_WHERE_AT_END)


# ---------------- a document asked for by its type ----------------
# "有哪些合同", "我的简历呢", "把会议纪要列一下", "最近的检测报告": no 找 / 在哪 / 那份, but the
# question ends in a document type (doc_cards.DOC_TYPES: the head of the noun phrase) and asks
# for documents of it: which there are, the user's own, the latest. A content question that
# names a type does not end in it ("合同里有哪些违约条款", "检测报告的结论是什么", "申报书要写
# 哪些内容") or asks about it ("什么是合同", "如何写会议纪要", "需要签合同吗").
#
# Type words that are as often what a question is about ("有哪些标准？" after a question on
# evaluation, "最近有什么新闻", "达成了哪些协议") are left to the other cues.
TYPE_WORDS_NOT_A_REQUEST = frozenset({
    "标准", "规定", "办法", "证明", "指南", "方案", "协议", "汇报", "述职", "答辩", "路演", "新闻", "报道", "快讯",
})
_REQUEST_END = re.compile(r"[\s呢吗吧啊呀哦嘛了～~?？。.!！,，、]+$")
_ASK_LIST = r"(?:都|还|一共|总共|共)?有(?:哪些|什么|几[份个篇张本]|多少[份个篇张本]?)"
_LIST_VERB = r"列(?:一下|出来|出|个清单)|列举(?:一下)?|罗列(?:一下)?"
# After the type: "合同有哪些", "把会议纪要列一下", "简历给我看看".
_LIST_AFTER = re.compile(r"(?:" + _ASK_LIST + r"|(?:帮我|给我)?(?:" + _LIST_VERB + r")|(?:给我)?看(?:看|一下))$")
# Before it: "有哪些合同", "列出所有简历", "看看我的简历".
_LIST_BEFORE = re.compile(_ASK_LIST + r"|" + _LIST_VERB + r"|(?:给我)?看(?:看|一下)|查看|打开|(?:所有|全部)(?:的)?")
_HAVE = re.compile(r"有没有|有无|是否有|^\s*(?:还|都)?有\s*$")
_OWN = re.compile(r"我(?:自己|本人)?的|我们(?:公司)?的|咱们的|本人的?|自己的")
_LATEST = re.compile(r"最新|最早|上次|之前|以前|新的|旧的")
_POLITE = re.compile(r"请问|帮我|帮忙|请你?|麻烦你?|给我|我想要?|我要|想要?|^\s*把|(?<=[我忙你请烦])把")
_FORMAT = re.compile(r"版本?|格式|扫描件|电子版|原件")
# Type words that follow the type ("合同扫描件", "简历文件").
_TYPE_TAIL = re.compile(r"(?:文件|文档|资料|原件|扫描件|复印件|电子版)$")
# What a question about the document, not a request for it, puts before the type word.
_ASKS_ABOUT = re.compile(r"什么|啥|哪|怎|如何|为何|为什么|多少|几|谁|是否|是不是|能否|可否|吗|[，,。？?；;：:]")
# "签合同", "写论文", "交报告": doing something with a document, not asking for one.
_VERB_BEFORE = re.compile(r"[签写交填开审读拟拍做改发收寄送订看找要带备]\s*$")
# A clause rather than a noun before the type ("所有员工都要签劳动合同"); a described type is
# asked for as "…的合同" or as a short compound ("劳动合同", "采购合同").
_CLAUSE_WORD = re.compile(r"[签写交填审读拟拍做改寄送订看找带都也还要是在没不会能该让]")


def type_request(question: str) -> str | None:
    """"strong" for "有哪些合同 / 我的简历呢 / 把会议纪要列一下 / 最近的检测报告"; "weak" when the
    question ends in a type word and asks for it less plainly ("有没有签合同", "简历呢"); None
    when it does not ask for documents of a type."""
    text = _REQUEST_END.sub("", _LIBRARY_PLACE.sub(" ", question or "").strip())
    ends_in_ne = bool(re.search(r"呢[\s?？。.!！~～]*$", question or ""))
    yes_no = bool(re.search(r"吗[\s?？。.!！~～]*$", question or ""))
    markers: set[str] = set()
    after = _LIST_AFTER.search(text)
    if after:
        markers.add("list_after")
        text = _REQUEST_END.sub("", text[: after.start()])
    text = text.rstrip()
    name, word = type_word_at_end(text)
    if not name:
        tail = _TYPE_TAIL.search(text)
        if tail:
            text = text[: tail.start()].rstrip()
            name, word = type_word_at_end(text)
    if not name or word in TYPE_WORDS_NOT_A_REQUEST:
        return None
    prefix = text[: len(text) - len(word)]
    if _POINTING.search(prefix) or HISTORY_REFERENCE.search(prefix):
        return None  # "这些合同有哪些": about documents already named
    verb_before = bool(_VERB_BEFORE.search(prefix))
    rest = _POLITE.sub(" ", prefix)
    for pattern, kind in ((_HAVE, "have"), (_LIST_BEFORE, "list"), (_OWN, "own"), (_ADDED_WORDS, "time"),
                          (_MODIFIED_WORDS, "time"), (_LATEST, "time")):
        if pattern.search(rest):
            markers.add(kind)
            rest = pattern.sub(" ", rest)
    for pattern, _ in _TIME_RULES:
        match = pattern.search(rest)
        if match:
            markers.add("time")
            rest = rest[: match.start()] + " " + rest[match.end() :]
            break
    for pattern, _ in _TYPE_WORDS:
        rest = pattern.sub(" ", rest)
    rest = _FORMAT.sub(" ", rest)
    descriptor = re.sub(r"\s+", "", rest).strip("的了都也还就")
    if _ASKS_ABOUT.search(descriptor) or len(descriptor) > 20:
        return None
    if not markers:
        # "简历呢": on its own a request, after an answer maybe "and what about contracts?".
        return "weak" if ends_in_ne else None
    if not descriptor:
        return "strong"
    if yes_no:
        return None  # "所有员工都要签劳动合同吗": asks whether, not for documents
    # "有哪些和华为签的合同", "列出所有劳动合同", "我的采购合同": described, still asked for.
    described = prefix.rstrip().endswith("的") or (len(descriptor) <= 4 and not _CLAUSE_WORD.search(descriptor))
    if markers & {"list", "own"} and described and not verb_before:
        return "strong"
    return "weak"


def find_file_cue(question: str) -> str | None:
    """"strong": certainly finding files; "weak": maybe (the planner decides); None: content."""
    question = _REQUIRED_DOCUMENTS.sub("，", question)
    if _STRONG_CUE.search(question):
        return "strong"
    # "一共有多少个 PDF": nothing but conditions. "报销要提交几份材料" is about reimbursement.
    if is_inventory(question) and not parse_query(question).topic:
        return "strong"
    # "上周加的报价单": when a file was added only matters for finding files.
    if _ADDED_WORDS.search(question) and resolve_time(question):
        return "strong"
    by_type = type_request(question)
    if by_type == "strong":
        return "strong"
    # "面试题200问在哪", "公司总部在哪": a file by its name, or a place in the content.
    if by_type or _WEAK_CUE.search(question) or _WEAK_WHERE.search(question):
        return "weak"
    return None


_FOLDER_EXPLICIT = re.compile(r"(?:文件夹|目录|folder|in)\s*[:：]\s*(?P<f>\S+)", re.I)
_WIN_PATH = re.compile(r"[A-Za-z]:\\[^\s，,。？?]*")
_FOLDER_WORD = re.compile(r"(?:文件夹|目录)(?:里面|里头|里边|下面|里|中|下|内)?")
_FOLDER_STOP = set("在进到从的里中和与把将放加了")

_FILLERS = re.compile(
    r"帮我|请|麻烦|给我|我想|我要|想要|"
    r"查找|搜索|搜一下|找一下|找找|找到|找出|找|搜|"
    r"有没有|是否有|有哪些|有什么|有关于|有几[份个篇张本]|有多少[份个篇张本]?|^\s*(?:还|都)?有(?=[^，,。？?]{1,12}吗)|"
    # "把会议纪要列一下", "看看我的简历", "所有合同", "最新的检测报告" (type_request).
    r"^\s*把|(?<=[我忙你请烦])把|列(?:一下|出来|出|个清单)|列举|罗列|看看|看一下|查看|所有|全部|最新|"
    r"那几份|那几个|那份|这份|那个|这个|那篇|那本|哪份|哪个|哪篇|哪本|哪些|什么|"
    r"在哪里|在哪儿|在哪|放在哪|放哪|位置|路径|打开|一下|"
    r"关于|有关|相关的?|讲的是|讲到|讲了|讲解|讲|提到|介绍|涉及|主题是|内容是|写的|"
    r"之前|上次|以前|曾经|"
    # "书" alone is filler ("那本书"), but not the end of 申报书 / 说明书 / 证书 (document types).
    r"文件夹|目录|文件|文档|资料|材料|东西|书籍|(?<![报请议明划决定诉解务证子])书|格式|版本|"
    r"[?？!！。，,、：:；;（）()\"“”'‘’《》]"
)
_EDGE_PARTICLES = "的了吗呢吧啊是我那些个份篇本里中和与及或也都 "
_MINE = re.compile(r"我(?:自己|本人)?的|本人的?")


def parse_query(text: str, now: datetime | None = None) -> FileFilters:
    """Extract time, type, folder and the remaining topic/name words from a question."""
    now = now or datetime.now()
    filters = FileFilters()
    rest = _LIBRARY_PLACE.sub(" ", text or "")
    inventory = is_inventory(rest)
    filters.mine = bool(_MINE.search(rest))
    newest = bool(_NEWEST.search(rest))  # before "新加的" takes the 新 of "最新加的"

    for match in _ADDED_WORDS.finditer(rest):
        filters.time_field = "added"
    if filters.time_field == "any" and _MODIFIED_WORDS.search(rest):
        filters.time_field = "modified"
    rest = _MODIFIED_WORDS.sub(" ", _ADDED_WORDS.sub(" ", rest))
    if inventory:
        # "上周加了几个文件": said without 的, but still about when files were added.
        if filters.time_field == "any":
            if _ADDED_COUNT.search(rest):
                filters.time_field = "added"
            elif _MODIFIED_COUNT.search(rest):
                filters.time_field = "modified"
        rest = _MODIFIED_COUNT.sub(" ", _ADDED_COUNT.sub(" ", rest))

    for pattern, resolver in _TIME_RULES:
        match = pattern.search(rest)
        if not match:
            continue
        try:
            filters.time_range = resolver(match, now)
            filters.time_phrase = match.group(0).strip()
        except ValueError:
            filters.unrecognized.append(match.group(0))
        rest = rest[: match.start()] + " " + rest[match.end() :]
        break

    for pattern, label in _TYPE_WORDS:
        if pattern.search(rest):
            if label not in filters.file_types:
                filters.file_types.append(label)
            rest = pattern.sub(" ", rest)
    if inventory and _IMAGE_WORDS.search(rest):
        filters.file_types.append("image")
        rest = _IMAGE_WORDS.sub(" ", rest)

    match = _FOLDER_EXPLICIT.search(rest) or _WIN_PATH.search(rest)
    if match:
        filters.folder = match.group("f") if "f" in match.re.groupindex else match.group(0)
        rest = rest[: match.start()] + " " + rest[match.end() :]
    else:
        word = _FOLDER_WORD.search(rest)
        if word:
            start = word.start()
            while start > 0 and rest[start - 1] not in _FOLDER_STOP and re.match(r"[\w\-.一-鿿]", rest[start - 1]):
                start -= 1
            name = rest[start : word.start()].strip()
            if name:
                filters.folder = name
                rest = rest[:start] + " " + rest[word.end() :]

    if inventory:
        rest = _COUNT_WORDS.sub(" ", rest)
    rest = _FILLERS.sub(" ", rest)
    segments: list[str] = []
    for chunk in re.split(r"的", rest):
        latin: list[str] = []
        for part in chunk.split():
            part = part.strip(_EDGE_PARTICLES)
            if not part:
                continue
            if re.fullmatch(r"[!-~]+", part):
                latin.append(part)  # "KV Cache" stays one keyword
                continue
            if latin:
                segments.append(" ".join(latin))
                latin = []
            if len(part) >= 2:
                segments.append(part)
        if latin:
            segments.append(" ".join(latin))
    filters.name_keywords = segments
    filters.topic = " ".join(segments)
    if not inventory:
        filters.doc_type, filters.doc_type_word = detect_doc_type(segments)
        latest = newest or bool(RECENT_ALONE.fullmatch(filters.time_phrase))
        filters.newest_first = latest and bool(filters.doc_type) and not filters.type_rest
    return filters


def merge_planner_filter(rules: FileFilters, planned: dict | None, now: datetime | None = None) -> FileFilters:
    """Rules win for what they recognized (dates, types); the planner fills in the rest."""
    if not planned:
        return rules
    merged = FileFilters(**{**asdict(rules), "time_range": rules.time_range})
    phrase = str(planned.get("time_phrase") or "").strip()
    if merged.time_range is None and phrase:
        resolved = resolve_time(phrase, now)
        if resolved:
            merged.time_range, merged.time_phrase = resolved, phrase
        elif _TIME_HINT.search(phrase):
            merged.unrecognized.append(phrase)
    if merged.time_field == "any" and planned.get("time_field") in ("added", "modified"):
        merged.time_field = planned["time_field"]
    for label in planned.get("file_types") or []:
        label = str(label).lower()
        label = {"docx": "word", "doc": "word", "xlsx": "excel", "xls": "excel", "md": "markdown", "txt": "text",
                 "pptx": "ppt"}.get(label, label)
        if label in FILE_TYPES and label not in merged.file_types:
            merged.file_types.append(label)
    if not merged.folder and planned.get("folder"):
        merged.folder = str(planned["folder"]).strip()
    topic = str(planned.get("topic") or "").strip()
    if topic:
        merged.topic = topic
        if not merged.doc_type:
            merged.doc_type, merged.doc_type_word = detect_doc_type([topic])
    for word in planned.get("name_keywords") or []:
        word = str(word).strip()
        if word and word not in merged.name_keywords:
            merged.name_keywords.append(word)
    return merged


def compact(text: str) -> str:
    return re.sub(r"[^0-9a-z一-鿿]", "", (text or "").lower())


def name_match(keywords: list[str], name: str) -> float:
    """How well the keywords match a file name, in [0, 1] (bigram overlap, length weighted)."""
    stem = compact(name.rsplit(".", 1)[0])
    if not stem:
        return 0.0
    total = matched = 0.0
    for keyword in keywords:
        key = compact(keyword)
        if len(key) < 2:
            continue
        if key in stem:
            overlap = 1.0
        else:
            grams = {key[i : i + 2] for i in range(len(key) - 1)}
            overlap = sum(gram in stem for gram in grams) / len(grams)
        total += len(key)
        matched += len(key) * overlap
    score = matched / total if total else 0.0
    return score if score >= 0.4 else 0.0
