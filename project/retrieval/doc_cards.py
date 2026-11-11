"""Document cards: what kind of document a file is ("找一下我的简历").

A resume rarely says 简历, a contract may only say 甲方 / 乙方: the type word a user asks
for is often in neither the file name nor its text, so neither the search nor the reranker
can find it (evaluation/doc_cards_v1/REPORT.md). A card is written once per file, by one
call to the chat model on the file name and the opening of its text:

    {"类型": "简历", "对象": "李明远", "时间": "2025", "摘要": "…"}

The model's type is normalised onto a small controlled vocabulary (``DOC_TYPES``) so that
个人简历 / 履历 / CV all become 简历, and a question's type word is matched against it as a
field (``FileSearchService``), not through the reranker.

Cards are built in the background (``DocCardBuilder``): paced by tokens, waiting while a
question is answered, never for files kept by name only nor for those whose own name gives the
type. After a big import the files the user looks at come first (listed by a search, opened,
likely matches of a type question), then the newest (evaluation/doc_cards_after_import). A card
is rebuilt only when its input (name, opening, prompt version) changes, so the store is its own
cache.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

# Bump when the prompt or the opening changes: every card is written again.
CARD_VERSION = 3
CARD_CHARS = 900


@dataclass(frozen=True)
class DocType:
    name: str
    # Words naming this type, in questions and in the model's labels. Longest wins when two
    # end at the same place ("申报书" over "书").
    words: tuple[str, ...]
    # A question for the parent ("报告") also finds this type ("检测报告").
    parent: str = ""
    # What the prompt says about it (only where the name alone is ambiguous).
    hint: str = ""


DOC_TYPES: tuple[DocType, ...] = (
    DocType("简历", ("简历", "履历", "个人简历", "个人简介", "求职简历", "简历表", "履历表", "CV", "resume"),
            hint="某一个人的履历：姓名、联系方式或出生年份、教育、工作或项目经历"),
    # Without it a company profile ("公司简介.pdf") had only 简历 or 说明书 to choose from.
    DocType("公司简介", ("公司简介", "企业简介", "机构简介", "单位简介", "公司介绍", "企业介绍", "宣传册", "宣传材料", "画册"),
            hint="介绍一家公司或机构的材料，不是简历"),
    DocType("合同", ("合同", "协议", "协议书", "合约", "契约", "contract", "agreement")),
    DocType("发票", ("发票", "票据", "收据", "收款凭证", "invoice", "receipt")),
    DocType("申报书", ("申报书", "申请书", "申报表", "申请表", "申报材料", "项目申报", "任务书", "proposal"),
            hint="申请项目、资金、奖项而写的材料；征集申报的官方公告不是申报书"),
    DocType("通知公告", ("通知", "公告", "通告", "公示", "启事", "通报", "notice"),
            hint="单位发布的通知、公告、公示"),
    DocType("回执", ("回执", "回执单")),
    DocType("会议纪要", ("纪要", "会议纪要", "会议记录", "minutes"),
            hint="记录某次会议的时间、参会人、议题和结论"),
    DocType("报告", ("报告", "report")),
    DocType("检测报告", ("检测报告", "检验报告", "测试报告", "化验单", "化验报告", "体检报告"), parent="报告"),
    DocType("工作报告", ("工作报告", "工作总结", "总结报告", "年度报告", "年报", "述职报告"), parent="报告"),
    DocType("研究报告", ("研究报告", "调研报告", "白皮书", "行业报告"), parent="报告"),
    DocType("论文", ("论文", "学位论文", "毕业论文", "期刊论文", "学术论文", "paper", "thesis"),
            hint="学术论文，也包括只有标题、摘要和关键词的论文页"),
    DocType("说明书", ("说明书", "手册", "使用说明", "操作指南", "用户指南", "指南", "manual")),
    DocType("汇报材料", ("汇报", "汇报材料", "汇报PPT", "述职", "答辩", "路演")),
    DocType("证书证件", ("证书", "证件", "证明", "执照", "身份证", "毕业证", "学位证", "资格证", "营业执照", "护照")),
    DocType("照片", ("照片", "相片", "合影", "合照", "证件照", "photo")),
    DocType("新闻", ("新闻", "报道", "新闻稿", "快讯", "news"),
            hint="媒体发布的新闻报道"),
    DocType("书籍", ("书籍", "电子书", "图书", "教材", "专著", "book")),
    DocType("制度规范", ("制度", "规章", "规定", "办法", "规范", "细则", "条例", "守则", "标准", "章程")),
    DocType("方案", ("方案", "计划书", "策划书", "策划案", "实施方案", "工作计划")),
    DocType("法律文书", ("判决书", "裁定书", "裁判文书", "起诉书", "起诉状", "调解书", "法律文书", "律师函", "答辩状")),
    DocType("邀请函", ("邀请函", "邀请信", "请柬")),
    DocType("课件讲义", ("课件", "讲义", "教案")),
)
TYPES_BY_NAME = {item.name: item for item in DOC_TYPES}

_ASCII_WORD = re.compile(r"[A-Za-z]+")


def _word_pattern(word: str) -> str:
    escaped = re.escape(word)
    if _ASCII_WORD.fullmatch(word):
        return rf"(?<![A-Za-z]){escaped}(?![A-Za-z])"
    return escaped


_PATTERNS = [
    (re.compile(_word_pattern(word), re.I), name)
    for word, name in sorted({(word, item.name) for item in DOC_TYPES for word in item.words}, key=lambda p: -len(p[0]))
]


def _matches(text: str) -> list[tuple[int, int, str, str]]:
    """Every vocabulary word in ``text`` (overlapping ones too): (start, end, word, type)."""
    text = text or ""
    return [(m.start(), m.end(), m.group(0), name) for pattern, name in _PATTERNS for m in pattern.finditer(text)]


def _head(text: str) -> tuple[str, str] | None:
    """The type named last in ``text`` (Chinese noun phrases end in their head: 合同审查制度
    is a 制度), the longest word on a tie. Returns (type, word) or None."""
    found = _matches(text)
    if not found:
        return None
    start, end, word, name = max(found, key=lambda item: (item[1], item[1] - item[0]))
    return name, word


def normalize_type(label: str) -> str:
    """The vocabulary type of a model's label ("个人简历" -> "简历"), "" when none fits."""
    label = re.sub(r"\s+", "", str(label or ""))
    if not label or label.startswith("其他") or label.startswith("其它"):
        return ""
    if label in TYPES_BY_NAME:
        return label
    head = _head(label)
    return head[0] if head else ""


def type_matches(query_type: str, card_type: str) -> bool:
    """A question for 报告 finds 检测报告 too; one for 检测报告 finds only those."""
    if not query_type or not card_type:
        return False
    if query_type == card_type:
        return True
    item = TYPES_BY_NAME.get(card_type)
    return bool(item and item.parent == query_type)


def matching_types(query_type: str) -> list[str]:
    return [item.name for item in DOC_TYPES if type_matches(query_type, item.name)]


# A file name whose stem ends in a type word says what it is ("01 申报书.pdf", "北方区运行维护
# 手册.md", "简历2024(1).pdf"). Matched on the reversed stem, anchored at its start, so a check
# costs one short match per name (10,000 names in a few milliseconds once cached).
def _reversed_word(word: str) -> str:
    escaped = re.escape(word[::-1])
    return rf"{escaped}(?![A-Za-z])" if _ASCII_WORD.fullmatch(word) else escaped


_NAME_HEAD_REVERSED = re.compile(
    "(?:" + "|".join(_reversed_word(word) for word in sorted(
        {word for item in DOC_TYPES for word in item.words}, key=len, reverse=True)) + ")",
    re.I,
)
_TYPE_OF_WORD = {word.lower(): item.name for item in DOC_TYPES for word in item.words}
# Version, date, copy and draft marks after the type word in a file name (reversed).
_NAME_NOISE_REVERSED = re.compile(
    "(?:\\d+[vV]|[\\s_\\-—.·、,，()（）\\[\\]【】\\d]|" + "|".join(
        re.escape(word[::-1]) for word in sorted(
            ("final", "copy", "副本", "最终版", "最终", "终版", "定稿", "修改稿", "修改", "修订版", "修订",
             "草稿", "初稿", "新版", "旧版", "扫描件", "电子版"), key=len, reverse=True)) + ")+",
    re.I,
)


def type_word_at_end(text: str) -> tuple[str, str]:
    """The type whose word ends ``text`` ("有哪些会议纪要" -> 会议纪要, the longest word wins),
    as (type, the word as written), or ("", "")."""
    match = _NAME_HEAD_REVERSED.match(str(text or "")[::-1])
    if not match:
        return "", ""
    word = match.group(0)[::-1]
    return _TYPE_OF_WORD.get(word.lower(), ""), word


@lru_cache(maxsize=65536)
def name_doc_type(name: str) -> str:
    """The type a file's own name gives it ("" when its name ends in no type word)."""
    reversed_stem = Path(str(name or "")).stem[::-1]
    noise = _NAME_NOISE_REVERSED.match(reversed_stem)
    if noise:
        reversed_stem = reversed_stem[noise.end():]
    match = _NAME_HEAD_REVERSED.match(reversed_stem)
    return _TYPE_OF_WORD.get(match.group(0)[::-1].lower(), "") if match else ""


# Trailing words that may follow the type word in a question segment ("简历文件", "合同扫描件").
_TAIL = re.compile(r"(?:文件|文档|资料|材料|原件|扫描件|电子版|那份|这份|吧|呢|啊)+$")


def detect_doc_type(segments: Iterable[str]) -> tuple[str, str]:
    """The document type a find-file question asks for, from its topic segments.

    Only a type word ending the last segment counts (the head of the noun phrase):
    "张三的简历" asks for a 简历; "简历模板" and "简历的模板" for a template. Returns
    (type, the word used) or ("", "")."""
    parts = [str(part) for part in segments if str(part).strip()]
    if not parts:
        return "", ""
    text = _TAIL.sub("", re.sub(r"\s+", "", parts[-1]))
    for start, end, word, name in sorted(_matches(text), key=lambda item: (item[1], item[1] - item[0]), reverse=True):
        if end == len(text):
            return name, word
    return "", ""


def type_probe(query_type: str) -> str:
    """What a file of this type looks like, to search the files still without a card for
    likely ones ("简历 履历 个人简历 …：某一个人的履历：姓名、…")."""
    item = TYPES_BY_NAME.get(query_type)
    if item is None:
        return query_type
    words = " ".join(word for word in item.words[:6] if word != item.name)
    return f"{item.name} {words}" + (f"：{item.hint}" if item.hint else "")


def type_words(query_type: str) -> list[str]:
    """Words a file "only mentions" when it is not of the type asked for."""
    words: list[str] = []
    for name in matching_types(query_type):
        for word in TYPES_BY_NAME[name].words:
            if word not in words:
                words.append(word)
    return words


# ---------------- the card itself ----------------


def _choices() -> str:
    lines = []
    for item in DOC_TYPES:
        lines.append(f"{item.name}（{item.hint}）" if item.hint else item.name)
    return "、".join(lines)


_CARD_RULES = (
    "- 类型：从下列中选最贴切的一个，只写名称：" + _choices() + "。都不贴切时写“其他：<2到6个字的类型>”。\n"
    "- 按文件本身是什么来判断，不按它提到了什么：申报书里有人员简介仍是申报书；征集申报的公告是通知公告；"
    "多人合影不是证书证件；记者写的报道（讲某篇论文、某份合同、某个通知或某个人的事迹）是新闻；"
    "只有标题、摘要和关键词的学术文章是论文。\n"
    "- 对象：文件主要关于哪个人或单位（简历写本人姓名，合同写签约双方，没有就留空）。\n"
    "- 时间：文件自身的日期或年份（没有就留空）。\n"
    "- 摘要：一句话说明内容，不超过40字。\n"
)

CARD_PROMPT = (
    "你是文件归档助手。根据文件名和文件开头的内容判断这是一份什么文件，只输出一行 JSON，不要解释：\n"
    '{{"类型": "", "对象": "", "时间": "", "摘要": ""}}\n'
    + _CARD_RULES.replace("{", "{{").replace("}", "}}") +
    "\n文件名：{name}\n开头内容：{text}"
)

# Several files in one call (settings.doc_card_batch > 1): the instructions above are about 530
# of a card's 810 tokens, so files sharing them cost about a third less; but Qwen3-8B then got
# the type right less often (0.70 to 0.75 against 0.89, evaluation/doc_cards_after_import), so
# the default stays one file per call. Same rules and fields, same input hash as a card asked alone.
CARD_BATCH_PROMPT = (
    "你是文件归档助手。下面有 {count} 份文件，每份给出文件名和开头的内容。逐份判断它是一份什么文件，"
    "按编号顺序每份输出一行 JSON，共 {count} 行，不要解释：\n"
    '{{"编号": 1, "类型": "", "对象": "", "时间": "", "摘要": ""}}\n'
    + _CARD_RULES.replace("{", "{{").replace("}", "}}") +
    "- 每份文件单独判断，不要把别的文件的内容算进来。\n\n"
    "{files}"
)
CARD_BATCH_FILE = "【文件 {number}】\n文件名：{name}\n开头内容：{text}\n\n"


def opening_text(parsed: dict[str, Any], limit: int = CARD_CHARS) -> str:
    text = " ".join(str(unit.get("text", "")) for unit in parsed.get("units") or [])
    return re.sub(r"\s+", " ", text).strip()[:limit]


def card_input_hash(name: str, opening: str, model: str = "") -> str:
    payload = json.dumps([CARD_VERSION, model, name, opening], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def parse_card(raw: str) -> dict[str, str]:
    """The model's one line of JSON, tolerant of fences and extra text."""
    raw = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.S).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the card reply")
    try:
        data = json.loads(raw[start : end + 1])
    except ValueError:
        # An unescaped quote inside the summary (γ"相) or a value left unquoted ("类型": 报告)
        # breaks the JSON: read field by field.
        data = {key: value for key, value in re.findall(r'"(类型|对象|时间|摘要)"\s*:\s*"([^"\n]*)"', raw)}
        for key, value in re.findall(r'"(类型|对象|时间|摘要)"\s*:\s*([^"\s,，{}][^",，{}\n]*)', raw):
            data.setdefault(key, value.strip())
        if "类型" not in data:
            raise
    if not isinstance(data, dict):
        raise ValueError("card reply is not an object")

    def field(*keys: str, limit: int = 60) -> str:
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                return re.sub(r"\s+", " ", str(value)).strip()[:limit]
        return ""

    label = field("类型", "type")
    label = re.sub(r"^其[他它]\s*[:：]\s*", "其他：", label)
    return {
        "label": label,
        "doc_type": normalize_type(label),
        "subject": field("对象", "subject"),
        "period": field("时间", "time", limit=30),
        "summary": field("摘要", "summary", limit=80),
    }


_OBJECT = re.compile(r"\{[^{}]*\}")


def parse_cards(raw: str, count: int) -> list[dict[str, str] | None]:
    """A batch reply: the card of each numbered file, None where the reply has none (that file
    is asked again on its own)."""
    raw = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.S)
    cards: list[dict[str, str] | None] = [None] * count
    for position, match in enumerate(_OBJECT.finditer(raw)):
        text = match.group(0)
        number = re.search(r'"编号"\s*:\s*"?(\d+)', text)
        index = int(number.group(1)) - 1 if number else position
        if not 0 <= index < count or cards[index] is not None:
            continue
        try:
            cards[index] = parse_card(text)
        except (ValueError, TypeError):
            continue
    return cards


OTHER_TYPE = "其他"


def file_card(name: str, card: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the library page shows for a file: its type the way "find a file" sees it (the
    file's own name wins over the card, see FileSearchService._typed_files) and the card's
    other fields. None when neither the name nor a card says anything."""
    by_name = name_doc_type(name)
    if not card and not by_name:
        return None
    card = card or {}
    model_type = str(card.get("doc_type") or "")
    label = str(card.get("label") or "")
    return {
        "type": by_name or model_type or OTHER_TYPE,
        # The model's own words when they say more than the type ("其他：面试题", "政策集合").
        "label": label if label and label not in (model_type, by_name) else "",
        "subject": str(card.get("subject") or ""),
        "period": str(card.get("period") or ""),
        "summary": str(card.get("summary") or ""),
        "has_card": bool(card),
        # The name decided the type (no card yet, or the card said another type).
        "by_name": bool(by_name) and by_name != model_type,
        "card_type": model_type,
    }


def subject_matches(subject: str, owner: str) -> bool:
    """Is the card about the user ("我的简历")? Names are compared without spaces."""
    owner = re.sub(r"\s+", "", owner or "")
    subject = re.sub(r"\s+", "", subject or "")
    return bool(owner) and len(owner) >= 2 and owner.lower() in subject.lower()


# ---------------- building cards ----------------


def _rate_limited(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status == 429


# Who waits for a card, most urgent first.
ASKED = 0  # the user is looking at it: listed by a search, opened, a likely match of a type question
NEW = 1  # added or changed on its own, not as part of a big import
BACKFILL = 2  # the rest (after a big import, or missing at start): newest modified first

# No question for this long: the idle token budget applies, and the gap between calls shrinks
# to this share of ``interval`` so the budget, not the gap, sets the pace.
IDLE_AFTER = 600.0
IDLE_GAP_SHARE = 1 / 3
# Tokens a card costs when the reply does not say: the instructions are about 530, the
# opening about 0.45 a character (evaluation/doc_cards_after_import).
_FIXED_TOKENS = 530
_TOKENS_PER_CHAR = 0.45
# Reply room per file in a batch (a card is 45 to 60 tokens).
_BATCH_REPLY_TOKENS = 100


def estimated_tokens(openings: list[str]) -> int:
    """What one call for these openings costs, for pacing when the service gives no count."""
    if not openings:
        return 0
    return int(_FIXED_TOKENS + sum(60 + _TOKENS_PER_CHAR * len(text) for text in openings))


class DocCardBuilder:
    """Writes cards for indexed files in one background thread.

    * which files: never one kept by name only (``eligible`` decides, from the registry), nor
      one whose own name gives the type ("01 申报书.pdf": the name wins over a card anyway,
      see FileSearchService._typed_files);
    * in what order: files the user is looking at (``prioritize``) first, then a file added on
      its own, then the rest, newest modified first;
    * up to ``batch`` files share one model call when several are waiting;
    * paced by tokens: at most ``tokens_per_minute`` (``idle_tokens_per_minute`` after ten
      minutes without a question), at least ``interval`` seconds between calls (a third of it
      when idle, so the budget sets the pace); nothing while a
      question is answered (``pause``) and, except for files the user is looking at, for
      ``cool_down`` seconds after one and while a big import runs (``library.importing``);
    * a rate limit (限流) waits 25 to 90 seconds and halves the pace, which recovers slowly; a
      file still failing is left for the next sync.
    """

    def __init__(
        self,
        store,
        settings,
        chat_model: Callable[[], Any],
        *,
        files: Callable[[], dict[str, str]] | None = None,
        eligible: Callable[[str], str | None] | None = None,
        pause: Callable[[], None] | None = None,
        interval: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        library=None,
        batch: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.store = store
        self.settings = settings
        self.chat_model = chat_model
        # doc_id -> name of every file that may have a card (indexed, not name-only).
        self.files = files or (lambda: {})
        # doc_id -> its current name, or None when it may not have a card (any more).
        self.eligible = eligible or (lambda doc_id: self.files().get(doc_id))
        # The file library (LibraryService): questions being answered, big imports, file times.
        self.library = library
        self.pause = pause or (library.pause_for_questions if library is not None else None)
        self.base_interval = float(getattr(settings, "doc_card_interval_seconds", 3.0) if interval is None else interval)
        # Grows on each rate limit (up to MAX_INTERVAL), shrinks back after successes; the
        # token budget is scaled by base_interval / interval with it.
        self.interval = self.base_interval
        self.batch = max(1, int(getattr(settings, "doc_card_batch", 1) if batch is None else batch))
        self.tokens_per_minute = float(getattr(settings, "doc_card_tokens_per_minute", 9000.0) or 0.0)
        self.idle_tokens_per_minute = float(
            getattr(settings, "doc_card_idle_tokens_per_minute", 0.0) or self.tokens_per_minute)
        self.cool_down = float(getattr(settings, "doc_card_cool_down_seconds", 60.0) or 0.0)
        self.sleep = sleep
        self.clock = clock
        # (priority, -modified, order, doc_id); an entry whose priority is no longer the file's
        # (it was asked for since) is skipped when it comes up.
        self._heap: list[tuple[int, float, int, str]] = []
        self._queued: dict[str, int] = {}
        self._order = 0
        # Files a batch reply left out: asked on their own next time.
        self._alone: set[str] = set()
        self._wake = threading.Condition()
        self._worker: threading.Thread | None = None
        self._stopped = False
        self._next_call = 0.0
        self._model = None
        self.stats = {"built": 0, "cached": 0, "failed": 0, "skipped": 0, "named": 0, "retry": 0, "tokens": 0,
                      "seconds": 0.0, "rate_limited": 0, "calls": 0, "asked": 0}

    RETRY_DELAYS = (25.0, 30.0, 60.0, 90.0)
    MAX_INTERVAL = 60.0

    @property
    def supported(self) -> bool:
        return hasattr(self.store, "upsert_doc_card")

    @property
    def model_name(self) -> str:
        return str(getattr(self.settings, "llm_model", ""))

    # ---------------- the library ----------------

    def _busy(self) -> bool:
        """A big import is running: the backfill waits for it (a file indexed in it too)."""
        try:
            return bool(self.library is not None and self.library.importing())
        except Exception:
            return False

    def _idle(self) -> float:
        """Seconds since the last question was answered (inf when none was)."""
        try:
            return float(self.library.idle_seconds()) if self.library is not None else float("inf")
        except Exception:
            return float("inf")

    def _modified(self, doc_ids: Iterable[str] | None = None) -> dict[str, float]:
        """The files' modification times (the newest are built first in the backfill)."""
        registry = getattr(self.library, "registry", None)
        if registry is None:
            return {}
        try:
            if doc_ids is not None:
                rows = [registry.get_file(doc_id) for doc_id in doc_ids]
            else:
                from library.registry import SEARCHABLE_STATUSES

                rows = registry.list_files(statuses=SEARCHABLE_STATUSES)
            return {row["file_id"]: float(row.get("mtime") or 0.0) for row in rows if row}
        except Exception:
            return {}

    # ---------------- inputs ----------------

    def card_input(self, doc_id: str, name: str) -> tuple[str, str] | None:
        """(opening text, input hash), or None without a parser snapshot."""
        path = Path(self.settings.parsed_dir) / f"{doc_id}.json"
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        opening = opening_text(parsed, int(getattr(self.settings, "doc_card_chars", CARD_CHARS)))
        return opening, card_input_hash(name, opening, self.model_name)

    # ---------------- pace ----------------

    def rate(self) -> float:
        """Tokens a minute the cards may use now (0: no budget, only ``interval``)."""
        budget = self.idle_tokens_per_minute if self._idle() >= IDLE_AFTER else self.tokens_per_minute
        if budget <= 0:
            return 0.0
        slower = self.base_interval / self.interval if self.interval > 0 and self.base_interval > 0 else 1.0
        return budget * slower

    def _wait_turn(self, urgent: Callable[[], bool] | bool = False) -> None:
        """Wait until a call may be made: no question being answered; unless the files are
        ones the user is looking at, no big import and the cool-down after a question over;
        then the pace."""
        check = urgent if callable(urgent) else (lambda: bool(urgent))
        self._pause()
        while not self._stopped and not check():
            if self._busy():
                with self._wake:
                    self._wake.wait(timeout=5)
                continue
            left = self.cool_down - self._idle()
            if left <= 0:
                break
            self.sleep(min(left, 5.0))
            self._pause()
        wait = self._next_call - self.clock()
        if wait > 0:
            self.sleep(wait)

    def _pause(self) -> None:
        if self.pause is not None:
            try:
                self.pause()
            except Exception as exc:  # the courtesy wait never stops the builder
                logger.debug("Doc card pause failed: %s", exc)

    def _after_call(self, started: float, tokens: int) -> None:
        ended = self.clock()
        rate = self.rate()
        spacing = tokens / rate * 60.0 if rate > 0 else 0.0
        gap = self.interval * (IDLE_GAP_SHARE if rate > 0 and self._idle() >= IDLE_AFTER else 1.0)
        self._next_call = max(ended + gap, started + spacing)

    # ---------------- building ----------------

    def build(self, doc_id: str, name: str | None = None) -> str:
        """Write or refresh one file's card. Returns built | cached | named | skipped | failed."""
        return self.build_many([doc_id]).get(doc_id, "skipped")

    def build_many(self, doc_ids: list[str], wait: bool = True, urgent: bool = False) -> dict[str, str]:
        """Write or refresh these files' cards, several files in one model call. ``wait``: wait
        for the turn first (``_wait_turn``; the background loop has already waited)."""
        if not self.supported:
            return {doc_id: "skipped" for doc_id in doc_ids}
        outcomes: dict[str, str] = {}
        hashes = self.store.doc_card_hashes()
        todo: list[tuple[str, str, str, str]] = []
        for doc_id in doc_ids:
            current = self.eligible(doc_id)
            if current is None:
                # Kept by name only, removed, not indexed: no card (and none left behind).
                self.store.delete_doc_card(doc_id)
                outcomes[doc_id] = "skipped"
                continue
            name = current or doc_id
            if name_doc_type(name):
                outcomes[doc_id] = "named"  # its name gives the type: no call
                continue
            prepared = self.card_input(doc_id, name)
            if prepared is None:
                outcomes[doc_id] = "skipped"
                continue
            opening, digest = prepared
            if hashes.get(doc_id) == digest:
                outcomes[doc_id] = "cached"
                continue
            todo.append((doc_id, name, opening, digest))
        if todo and wait:
            self._wait_turn(urgent)
        # Checked again after the wait: a file may have been switched to name-only.
        for item in list(todo):
            if self.eligible(item[0]) is None:
                self.store.delete_doc_card(item[0])
                outcomes[item[0]] = "skipped"
                todo.remove(item)
        if todo:
            outcomes.update(self._write(todo))
        for outcome in outcomes.values():
            self.stats[outcome] = self.stats.get(outcome, 0) + 1
        return outcomes

    def _write(self, todo: list[tuple[str, str, str, str]]) -> dict[str, str]:
        started = self.clock()
        tokens = 0
        try:
            cards, tokens = self._ask_files([(name, opening) for _, name, opening, _ in todo])
        except Exception as exc:
            logger.warning("Doc cards for %d file(s) not written: %s", len(todo), exc)
            return {doc_id: "failed" for doc_id, _, _, _ in todo}
        finally:
            self.stats["calls"] += 1
            self.stats["seconds"] += self.clock() - started
            self._after_call(started, tokens or estimated_tokens([opening for _, _, opening, _ in todo]))
        self.stats["tokens"] += tokens
        share = round(tokens / len(todo))
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        outcomes: dict[str, str] = {}
        for (doc_id, _, _, digest), card in zip(todo, cards):
            if card is None:
                # Left out of a batch reply: asked on its own next time.
                self._alone.add(doc_id)
                self.enqueue(doc_id, NEW)
                outcomes[doc_id] = "retry"
                continue
            self._alone.discard(doc_id)
            if self.eligible(doc_id) is None:
                self.store.delete_doc_card(doc_id)
                outcomes[doc_id] = "skipped"
                continue
            self.store.upsert_doc_card(doc_id, {**card, "input_hash": digest, "version": CARD_VERSION,
                                                "tokens": share, "model": self.model_name, "created_at": created})
            outcomes[doc_id] = "built"
        return outcomes

    def _ask_files(self, items: list[tuple[str, str]]) -> tuple[list[dict[str, str] | None], int]:
        if len(items) == 1:
            card, tokens = self._ask(*items[0])
            return [card], tokens
        files = "".join(CARD_BATCH_FILE.format(number=number, name=name, text=opening or "（无文字内容）")
                        for number, (name, opening) in enumerate(items, start=1))
        message = self._invoke(CARD_BATCH_PROMPT.format(count=len(items), files=files),
                               max_tokens=_BATCH_REPLY_TOKENS * len(items))
        usage = getattr(message, "usage_metadata", None) or {}
        return parse_cards(str(message.content), len(items)), int(usage.get("total_tokens") or 0)

    def _ask(self, name: str, opening: str) -> tuple[dict[str, str], int]:
        message = self._invoke(CARD_PROMPT.format(name=name, text=opening or "（无文字内容）"))
        usage = getattr(message, "usage_metadata", None) or {}
        tokens = int(usage.get("total_tokens") or 0)
        try:
            return parse_card(str(message.content)), tokens
        except (ValueError, TypeError) as exc:
            # A reply that is not JSON is kept as "unknown type", not asked again until the
            # file changes: the same input would most likely get the same reply.
            logger.info("Doc card reply unreadable for %s: %s", name, exc)
            return {"label": "", "doc_type": "", "subject": "", "period": "", "summary": ""}, tokens

    def _invoke(self, prompt: str, max_tokens: int | None = None):
        from langchain_core.messages import HumanMessage
        from services.model_clients import background_calls, transient_error

        if self._model is None:
            self._model = self.chat_model()
        model = self._model
        if max_tokens and hasattr(model, "bind"):
            model = model.bind(max_tokens=max_tokens)
        for attempt in range(len(self.RETRY_DELAYS) + 1):
            try:
                with background_calls.slot():
                    message = model.invoke([HumanMessage(content=prompt)])
                break
            except Exception as exc:
                if attempt >= len(self.RETRY_DELAYS) or not transient_error(exc):
                    raise
                if _rate_limited(exc):
                    # 限流: the account's per-minute budget is shared with answers; slow down.
                    self.stats["rate_limited"] += 1
                    self.interval = min(self.MAX_INTERVAL, max(self.interval, 1.0) * 2)
                logger.info("Doc card call failed (%s), retrying in %.0f s", type(exc).__name__, self.RETRY_DELAYS[attempt])
                self.sleep(self.RETRY_DELAYS[attempt])
        self.interval = max(self.base_interval, self.interval * 0.8)
        return message

    # ---------------- the queue ----------------

    def start(self) -> None:
        if self._worker is None and self.supported:
            self._worker = threading.Thread(target=self._loop, name="doc-cards", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        with self._wake:
            self._stopped = True
            self._wake.notify_all()

    def enqueue(self, doc_id: str, priority: int = BACKFILL, modified: float | None = None) -> bool:
        """Queue a file, or move it up. Returns whether it was."""
        if modified is None:
            modified = self._modified([doc_id]).get(doc_id, 0.0) if priority == BACKFILL else 0.0
        with self._wake:
            if self._queued.get(doc_id, BACKFILL + 1) <= priority:
                return False
            self._queued[doc_id] = priority
            self._order += 1
            heapq.heappush(self._heap, (priority, -float(modified or 0.0), self._order, doc_id))
            self._wake.notify_all()
            return True

    def file_indexed(self, doc_id: str) -> None:
        """A file was (re)indexed: on its own it comes next; in a big import, with the backfill."""
        self.enqueue(doc_id, BACKFILL if self._busy() else NEW)

    def prioritize(self, doc_ids: Iterable[str]) -> int:
        """Files the user is looking at (listed by a search, opened, likely matches of a type
        question): their cards come first, even during a big import. Files with a card, or
        whose name gives the type, are left alone. Returns how many were moved up."""
        if not self.supported:
            return 0
        ids = [doc_id for doc_id in dict.fromkeys(doc_ids) if doc_id]
        if not ids:
            return 0
        try:
            have = self.store.doc_card_hashes()
        except Exception:
            return 0
        moved = 0
        for doc_id in ids:
            if doc_id in have:
                continue
            name = self.eligible(doc_id)
            if name is None or name_doc_type(name):
                continue
            moved += self.enqueue(doc_id, ASKED)
        self.stats["asked"] += moved
        return moved

    def pending(self) -> int:
        with self._wake:
            return len(self._queued)

    def progress(self) -> dict[str, Any]:
        """Files waiting, how many of them the user is looking at, and the time left at the
        current pace (seconds; None without a token budget)."""
        with self._wake:
            waiting = len(self._queued)
            asked = sum(1 for priority in self._queued.values() if priority == ASKED)
        built = self.stats["built"]
        per_file = (self.stats["tokens"] / built if built and self.stats["tokens"]
                    else _FIXED_TOKENS / self.batch + 60 + _TOKENS_PER_CHAR * 650)
        rate = self.rate()
        return {"waiting": waiting, "asked": asked, "tokens_per_minute": round(rate),
                "tokens_per_file": round(per_file),
                "seconds_left": round(waiting * per_file / rate * 60) if rate > 0 else None}

    def sync(self) -> int:
        """Queue every eligible file whose card is missing or out of date (newest first); drop
        cards of files that may no longer have one. Returns how many were queued."""
        if not self.supported:
            return 0
        self.store.renormalize_doc_cards(normalize_type)
        files = self.files()
        hashes = self.store.doc_card_hashes()
        for doc_id in set(hashes) - set(files):
            if self.eligible(doc_id) is None:
                self.store.delete_doc_card(doc_id)
        modified = self._modified()
        queued = 0
        for doc_id, name in files.items():
            if name_doc_type(name):
                continue  # its name gives the type
            if doc_id in hashes:
                # A card from another input (the file or the prompt changed) is written again.
                prepared = self.card_input(doc_id, name)
                if prepared is None or hashes[doc_id] == prepared[1]:
                    continue
            queued += self.enqueue(doc_id, BACKFILL, modified.get(doc_id, 0.0))
        if queued:
            logger.info("Doc cards: %d files queued", queued)
        return queued

    def _take(self, urgent_only: bool = False) -> tuple[list[str], bool]:
        """The next files to ask about together, most urgent first, and whether one of them is
        a file the user is looking at. A file a batch reply left out goes alone."""
        with self._wake:
            batch: list[str] = []
            urgent = False
            while self._heap and len(batch) < self.batch:
                priority, _, _, doc_id = self._heap[0]
                if self._queued.get(doc_id) != priority:
                    heapq.heappop(self._heap)  # moved up since
                    continue
                if urgent_only and priority > ASKED:
                    break
                if batch and doc_id in self._alone:
                    break
                heapq.heappop(self._heap)
                del self._queued[doc_id]
                urgent = urgent or priority == ASKED
                batch.append(doc_id)
                if doc_id in self._alone:
                    break
            return batch, urgent

    def _top_urgent(self) -> bool:
        with self._wake:
            while self._heap and self._queued.get(self._heap[0][3]) != self._heap[0][0]:
                heapq.heappop(self._heap)
            return bool(self._heap) and self._heap[0][0] == ASKED

    def run_pending(self) -> dict[str, int]:
        """Build every queued card now, in this thread (tests, evaluation)."""
        counts: dict[str, int] = {}
        while not self._stopped and self.pending():
            self._wait_turn(self._top_urgent)
            batch, urgent = self._take()
            for outcome in self.build_many(batch, wait=False, urgent=urgent).values():
                counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    def _loop(self) -> None:
        while True:
            with self._wake:
                while not self._queued and not self._stopped:
                    self._wake.wait(timeout=30)
                if self._stopped:
                    return
            try:
                self._wait_turn(self._top_urgent)
                batch, urgent = self._take(urgent_only=self._busy())
                if batch:
                    self.build_many(batch, wait=False, urgent=urgent)
            except Exception as exc:
                logger.warning("Doc cards failed: %s", exc)
