"""Follow-up questions after an answer: the likeliest next question and a few alternatives.

They are written only from what this turn already used: the question, the answer, and the
passages retrieval found (file names and section titles, plus the titles of the sections
next to them in the same files). The question, answer and passages went to the model
service with the answer, so asking for follow-ups sends nothing new off the machine.

* One short model call (a few hundred tokens, a 4-second deadline) writes them; each
  must be answerable from a listed passage or section. The reply streams in, so at the
  deadline the questions that have fully arrived are kept and templates add the rest.
* Without a key, on a timeout or a model error, local templates write them from the
  cited files' neighbouring sections and the names of the files found.
* After small talk they come from the home-page pool, with no model call.
* Questions already asked in this conversation, and rewordings of the last one, are dropped.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Iterable

from config import MissingApiKeyError, Settings
from services.model_clients import chat_busy
from rag_agent.answering import CITATION, cited_claims, evidence_for
from retrieval.file_filters import FILE_TYPES, parse_query

from .templates import _heading_question, clean_heading, clean_title, document_kind, title_key

logger = logging.getLogger(__name__)

# What the turn was, which decides what kind of follow-up makes sense.
SCENE_ANSWER = "answer"      # an answer from the library
SCENE_SCOPED = "scoped"      # an answer while asking within one file
SCENE_FILES = "files"        # files found
SCENE_REFUSED = "refused"    # not enough evidence, or no file found
SCENE_CHAT = "chat"          # small talk: home-page questions
SCENE_NONE = ""              # a clarifying question or a model failure: nothing to suggest

# How a follow-up changes the conversation's question scope when it is sent.
SCOPE_KEEP, SCOPE_SET, SCOPE_CLEAR = "keep", "set", "clear"

FIND_PREFIX = re.compile(r"^(?:找一下|帮我找|找找|搜一下|查找|找出)")
TYPE_LABEL = {"pdf": "PDF 文件", "word": "Word 文档", "excel": "Excel 表格", "ppt": "PPT",
              "markdown": "Markdown 文件", "text": "TXT 文件", "image": "图片"}
_EXT_TYPE = {ext: label for label, extensions in FILE_TYPES.items() for ext in extensions}

MAX_OUTLINES = 32
MAX_UNUSED = 6
MAX_NEIGHBOURS = 4
SUMMARY_CHARS = 300
# Rewording thresholds (character-bigram Jaccard, see ``similarity``).
SAME_AS_LAST = 0.6
SAME_AS_EARLIER = 0.8
SAME_AS_OTHER = 0.7
# The same on the topic words alone (question words stripped): "什么是 KV Cache" = "KV Cache 是什么".
SAME_TOPIC_AS_ASKED = 0.9
SAME_TOPIC_AS_OTHER = 0.85
# Share of a model question's content bigrams that must come from this turn's material.
ON_TOPIC = 0.34
# A question asking for one of these needs the word in this turn's answer or passages;
# otherwise the passages most likely do not give it, and the question goes after the others.
SPECIFICS = (
    "公式", "具体数值", "数值", "参数设置", "超参数", "阈值", "金额", "价格", "费率", "费用", "成本", "预算",
    "投资", "融资", "估值", "营收", "哪一年", "哪年", "日期", "具体时间", "百分比", "准确率", "多少钱",
)


@dataclass
class Followup:
    text: str
    kind: str
    label: str = ""
    scope_action: str = SCOPE_KEEP
    scope: dict[str, Any] | None = None
    # A model question that most likely asks for more than the passages give: never the first.
    doubtful: bool = field(default=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"text": self.text, "kind": self.kind, "label": self.label,
                                "scope_action": self.scope_action}
        if self.scope is not None:
            data["scope"] = self.scope
        return data


@dataclass(frozen=True)
class Passage:
    doc_id: str
    source: str
    heading: str
    locator: str = ""
    text: str = ""

    @property
    def section(self) -> str:
        return clean_heading(self.heading)


@dataclass
class Turn:
    """What one finished turn left in the graph state, as the follow-ups need it."""

    question: str
    answer: str = ""
    intent: str = ""
    rewritten: str = ""
    refused: bool = False
    clarifying: bool = False
    model_error: bool = False
    scope: list[str] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    cited: list[Passage] = field(default_factory=list)
    unused: list[Passage] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    file_types: list[str] = field(default_factory=list)

    @classmethod
    def from_state(cls, values: dict[str, Any]) -> "Turn":
        answer = str(values.get("answer") or "")
        blocks = [_passage(d) for d in (values.get("context_docs") or values.get("documents") or [])]
        cited: list[Passage] = []
        refs = [ref for _, claim_refs in cited_claims(answer) for ref in claim_refs]
        if refs and blocks:
            indexes, _ = evidence_for(refs, [{"source": b.source, "locator": b.locator or b.heading} for b in blocks])
            cited = _distinct(blocks[i] for i in indexes)
        cited_keys = {(p.doc_id, p.section) for p in cited}
        retrieved = [*blocks, *(_passage(d) for d in values.get("documents") or [])]
        unused = [p for p in _distinct(retrieved) if (p.doc_id, p.section) not in cited_keys][:MAX_UNUSED]
        asked = [str(getattr(m, "content", "")).strip() for m in values.get("messages") or []
                 if getattr(m, "type", "") == "human"]
        filters = (values.get("file_search") or {}).get("filters") or {}
        return cls(
            question=str(values.get("question") or "").strip(),
            answer=answer,
            intent=str(values.get("intent") or ""),
            rewritten=str(values.get("rewritten_query") or ""),
            refused=bool(values.get("refused")),
            clarifying=bool((values.get("plan") or {}).get("needs_clarification")),
            model_error=bool(values.get("model_error")),
            scope=list(values.get("scope_doc_ids") or []),
            asked=[q for q in asked if q],
            cited=cited,
            unused=unused,
            files=list(values.get("files") or []),
            file_types=list(filters.get("file_types") or []),
        )

    @property
    def scenario(self) -> str:
        if self.clarifying or self.model_error:
            return SCENE_NONE
        if self.intent == "find_file":
            return SCENE_FILES if self.files else SCENE_REFUSED
        if self.intent == "chitchat":
            return SCENE_CHAT
        if self.refused or not self.answer:
            return SCENE_REFUSED
        return SCENE_SCOPED if len(self.scope) == 1 else SCENE_ANSWER


def sendable(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Found files the model may hear about: not those kept by name only (library/privacy.py),
    whose name and text never leave the machine. The local templates still use them."""
    return [item for item in files if item.get("status") != "named"]


def _passage(doc: dict[str, Any]) -> Passage:
    return Passage(
        doc_id=str(doc.get("doc_id", "")),
        source=str(doc.get("source", "")),
        heading=str(doc.get("heading", "")),
        locator=str(doc.get("locator", "")),
        text=str(doc.get("child_text") or doc.get("text") or ""),
    )


def _distinct(passages: Iterable[Passage]) -> list[Passage]:
    seen: set[tuple[str, str]] = set()
    out = []
    for passage in passages:
        key = (passage.doc_id, passage.section or passage.heading)
        if passage.doc_id and key not in seen:
            seen.add(key)
            out.append(passage)
    return out


# ---------------- rewording ----------------

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    return _NON_WORD.sub("", str(text).lower())


def _bigrams(text: str) -> set[str]:
    text = normalize(text)
    return {text[i : i + 2] for i in range(len(text) - 1)} or ({text} if text else set())


def similarity(a: str, b: str) -> float:
    """Character-bigram Jaccard of two questions, punctuation and spacing ignored."""
    left, right = _bigrams(a), _bigrams(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def dedupe(items: Iterable[Followup], asked: list[str], n: int) -> list[Followup]:
    """Drop questions already asked, rewordings of the last one and near-duplicates."""
    asked_norm = {normalize(q) for q in asked}
    last = asked[-1] if asked else ""
    kept: list[Followup] = []
    for item in items:
        if not item.text or len(kept) >= n:
            continue
        # "在整个知识库中重问" and "只要 PDF" repeat the question on purpose, with another scope or filter.
        repeats = item.kind in ("widen", "refine_search")
        if normalize(item.text) in asked_norm and not (repeats and item.scope_action != SCOPE_KEEP):
            continue
        # Asking about a found file shares its topic with "找一下…" but is another request.
        same_topic = SAME_TOPIC_AS_ASKED if item.scope_action != SCOPE_SET else 2.0
        if not repeats and last and (
            similarity(item.text, last) >= SAME_AS_LAST or topic_similarity(item.text, last) >= same_topic
        ):
            continue
        if not repeats and any(
            similarity(item.text, q) >= SAME_AS_EARLIER or topic_similarity(item.text, q) >= same_topic
            for q in asked[:-1]
        ):
            continue
        if any(
            normalize(item.text) == normalize(k.text)
            or similarity(item.text, k.text) >= SAME_AS_OTHER
            or topic_similarity(item.text, k.text) >= SAME_TOPIC_AS_OTHER
            for k in kept
        ):
            continue
        kept.append(item)
    return kept


# ---------------- staying on the material ----------------

_FILLER = re.compile(
    r"什么是|是什么|什么|如何|怎么样|怎么|怎样|为什么|为何|哪些|哪个|哪份|有没有|是否|可以|能否|能不能|"
    r"请|一下|举个|举例|例子|具体|详细|展开|讲讲|说说|介绍|区别|对比|比较|主要|内容|部分|问题|方面|"
    r"约定|提到|包括|还有|分别|其中|这份|这个|那个|文件|资料|文档|找|有|的|了|吗|呢|和|与|及|在|中|是|会|要|用|对|讲"
)


def _content_bigrams(text: str) -> set[str]:
    return _bigrams(_FILLER.sub("", normalize(text)))


def topic_similarity(a: str, b: str) -> float:
    """``similarity`` on what two questions are about, their question words left out."""
    left, right = _content_bigrams(a), _content_bigrams(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def on_topic(question: str, material: set[str]) -> bool:
    """Most of what a question is about must appear in this turn's material."""
    grams = _content_bigrams(question)
    return not grams or len(grams & material) / len(grams) >= ON_TOPIC


# "…具体怎么映射成…？", "…差距是多少？": how-exactly probes into what the answer did not spell out.
# On real data these were the follow-ups that went unanswered ("文档只给出了设计思路，没有给出具体…").
DETAIL_PROBE = re.compile(
    r"具体(?:是|会|要|该)?(?:怎么|如何|怎样)|(?:差距|差异|差别|提升|提高|降低|减少|差|高|低|快|慢|好)(?:是|有|了)?多少|"
    r"具体(?:的)?(?:数值|数字|参数|步骤|细节|算法|实现|做法)"
)


def wants_missing_detail(question: str, evidence: str) -> bool:
    """Asks for a formula, an amount, a date… that this turn's answer and passages never mention."""
    return any(word in question and word not in evidence for word in SPECIFICS)


def doubtful(question: str, evidence: str) -> bool:
    return wants_missing_detail(question, evidence) or bool(DETAIL_PROBE.search(question))


def keeps_topic(question: str, asked: str) -> bool:
    """After a refusal, a rephrasing still names something the user asked about."""
    return bool(_content_bigrams(question) & _content_bigrams(asked))


class _Reply:
    """A model reply as it streams in."""

    def __init__(self):
        self.text = ""
        self.usage: dict[str, int] | None = None
        self.lock = threading.Lock()

    def add(self, text: str) -> None:
        with self.lock:
            self.text += text

    def complete_lines(self) -> str:
        """What arrived up to the last line break: a question cut off midway is left out."""
        with self.lock:
            return self.text[: self.text.rfind("\n") + 1]


# ---------------- the model ----------------

PROMPT_HEAD = """你在帮知识库问答应用的用户想“下一句可能会问什么”。根据下面这一轮的内容，写 {n} 个用户最可能接着问的问题。
要求：
- 每个问题都必须能用下面列出的资料回答，不要问资料里没有的话题，也不要编造文件名或章节；
- 只问资料里明确写到的内容：回答摘要里没出现的具体数字、公式、参数、日期、人名，不要追问；
- 不要用“具体怎么……”“差距是多少”去追问回答里没有展开的实现细节，资料多半没写；
- 不要重复“已经问过”的问题，也不要只是换个说法重问本轮问题；
- 第一个写最可能、而且资料里一定有答案的那个；
- 中文，口语自然，每个不超过 30 个字；问题要能单独看懂：写出具体名词，不用“它”“这个”“上面”；
- 每行一个问题，不要编号，不要解释。
"""

SCENE_GUIDE = {
    SCENE_ANSWER: "方向：深入回答里的某一点、要一个例子、和相关概念对比，或者问相邻小节的内容。",
    SCENE_SCOPED: "用户正在只针对《{file}》这一个文件提问。方向：优先问这个文件里接下来的小节，其次深入本轮回答。问题只能关于这个文件。",
    SCENE_FILES: ("用户刚才在找文件。方向：前两个问题针对找到的文件的内容提问（优先第 1 个文件），问题里用《文件名》写明是哪个文件，"
                  "要能从它的匹配片段或小节回答；最后一个可以是缩小范围或换个条件重新找，以“找一下”开头。"),
    SCENE_REFUSED: ("这一轮没能从知识库得到回答（证据不足或没找到文件）。方向：保留本轮问题的主题（产品名、概念名等），"
                    "换成下面片段里更可能有答案的问法；或者改为找相关的文件，以“找一下”开头。"
                    "不要转去问片段里和本轮问题无关的话题。"),
}


def _summary(answer: str) -> str:
    text = CITATION.sub("", answer)
    text = re.sub(r"[#>*`|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:SUMMARY_CHARS] + ("…" if len(text) > SUMMARY_CHARS else "")


def _spaced(word: str) -> str:
    """A Latin word set off by spaces inside Chinese text: "和 Manus 有关"."""
    left = " " if word[:1].isascii() and word[:1].isalnum() else ""
    right = " " if word[-1:].isascii() and word[-1:].isalnum() else ""
    return f"{left}{word}{right}"


def _mentions(passage: "Passage", subject: str) -> bool:
    return not subject or normalize(subject) in normalize(f"{passage.source}{passage.heading}{passage.text}")


def _model_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in str(text).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+\s*[.、)）:：]|[（(]\d+[)）]|Q\d*[:：])\s*", "", line).strip()
        line = line.strip("\"“”'‘’ ")
        if not (5 <= len(line) <= 60) or line.endswith(("：", ":")):
            continue
        if not FIND_PREFIX.match(line) and not line.endswith(("？", "?")):
            line += "？"
        if line not in out:
            out.append(line)
    return out


@dataclass
class Result:
    scenario: str
    items: list[Followup]
    source: str = ""          # "model" | "template" | "pool" | ""
    fallback: str = ""        # why the model was not used: "no_key" | "timeout" | "error" | "empty"
    elapsed_ms: int = 0
    usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "source": self.source,
            "fallback": self.fallback,
            "elapsed_ms": self.elapsed_ms,
            "usage": self.usage,
            "items": [item.to_dict() for item in self.items],
        }


class FollowupService:
    def __init__(
        self,
        settings: Settings,
        chat_model: Callable[[], Any] | None = None,
        pool: Callable[[int], list[dict[str, Any]]] | None = None,
        describe_file: Callable[[str], dict[str, Any] | None] | None = None,
    ):
        self.settings = settings
        self.chat_model = chat_model
        # Home-page questions after small talk: n -> cards (see SuggestionService.suggest).
        self.pool = pool
        self.describe_file = describe_file
        self._outlines: OrderedDict[tuple[str, int], list[tuple[str, str]]] = OrderedDict()
        self._outline_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="followups")
        # Model calls and their tokens, including calls that finished after the deadline.
        self.stats = {"calls": 0, "completed": 0, "input_tokens": 0, "output_tokens": 0}
        self._stats_lock = threading.Lock()

    # ---------------- entry point ----------------

    def suggest(self, values: dict[str, Any], n: int = 4, use_model: bool = True) -> Result:
        started = time.perf_counter()
        if values.get("choices"):
            # This turn asked back ("哪一份？"): the possible replies, no model call.
            return Result(SCENE_NONE, [Followup(str(text), "reply") for text in values["choices"]], source="choices")
        turn = Turn.from_state(values)
        scenario = turn.scenario
        result = Result(scenario, [])
        if scenario == SCENE_NONE or not turn.question:
            return result
        if scenario == SCENE_CHAT:
            result.items, result.source = self._from_pool(turn, n), "pool"
        else:
            neighbours = self.neighbours(turn)
            written: list[Followup] = []
            if use_model and self.chat_model is not None:
                written, result.fallback, result.usage = self._from_model(turn, neighbours, n)
            elif use_model:
                result.fallback = "no_model"
            kept = dedupe(written, turn.asked, n)
            if written and not kept:
                result.fallback = "repeated"
            head = [item for item in kept if not item.doubtful]
            if turn.scenario == SCENE_REFUSED and turn.scope:
                # After a refusal within a file or folder, asking the whole library is always on offer.
                head.insert(min(1, len(head)), Followup(turn.question, "widen", "全库", SCOPE_CLEAR))
            # Templates fill in what the model left out (or everything, without it); doubtful
            # model questions come last, as alternatives only.
            later = [item for item in kept if item.doubtful]
            result.items = dedupe([*head, *self.templates(turn, neighbours), *later], turn.asked, n)
            result.source = "model" if result.items and any(result.items[0] is item for item in kept) else "template"
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Follow-ups: %s via %s%s, %d items, %d ms, tokens %s",
            scenario, result.source or "-", f" ({result.fallback})" if result.fallback else "",
            len(result.items), result.elapsed_ms, result.usage or "-",
        )
        return result

    # ---------------- material ----------------

    def _outline(self, doc_id: str) -> list[tuple[str, str]]:
        """(heading path, locator) of each parsed unit of a file, cached by file version."""
        path = Path(self.settings.parsed_dir) / f"{doc_id}.json"
        try:
            version = path.stat().st_mtime_ns
        except OSError:
            return []
        key = (doc_id, version)
        with self._outline_lock:
            if key in self._outlines:
                self._outlines.move_to_end(key)
                return self._outlines[key]
        try:
            units = json.loads(path.read_text(encoding="utf-8")).get("units") or []
        except (OSError, ValueError):
            return []
        outline = [(str(u.get("heading", "")), str(u.get("locator", ""))) for u in units]
        with self._outline_lock:
            self._outlines[key] = outline
            while len(self._outlines) > MAX_OUTLINES:
                self._outlines.popitem(last=False)
        return outline

    @staticmethod
    def _position(outline: list[tuple[str, str]], passage: Passage) -> int | None:
        for index, (heading, locator) in enumerate(outline):
            if heading == passage.heading and (not passage.locator or locator == passage.locator):
                return index
        for index, (heading, _) in enumerate(outline):
            if heading == passage.heading:
                return index
        last = passage.heading.split(" > ")[-1].strip()
        if last:
            for index, (heading, _) in enumerate(outline):
                if heading.endswith(last):
                    return index
        page = re.match(r"page:(\d+)", passage.locator)
        if page:
            for index, (_, locator) in enumerate(outline):
                bounds = re.fullmatch(r"page:(\d+)(?:-(\d+))?", locator)
                if bounds and int(bounds.group(1)) <= int(page.group(1)) <= int(bounds.group(2) or bounds.group(1)):
                    return index
        return None

    def neighbours(self, turn: Turn) -> list[Passage]:
        """Sections next to the cited ones in the same files: two after, one before."""
        anchors = turn.cited or turn.unused[:1]
        if turn.scenario == SCENE_SCOPED:
            anchors = [p for p in anchors if p.doc_id in turn.scope] or [
                Passage(turn.scope[0], self._file_name(turn.scope[0]), "")]
        covered = {p.section for p in (*turn.cited, *turn.unused)} | {""}
        found: list[Passage] = []
        for anchor in _distinct(anchors)[:2]:
            outline = self._outline(anchor.doc_id)
            if not outline:
                continue
            position = self._position(outline, anchor) if anchor.heading else -1
            if position is None:
                continue
            after = self._sections(outline[position + 1 :], covered, 2 if anchor.heading else 3)
            before = self._sections(reversed(outline[:position]), covered, 1) if position > 0 else []
            for heading in [*after, *before]:
                covered.add(clean_heading(heading))
                found.append(Passage(anchor.doc_id, anchor.source or self._file_name(anchor.doc_id), heading))
            if len(found) >= MAX_NEIGHBOURS:
                break
        return found[:MAX_NEIGHBOURS]

    @staticmethod
    def _sections(units: Iterable[tuple[str, str]], covered: set[str], limit: int) -> list[str]:
        out: list[str] = []
        for heading, _ in units:
            section = clean_heading(heading)
            if section in covered or any(clean_heading(h) == section for h in out):
                continue
            out.append(heading)
            if len(out) >= limit:
                break
        return out

    def _file_name(self, doc_id: str) -> str:
        record = self.describe_file(doc_id) if self.describe_file else None
        return str(record.get("name", "")) if record else ""

    # ---------------- templates ----------------

    def templates(self, turn: Turn, neighbours: list[Passage]) -> list[Followup]:
        """Follow-ups written on this machine from section titles and file names."""
        scenario = turn.scenario
        if scenario == SCENE_FILES:
            return self._file_templates(turn)
        out: list[Followup] = []
        if scenario == SCENE_REFUSED:
            if turn.scope:
                out.append(Followup(turn.question, "widen", "全库", SCOPE_CLEAR))
            subject = self._subject(turn)
            # Within one file nothing can be found or broadened: those two leave the file.
            leave = SCOPE_CLEAR if len(turn.scope) == 1 else SCOPE_KEEP
            if subject:
                noun = "资料" if turn.intent == "find_file" else "文件"
                out.append(Followup(f"找一下和{_spaced(subject)}有关的{noun}", "find_files", "找文件", leave))
                if turn.intent != "find_file":
                    # A broader question about the same thing: the library may cover it in general.
                    label = "全库" if leave == SCOPE_CLEAR else ""
                    out.append(Followup(f"什么是{_spaced(subject).rstrip()}？", "rephrase", label, leave))
            # Weak passages are offered only when they are about the same thing: a refused
            # "Manus 的投资金额" retrieves funding documents that have nothing to do with Manus.
            out += [self._section_followup(p, "related") for p in turn.unused if p.section and _mentions(p, subject)]
            return out
        out += [self._section_followup(p, "next_section") for p in neighbours if p.section]
        if scenario == SCENE_ANSWER:
            out += [self._section_followup(p, "related") for p in turn.unused if p.section]
        if turn.cited and turn.cited[0].section:
            out.append(Followup(f"能再详细讲讲{turn.cited[0].section}吗？", "deeper"))
        return out

    @staticmethod
    def _section_followup(passage: Passage, kind: str) -> Followup:
        title = clean_title(passage.source) if passage.source else ""
        doc_kind = document_kind(title, Path(passage.source).suffix)[0] if title else "general"
        question, template = _heading_question(passage.section, title or passage.section, doc_kind)
        if template == "section_what" and "的" in passage.section:
            # "Agent 状态栏的构成" is no concept: "…的构成是什么？" reads better than "什么是…的构成？".
            question = f"{passage.section}是什么？"
        return Followup(question, kind)

    @staticmethod
    def _subject(turn: Turn) -> str:
        """What a refused question is about: the first word of its search query ("Manus", "推理加速")."""
        filters = parse_query(turn.rewritten or turn.question)
        for word in (filters.topic or " ".join(filters.name_keywords)).split():
            word = re.sub(r"^(?:请问|请|什么是|如何|怎么)|(?:是什么|是多少|有哪些|是怎样的|是如何.*|怎么样|吗|呢)$", "", word)
            if len(word.strip()) >= 2:
                return word.strip()[:20]
        return ""

    def _file_templates(self, turn: Turn) -> list[Followup]:
        """Questions within the files found: by what kind of document each is, then by the
        first file's matched section and outline (read here, nothing is sent anywhere)."""
        primary: list[Followup] = []
        extra: list[Followup] = []
        for rank, item in enumerate(turn.files[:2]):
            name = str(item.get("name", ""))
            file_id = str(item.get("file_id", ""))
            title = clean_title(name)
            _, _, questions = document_kind(title, str(item.get("ext") or Path(name).suffix))
            scope = {"doc_ids": [file_id], "label": name}

            def within(text: str) -> Followup:
                return Followup(text, "ask_file", "文件内", SCOPE_SET, scope)

            asks = [template.format(t=title) for template in questions]
            if rank:
                primary.append(within(asks[0]))
                continue
            headings = [str(s.get("heading", "")) for s in item.get("snippets") or []]
            headings += self._sections(self._outline(file_id), {""}, 3)
            sections = []
            for heading in headings:
                passage = Passage(file_id, name, heading)
                if passage.section and passage.section not in (p.section for p in sections):
                    sections.append(passage)
            section_asks = [self._section_followup(p, "ask_file").text for p in sections]
            primary += [within(text) for text in [asks[0], *section_asks[:1]]]
            extra += [within(text) for text in [*asks[1:], *section_asks[1:3], f"《{title}》主要讲了什么？"]]
        out = primary
        types = {_EXT_TYPE.get(str(f.get("ext", "")).lower()) for f in turn.files} - {None}
        top_type = _EXT_TYPE.get(str(turn.files[0].get("ext", "")).lower()) if turn.files else None
        if top_type and len(types) > 1 and not turn.file_types:
            question = turn.question.rstrip("？?。. ")
            out.append(Followup(f"{question}，只要 {TYPE_LABEL[top_type]}", "refine_search", "找文件"))
        return [*out, *extra]

    # ---------------- model ----------------

    def prompt(self, turn: Turn, neighbours: list[Passage], n: int) -> str:
        scenario = turn.scenario
        lines = [PROMPT_HEAD.format(n=n)]
        scoped_file = self._file_name(turn.scope[0]) if scenario == SCENE_SCOPED else ""
        lines.append(SCENE_GUIDE[scenario].format(file=clean_title(scoped_file) if scoped_file else "当前文件"))
        lines.append(f"\n本轮问题：{turn.question}")
        if scenario in (SCENE_ANSWER, SCENE_SCOPED) and turn.answer:
            lines.append(f"回答摘要：{_summary(turn.answer)}")
        earlier = [q for q in turn.asked[:-1]][-5:]
        if earlier:
            lines.append("已经问过：" + "；".join(earlier))

        def passage_line(p: Passage, snippet: bool = False) -> str:
            where = p.section or p.heading.split(" > ")[-1].strip() or "（无小节标题）"
            text = f"- 《{clean_title(p.source)}》{where}"
            if snippet and p.text:
                text += "：" + re.sub(r"\s+", " ", p.text)[:60]
            return text

        if scenario == SCENE_FILES:
            lines.append("找到的文件：")
            for index, item in enumerate(sendable(turn.files)[:3], start=1):
                snippet = (item.get("snippets") or [{}])[0]
                section = clean_heading(str(snippet.get("heading", "")))
                text = re.sub(r"\s+", " ", str(snippet.get("text", "")))[:60]
                lines.append(f"{index}. 《{clean_title(str(item.get('name', '')))}》"
                             + (f" 匹配小节：{section}" if section else "") + (f" 片段：{text}" if text else ""))
        else:
            if turn.cited:
                lines.append("回答引用的资料：")
                lines += [passage_line(p) for p in turn.cited[:4]]
            if neighbours:
                lines.append("同一文件里相邻的小节：")
                lines += [passage_line(p) for p in neighbours]
            if turn.unused:
                lines.append("检索到但回答没用上的片段：")
                lines += [passage_line(p, snippet=scenario == SCENE_REFUSED) for p in turn.unused[:MAX_UNUSED]]
        return "\n".join(lines)

    def material(self, turn: Turn, neighbours: list[Passage]) -> set[str]:
        texts = [turn.question, turn.rewritten, _summary(turn.answer)]
        for p in (*turn.cited, *turn.unused, *neighbours):
            texts += [clean_title(p.source), p.heading, p.text[:200]]
        for item in sendable(turn.files)[:3]:
            texts.append(clean_title(str(item.get("name", ""))))
            for snippet in item.get("snippets") or []:
                texts += [str(snippet.get("heading", "")), str(snippet.get("text", ""))[:200]]
        grams: set[str] = set()
        for text in texts:
            grams |= _content_bigrams(text)
        return grams

    def _from_model(self, turn: Turn, neighbours: list[Passage], n: int):
        try:
            self.settings.require_api_key()
        except MissingApiKeyError:
            return [], "no_key", None
        prompt = self.prompt(turn, neighbours, n)
        reply = _Reply()
        future = self._executor.submit(self._call, prompt, reply)
        partial = False
        try:
            future.result(timeout=self.settings.followup_timeout_seconds)
            text = reply.text
        except FutureTimeout:
            # The questions that have fully arrived are kept; the call finishes on its own.
            text, partial = reply.complete_lines(), True
            if not text.strip():
                return [], "timeout", None
        except Exception as exc:  # noqa: BLE001 - any model trouble falls back to templates
            logger.warning("Follow-up model call failed: %r", exc)
            return [], "error", None
        material = self.material(turn, neighbours)
        evidence = " ".join([turn.answer, *(p.text for p in (*turn.cited, *turn.unused))])
        subject = self._subject(turn) if turn.scenario == SCENE_REFUSED else ""
        items = []
        for line in _model_lines(text):
            if not on_topic(line, material):
                logger.debug("Follow-up off the material dropped: %s", line)
                continue
            if turn.scenario == SCENE_REFUSED and not (
                normalize(subject) in normalize(line) if subject else keeps_topic(line, turn.question)
            ):
                logger.debug("Follow-up drifting from the refused question dropped: %s", line)
                continue
            item = self._classify(turn, line)
            item.doubtful = doubtful(line, evidence)
            items.append(item)
        if not items:
            return [], "timeout" if partial else "empty", None
        return items, "partial" if partial else "", None if partial else reply.usage

    def _call(self, prompt: str, reply: _Reply) -> None:
        model = self.chat_model()
        name = str(getattr(model, "model_name", "") or getattr(model, "model", "") or "")
        if name and chat_busy.left(name) > 0:
            # The model is overloaded right now: template follow-ups instead of adding a call.
            raise RuntimeError(f"{name} busy, follow-ups from templates")
        with self._stats_lock:
            self.stats["calls"] += 1
        for chunk in model.stream(prompt):
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                reply.add(content)
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                reply.usage = {"input_tokens": int(usage.get("input_tokens") or 0),
                               "output_tokens": int(usage.get("output_tokens") or 0)}
        with self._stats_lock:
            self.stats["completed"] += 1
            for key in ("input_tokens", "output_tokens"):
                self.stats[key] += (reply.usage or {}).get(key, 0)

    def _classify(self, turn: Turn, text: str) -> Followup:
        """A model question with its scope: about a found file it asks within that file."""
        scenario = turn.scenario
        if FIND_PREFIX.match(text):
            # Asking within one file turns "找一下…" into a question about that file: it leaves the file.
            leave = SCOPE_CLEAR if len(turn.scope) == 1 else SCOPE_KEEP
            return Followup(text, "refine_search" if scenario == SCENE_FILES else "find_files", "找文件", leave)
        if scenario == SCENE_FILES:
            item = self._mentioned_file(turn.files[:3], text)
            if item is not None:
                scope = {"doc_ids": [item.get("file_id")], "label": item.get("name", "")}
                return Followup(text, "ask_file", "文件内", SCOPE_SET, scope)
            return Followup(text, "ask")
        if scenario == SCENE_REFUSED:
            return Followup(text, "rephrase")
        return Followup(text, "ask")

    @staticmethod
    def _mentioned_file(files: list[dict[str, Any]], text: str) -> dict[str, Any] | None:
        quoted = re.findall(r"《([^》]{1,80})》", text)
        for name in quoted:
            key = title_key(name)
            for item in files:
                other = title_key(str(item.get("name", "")))
                if key and other and (key in other or other in key or similarity(key, other) >= 0.5):
                    return item
        if len(files) == 1 and not quoted:
            # "这份协议的付款条款是什么？" with one file found is about that file.
            return files[0] if similarity(text, str(files[0].get("name", ""))) > 0.1 else None
        return None

    # ---------------- small talk ----------------

    def _from_pool(self, turn: Turn, n: int) -> list[Followup]:
        if self.pool is None:
            return []
        try:
            cards = self.pool(n + len(turn.asked))
        except Exception as exc:  # noqa: BLE001 - suggestions are optional
            logger.warning("Home-page questions unavailable for follow-ups: %s", exc)
            return []
        items = [
            Followup(str(card["question"]), "pool", "找文件" if card.get("qtype") == "find" else "")
            for card in cards if card.get("question")
        ]
        return dedupe(items, turn.asked, n)
