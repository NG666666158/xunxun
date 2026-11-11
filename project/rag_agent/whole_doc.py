"""Reading a whole file for "总结一下《X》": notes per passage, cached, then one answer from all of them.

Search finds the passages closest to a question, which for "主要内容" means a few pages of a
long file. Here every parent passage is read instead, in order:

1. Map: consecutive passages go to the model in batches and come back as one note per
   passage (``P12: …``). Batches run side by side, up to ``summary_concurrency``.
2. Cache: notes are stored per passage, keyed by its text, in ``summaries/<doc_id>.json``.
   Asking again costs no reading; after an edit only the changed passages are read.
3. Reduce: the notes are the answer's evidence (with their page numbers), so the answer can
   cite every part of the file. Very long files get their notes compressed first.

A passage that could not be read is reported, never silently skipped; the next question
continues from the cache.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path, PureWindowsPath
import re
import threading
import time
from typing import Any, Callable, Iterable

import openai

from langchain_core.messages import HumanMessage, SystemMessage

from core.timings import bind_context
from services.model_clients import MODEL_SERVICE_ERRORS, describe_model_error, error_kind

logger = logging.getLogger(__name__)

# Bump when the prompts change what a note contains: old notes are then read again.
NOTES_VERSION = 1

NOTE_PROMPT = """你在通读一份文档，为后面的全文总结做笔记。下面是文档里连续的片段，每段以 [P编号] 开头。
为每个片段写一条笔记，一行一条，格式严格为：P编号: 笔记
要求：
1. 一到两句话，写清这段讲了什么；保留关键事实：数字、金额、日期、人名、机构、指标、结论。
2. 只写片段里有的内容，不加评论，不补充。
3. 目录、页眉页脚、签名栏、纯数字或识别乱码，没有实质内容的片段写：P编号: -
不要输出其他内容。"""

COMPRESS_PROMPT = """下面是一份长文档的笔记，每条以 [N编号] 开头，按文档顺序排列。把它们合并压缩，不超过 {limit} 条。
一行一条，格式严格为：N起始编号-N结束编号: 合并后的内容
要求：相邻、同主题的笔记合并；保留关键事实（数字、金额、日期、人名、机构、结论）；不补充笔记里没有的内容；编号必须覆盖全部笔记。
不要输出其他内容。"""

_NOTE_LINE = re.compile(r"^\s*\[?\s*[Pp]\s*(\d+)\s*(?:[｜|][^\]:：]*(?::[^\]:：]*)?)?\]?\s*[:：]\s*(.*\S)\s*$")
_COMPRESSED_LINE = re.compile(r"^\s*\[?\s*[Nn]\s*(\d+)\s*(?:[-–—~]\s*[Nn]?\s*(\d+))?\s*\]?\s*[:：]\s*(.*\S)\s*$")
_EMPTY_NOTE = re.compile(r"^[-—–\s.。无]*$")

_TITLE = re.compile(r"《([^》]{1,160})》")
SUMMARY_CUE = re.compile(
    r"总结|概括|概述|归纳|梳理|摘要|简介|介绍|主要(?:内容|讲|说|写|是)|(?:讲|说|写)(?:了|的)?(?:是)?(?:什么|啥)|大意|要点|核心内容|重点内容",
    re.I,
)
# Words that say nothing about which part of the file is meant. Anything else left over
# ("风险", "第3章", "Agent 的记忆机制") makes it a question about that part: search answers it.
_FILLER = re.compile(
    "|".join(sorted((
        "帮我", "帮忙", "麻烦", "给我", "能否", "可以", "能不能", "请问", "请", "一下", "下", "我想", "想", "知道", "了解", "告诉我",
        "总结", "概括", "概述", "归纳", "梳理", "摘要", "简介", "介绍", "说说", "讲讲", "谈谈", "看看", "简单", "简要",
        "这份", "这本", "这个", "这篇", "这张", "这", "那份", "那本", "那个", "那篇", "该", "此", "份", "本", "篇", "张",
        "文件", "文档", "资料", "书", "报告", "ppt", "演示文稿", "幻灯片", "表格", "工作簿", "表", "文章", "论文", "材料",
        "全文", "整份", "整个", "全部", "所有", "完整", "的", "里", "里面", "中", "主要", "内容", "讲", "说", "写", "了", "是",
        "什么", "啥", "大意", "要点", "核心", "重点", "关键", "有哪些", "哪些", "都", "呢", "吗", "吧", "一", "要",
        "把", "将", "对", "给", "刚才", "刚刚", "上面", "上述", "找到", "那",
    ), key=len, reverse=True)),
    re.I,
)
_ORDINAL = re.compile(r"第?\s*([1-9一二三四五六七八九两])\s*(?:份|个|篇|本|张)?")
_CN_ORDINAL = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CONFIRM = re.compile(r"^\s*(?:继续|开始|开始通读|通读|确定|确认|好|好的|行|可以|是|是的|要|读吧|读)[\s,，。.!！吧呀啊]*$")

# CNY per million input/output tokens (SiliconFlow CN; evaluation/answer_speed_v1/REPORT.md).
# Only for the estimate shown before reading a very long file.
MODEL_PRICES = {
    "deepseek-ai/DeepSeek-V4-Flash": (3.0, 9.0),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (0.7, 2.8),
    "Qwen/Qwen3-32B": (1.0, 4.0),
    "deepseek-ai/DeepSeek-V3": (2.0, 8.0),
    "Qwen/Qwen3-8B": (0.0, 0.0),
}
# A note runs about 15% of its passage (measured, see evaluation/doc_capabilities_v1/REPORT.md).
NOTE_OUTPUT_SHARE = 0.15


def estimate_tokens(text: str) -> int:
    cjk = len(re.findall(r"[㐀-鿿]", text))
    return cjk + math.ceil(max(0, len(text) - cjk) / 4)


def loose(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _strip_numbering(stem: str) -> str:
    return re.sub(r"^\s*\d+\s*[.、．\-_\s]\s*", "", stem)


def _name_stem(name: str) -> str:
    """A file name without its extension; "0.政策补贴汇总" has none, whatever follows its dot."""
    return re.sub(r"\.(?:pdf|docx?|xlsx?|pptx?|md|txt|csv)$", "", name.strip(), flags=re.I)


def wants_whole_file(question: str, names: Iterable[str] = (), folders: Iterable[str] = ()) -> bool:
    """The question asks what a file says as a whole, naming no part of it.

    folders: folder names of the candidate files; words taken from them only say which
    file is meant ("合金粉末那份答辩PPT"), not which part of it.
    """
    text = _TITLE.sub(" ", question)
    for name in names:
        text = re.sub(re.escape(name), " ", text, flags=re.I)
    if not SUMMARY_CUE.search(text):
        return False
    rest = re.sub(r"[\W_]+", " ", _FILLER.sub(" ", text)).split()
    places = [loose(folder) for folder in folders if folder]
    return all(len(word) >= 2 and any(loose(word) in place for place in places) for word in rest)


def titles_in(question: str) -> list[str]:
    return [title.strip() for title in _TITLE.findall(question) if title.strip()]


def match_files(name: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Registry rows a file name refers to, best tier only: exact, then without its number
    ("政策补贴汇总" for "0.政策补贴汇总.xlsx"), then containing it (at least 4 characters)."""
    want = loose(name)
    if not want:
        return []
    bare = loose(_strip_numbering(_name_stem(name)))
    exact, without_number, partial = [], [], []
    for row in rows:
        full, stem = loose(row["name"]), loose(_name_stem(row["name"]))
        if want in (full, stem):
            exact.append(row)
        elif bare == loose(_strip_numbering(_name_stem(row["name"]))):
            without_number.append(row)
        elif len(want) >= 4 and (want in stem or (len(stem) >= 4 and stem in want)):
            partial.append(row)
    return exact or without_number or partial


def files_named_in(question: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """The files a question names, and the names as written: 《…》 first, else a file's name
    (without its number) written out in the question."""
    titles = titles_in(question)
    if titles:
        found: list[dict[str, Any]] = []
        for title in titles:
            found.extend(row for row in match_files(title, rows) if row not in found)
        return found, titles
    text = loose(question)
    best: list[dict[str, Any]] = []
    best_name = ""
    for row in rows:
        stem = _strip_numbering(_name_stem(row["name"]))
        key = loose(stem)
        if len(key) >= 4 and key in text and len(key) >= len(loose(best_name)):
            if len(key) > len(loose(best_name)):
                best = []
            best.append(row)
            best_name = stem
    return best, [best_name] if best_name else []


def distinct_files(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies with identical content count once (the most recently modified stays)."""
    by_content: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: float(r.get("mtime") or 0.0), reverse=True):
        by_content.setdefault(str(row.get("sha256") or row["file_id"]), row)
    return list(by_content.values())


def _folder_words(path: str) -> list[str]:
    # Registered paths are Windows paths; this reading also takes "/" on any system.
    return [part for part in PureWindowsPath(path).parent.parts[-3:] if len(part) >= 2]


def narrow_by_folder(question: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same-named files: those whose folder the question mentions (4 characters in common)."""
    text = loose(_TITLE.sub(" ", question))
    kept = []
    for row in rows:
        folders = loose(" ".join(_folder_words(str(row.get("path") or ""))))
        if any(text[i : i + 4] in folders for i in range(max(0, len(text) - 3))):
            kept.append(row)
    return kept


def pick_choice(reply: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Which of the listed files a reply means: its number, or words of its folder."""
    match = _ORDINAL.fullmatch(re.sub(r"[\s。.!！,，吧呀啊的那]+$", "", reply.strip()))
    if match:
        raw = match.group(1)
        index = _CN_ORDINAL.get(raw) or int(raw)
        return candidates[index - 1] if index <= len(candidates) else None
    narrowed = narrow_by_folder(reply, candidates)
    return narrowed[0] if len(narrowed) == 1 else None


def folder_label(row: dict[str, Any]) -> str:
    """"…\\01 面向精密级对准\\01 申报书.pdf": enough of the path to tell same-named files apart."""
    path = Path(str(row.get("path") or row["name"]))
    return str(Path(*path.parts[-2:])) if len(path.parts) >= 2 else str(path)


# ---------------------------------------------------------------- the cache


class NoteStore:
    """Notes of one document: ``notes`` by passage text, ``compressed`` by the notes they came from."""

    def __init__(self, directory: str | os.PathLike[str]):
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def _path(self, doc_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{24,64}", doc_id):
            raise ValueError("bad document id")
        return self.directory / f"{doc_id}.json"

    def load(self, doc_id: str) -> dict[str, Any]:
        try:
            data = json.loads(self._path(doc_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"notes": {}, "compressed": {}}
        data.setdefault("notes", {})
        data.setdefault("compressed", {})
        return data

    def save(self, doc_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self._path(doc_id)
            temporary = target.with_suffix(f".{threading.get_ident()}.tmp")
            temporary.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            temporary.replace(target)

    def remove(self, doc_id: str) -> None:
        try:
            self._path(doc_id).unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def _key(text: str) -> str:
    return hashlib.sha1(f"{NOTES_VERSION}:{text}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- map: one note per passage


@dataclass
class Note:
    index: int
    locator: str
    heading: str
    text: str
    quote: str = ""


@dataclass
class ReadResult:
    notes: list[Note]
    read: int = 0  # passages read, or already in the cache
    cached: int = 0
    unread: list[int] = field(default_factory=list)  # passage indexes that could not be read
    error: str = ""  # why, for the reader
    error_kind: str = ""  # services.model_clients.ERROR_KINDS
    fatal: bool = False  # a bad key, no balance: nothing else will work either
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def plan_batches(pending: list[int], parents: list[dict[str, Any]], batch_tokens: int) -> list[list[int]]:
    """Consecutive passages up to ``batch_tokens``; a gap (a cached passage) starts a new batch."""
    batches: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index in pending:
        tokens = estimate_tokens(str(parents[index].get("text", "")))
        if current and (used + tokens > batch_tokens or index != current[-1] + 1):
            batches.append(current)
            current, used = [], 0
        current.append(index)
        used += tokens
    if current:
        batches.append(current)
    return batches


def _batch_text(batch: list[int], parents: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[P{index + 1}｜{parents[index].get('locator') or '-'}]\n{parents[index].get('text', '')}" for index in batch)


def parse_notes(text: str, batch: list[int]) -> dict[int, str]:
    """Notes by passage index. Passages the model passed over count as read, with nothing to
    note; a reply with no note line at all counts as nothing read."""
    allowed = set(batch)
    notes: dict[int, str] = {}
    for line in str(text).splitlines():
        match = _NOTE_LINE.match(line)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if index in allowed:
            body = match.group(2).strip()
            notes[index] = "" if _EMPTY_NOTE.fullmatch(body) else body
    return notes


def read_passages(
    doc_id: str,
    parents: list[dict[str, Any]],
    call_model: Callable[..., Any],
    store: NoteStore,
    *,
    batch_tokens: int = 4000,
    concurrency: int = 6,
    progress: Callable[[int, int], None] | None = None,
) -> ReadResult:
    """Notes for every passage, reading only those not in the cache."""
    started = time.perf_counter()
    data = store.load(doc_id)
    notes: dict[str, str] = data["notes"]
    keys = [_key(str(parent.get("text", ""))) for parent in parents]
    pending = [index for index, key in enumerate(keys) if key not in notes]
    result = ReadResult([], cached=len(parents) - len(pending))
    total = len(parents)
    done = result.cached
    if progress:
        progress(done, total)
    lock = threading.Lock()
    failed: list[list[int]] = []

    def run(batch: list[int]) -> tuple[list[int], dict[int, str] | None, Exception | None]:
        output = max(200, min(2000, 40 + 100 * len(batch)))
        try:
            response = call_model(
                [SystemMessage(content=NOTE_PROMPT), HumanMessage(content=_batch_text(batch, parents))],
                max_tokens=output,
            )
        except MODEL_SERVICE_ERRORS as exc:
            return batch, None, exc
        usage = getattr(response, "usage_metadata", None) or {}
        with lock:
            result.calls += 1
            result.input_tokens += int(usage.get("input_tokens", 0))
            result.output_tokens += int(usage.get("output_tokens", 0))
        parsed = parse_notes(str(response.content), batch)
        return batch, parsed if parsed else None, None

    def absorb(batch: list[int], parsed: dict[int, str]) -> None:
        nonlocal done
        for index in batch:
            notes[keys[index]] = parsed.get(index, "")
        done += len(batch)

    def note_error(exc: Exception) -> None:
        result.error = result.error or describe_model_error(exc)
        result.error_kind = result.error_kind or error_kind(exc)
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)) or (
            isinstance(exc, openai.APIStatusError) and exc.status_code == 402
        ):
            result.fatal = True

    batches = plan_batches(pending, parents, batch_tokens)
    last_save = time.monotonic()
    if batches:
        with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="whole-file") as pool:
            futures = {pool.submit(bind_context(run), batch): batch for batch in batches}
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                batch, parsed, error = future.result()
                if error is not None:
                    note_error(error)
                if parsed is None:
                    failed.append(batch)
                else:
                    absorb(batch, parsed)
                    if progress:
                        progress(done, total)
                    if time.monotonic() - last_save > 5:
                        store.save(doc_id, data)
                        last_save = time.monotonic()
                if result.fatal:
                    for other in futures:
                        other.cancel()
        # What failed gets one more try, one at a time: a burst may have hit a rate limit.
        unfinished = [b for b in batches if not all(keys[index] in notes for index in b)]
        if not result.fatal:
            for batch in unfinished:
                batch_, parsed, error = run(batch)
                if parsed is None:
                    if error is not None:
                        note_error(error)
                    continue
                absorb(batch_, parsed)
                if progress:
                    progress(done, total)
        # Stale notes (edited passages) go; what was read this time stays, even after a failure.
        live = set(keys)
        data["notes"] = {key: note for key, note in notes.items() if key in live}
        store.save(doc_id, data)
        notes = data["notes"]
    result.unread = [index for index, key in enumerate(keys) if key not in notes]
    result.read = total - len(result.unread)
    result.notes = [
        Note(
            index, str(parents[index].get("locator") or parents[index].get("heading") or "未标注位置"),
            str(parents[index].get("heading") or ""), notes[keys[index]], str(parents[index].get("text", ""))[:200],
        )
        for index in range(total)
        if keys[index] in notes and notes[keys[index]]
    ]
    result.seconds = round(time.perf_counter() - started, 1)
    return result


# ---------------------------------------------------------------- very long files: fewer notes


def merge_locators(locators: list[str]) -> str:
    """One locator for several neighbours: page and slide numbers merge into a range."""
    spans: list[tuple[str, int, int]] = []
    for locator in locators:
        match = re.fullmatch(r"(page|slide):(\d+)(?:-(\d+))?", locator)
        if not match:
            sheet = re.fullmatch(r"(sheet:[^;]+;rows:)(\d+)-(\d+)", locator)
            if not sheet:
                return locators[0]
            spans.append((sheet.group(1), int(sheet.group(2)), int(sheet.group(3))))
            continue
        spans.append((f"{match.group(1)}:", int(match.group(2)), int(match.group(3) or match.group(2))))
    if len({span[0] for span in spans}) != 1:
        return locators[0]
    head = spans[0][0]
    low, high = min(span[1] for span in spans), max(span[2] for span in spans)
    if head.startswith("sheet:"):
        return f"{head}{low}-{high}"
    return f"{head}{low}" if low == high else f"{head}{low}-{high}"


def compress_notes(
    doc_id: str,
    notes: list[Note],
    call_model: Callable[..., Any],
    store: NoteStore,
    result: ReadResult,
    *,
    limit_tokens: int = 12000,
    concurrency: int = 6,
) -> list[Note]:
    """Merge neighbouring notes until they fit ``limit_tokens``; groups are cached like notes."""
    data = store.load(doc_id)
    compressed: dict[str, list[list[Any]]] = data["compressed"]
    current = notes
    lock = threading.Lock()
    for _ in range(3):
        total = sum(estimate_tokens(note.text) for note in current)
        if total <= limit_tokens:
            break
        group_tokens = max(2000, limit_tokens // 2)
        groups: list[list[Note]] = []
        group: list[Note] = []
        used = 0
        for note in current:
            tokens = estimate_tokens(note.text)
            if group and used + tokens > group_tokens:
                groups.append(group)
                group, used = [], 0
            group.append(note)
            used += tokens
        if group:
            groups.append(group)
        keep = max(3, int(len(current) * limit_tokens / total * 0.8 / len(groups)))

        def run(members: list[Note]) -> tuple[list[Note], list[list[Any]] | None]:
            body = "\n".join(f"[N{n}] {note.text}" for n, note in enumerate(members, start=1))
            key = _key(f"compress:{keep}:{body}")
            if key in compressed:
                return members, compressed[key]
            response = call_model(
                [SystemMessage(content=COMPRESS_PROMPT.format(limit=keep)), HumanMessage(content=body)],
                max_tokens=min(2500, 80 * keep + 200),
            )
            usage = getattr(response, "usage_metadata", None) or {}
            with lock:
                result.calls += 1
                result.input_tokens += int(usage.get("input_tokens", 0))
                result.output_tokens += int(usage.get("output_tokens", 0))
            lines: list[list[Any]] = []
            for line in str(response.content).splitlines():
                match = _COMPRESSED_LINE.match(line)
                if match:
                    lines.append([int(match.group(1)), int(match.group(2) or match.group(1)), match.group(3).strip()])
            if lines:
                compressed[key] = lines
            return members, lines or None

        merged: list[Note] = []
        with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="whole-file") as pool:
            for members, lines in pool.map(bind_context(run), groups):
                if not lines:
                    merged.extend(members)  # keep the group's notes as they are
                    continue
                for first, last, text in lines:
                    part = members[max(0, first - 1) : max(first, last)] or members[:1]
                    merged.append(Note(
                        part[0].index, merge_locators([n.locator for n in part]), part[0].heading, text, part[0].quote,
                    ))
        if len(merged) >= len(current):
            break
        current = merged
    store.save(doc_id, data)
    return current


# ---------------------------------------------------------------- saying what was covered


def _number_list(numbers: list[int], limit: int = 12) -> str:
    runs: list[tuple[int, int]] = []
    for number in sorted(set(numbers)):
        if runs and number == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], number)
        else:
            runs.append((number, number))
    parts = [f"{a}" if a == b else f"{a}–{b}" for a, b in runs[:limit]]
    return "、".join(parts) + (" 等" if len(runs) > limit else "")


def _units_of(locators: Iterable[str], prefix: str) -> set[int]:
    found: set[int] = set()
    for locator in locators:
        match = re.fullmatch(rf"{prefix}:(\d+)(?:-(\d+))?", locator or "")
        if match:
            found.update(range(int(match.group(1)), int(match.group(2) or match.group(1)) + 1))
    return found


def describe_coverage(
    parents: list[dict[str, Any]], result: ReadResult, metadata: dict[str, Any]
) -> dict[str, Any]:
    """What was read, in the file's own units; ``text`` goes under the answer."""
    locators = [str(parent.get("locator") or "") for parent in parents]
    unread_locators = [locators[index] for index in result.unread]
    total_parents = len(parents)
    coverage: dict[str, Any] = {
        "passages": total_parents, "read": result.read, "unread": len(result.unread), "cached": result.cached,
        "calls": result.calls, "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "seconds": result.seconds,
    }
    pages = _units_of(locators, "page")
    slides = _units_of(locators, "slide")
    sheets = list(dict.fromkeys(m.group(1).strip() for m in (re.match(r"sheet:([^;]+)", loc) for loc in locators) if m))
    if pages:
        total = int(metadata.get("pages") or max(pages))
        empty = sorted(set(range(1, total + 1)) - pages)
        unit, count, missing = "页", total, sorted(_units_of(unread_locators, "page"))
        whole = f"已通读全文：共 {total} 页"
        whole += f"，其中第 {_number_list(empty)} 页没有可读文字。" if empty else "。"
    elif slides:
        total = int(metadata.get("slides") or max(slides))
        unit, count, missing = "张", total, sorted(_units_of(unread_locators, "slide"))
        whole = f"已通读全部 {total} 张幻灯片"
        pictures = int(metadata.get("pictures") or 0)
        whole += f"（{pictures} 张图片里的文字没有读取）。" if pictures else "。"
    elif sheets:
        unit, count, missing = "个工作表", len(sheets), []
        whole = f"已通读全部 {len(sheets)} 个工作表。"
    else:
        unit, count, missing = "段", total_parents, []
        whole = f"已通读全文（{total_parents} 段）。"
    coverage.update(unit=unit, total=count)
    if not result.unread:
        coverage["text"] = whole
    else:
        where = f"第 {_number_list(missing)} 页" if missing and unit == "页" else (
            f"第 {_number_list(missing)} 张" if missing and unit == "张" else f"{len(result.unread)} 段"
        )
        reason = f"（{result.error.rstrip('。')}）" if result.error else ""
        coverage["text"] = f"这次没有读完：{where}没读到{reason}。再问一次会接着读。"
    return coverage


def estimate_reading(parents: list[dict[str, Any]], store: NoteStore, doc_id: str, model: str, *, concurrency: int = 6) -> dict[str, Any]:
    """Tokens still to read, and roughly how long and how much that takes (before asking)."""
    notes = store.load(doc_id)["notes"]
    unread = [parent for parent in parents if _key(str(parent.get("text", ""))) not in notes]
    tokens = sum(estimate_tokens(str(parent.get("text", ""))) for parent in unread)
    characters = sum(len(str(parent.get("text", ""))) for parent in unread)
    calls = math.ceil(tokens / 4000)
    # About 10 s per batch at this size, batches side by side (REPORT.md).
    seconds = max(10, 10 * calls / max(1, concurrency) + 20)
    price = MODEL_PRICES.get(model)
    yuan = None
    if price is not None:
        yuan = (tokens * price[0] + tokens * NOTE_OUTPUT_SHARE * price[1]) / 1e6
    return {"tokens": tokens, "characters": characters, "seconds": round(seconds), "yuan": yuan}
