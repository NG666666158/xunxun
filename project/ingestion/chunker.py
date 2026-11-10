"""Heading-aware semantic grouping followed by overlapping child windows."""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import re

from config import Settings, get_settings
from .models import ChunkDraft, ParentChunk, ParsedDocument


# Cut points, best first: paragraph break > sentence end > semicolon > line break.
# A "." only ends a sentence before whitespace, so 3.14 and file.py stay whole.
_BOUNDARIES = (
    re.compile(r"\n[ \t]*\n\s*"),
    re.compile(r"(?:[。！？!?…]+|\.(?=\s))[”’」』）)】》\"']*"),
    re.compile(r"[；;]"),
    re.compile(r"\n"),
)


_PARAGRAPH = re.compile(r"(?:[^\n]|\n(?!\n))+")
_QUESTION_ITEM = re.compile(r"\s*(?:Q|问题?|题)\s*\d{1,3}\s*[：:.、]")
# "1. …", "1、…", "1）…", "（1）…", "第1题 …": the number is group 1, 2 or 3. "3.14" is not one.
_NUMBERED_ITEM = re.compile(
    r"\s*(?:第\s*(\d{1,3})\s*题\s*[：:.、]?|[（(](\d{1,3})[)）]|(\d{1,3})(?:[.．](?!\d)|[、)）]))[ \t]*\S"
)
_ITEM_LINE = re.compile(rf"(?:{_QUESTION_ITEM.pattern}|{_NUMBERED_ITEM.pattern})")
# A numbered list counts as a question list only when its items are short and dense:
# "1. 员工出差……(three paragraphs)" in a policy stays windowed as prose.
NUMBERED_MIN_ITEMS = 4
NUMBERED_MAX_AVERAGE = 120

# Bump when chunk boundaries change; together with the parser version it decides
# which indexed files (every format) the library re-indexes at start-up.
CHUNKER_VERSION = 4


def context_prefix(source: str, heading: str) -> str:
    """"【大模型面试题200问 · 第3 章 LLM 的内部机制】" for a child of that section.

    Children carry it in their text, so both the embedding and BM25 see which document
    and chapter a passage belongs to; the last two heading levels keep it short.
    """
    title = re.sub(r"^[【\[].*?[】\]]", "", source.rsplit(".", 1)[0]).strip() or source
    path = [part.strip() for part in heading.split(">") if part.strip()]
    section = " > ".join(path[-2:]) if path else ""
    if not section or section == title or section.startswith("第 ") and section.endswith(" 页"):
        return f"【{title}】\n"
    return f"【{title} · {section}】\n"


def _is_lead_in(text: str, end: int) -> bool:
    """A short paragraph without a sentence end ("max_new_tokens", "实验1-2 ★：…") right before ``end``."""
    start = text.rfind("\n\n", 0, end) + 1
    paragraph = text[start:end].strip()
    return 0 < len(paragraph) <= 30 and not _BOUNDARIES[1].search(paragraph[-2:] + " ")


def _page_locator(page_map: list[list[int]] | None, start: int, end: int) -> str:
    """``page:N`` / ``page:N-M`` for the characters ``[start, end)`` of a unit."""
    if not page_map:
        return ""
    offsets = [offset for offset, _ in page_map]
    low = max(0, bisect_right(offsets, start) - 1)
    high = max(low, bisect_right(offsets, max(start, end - 1)) - 1)
    # Pages are not always monotonic (a caption kept after the paragraph it interrupted).
    pages = [page for _, page in page_map[low : high + 1]]
    first, last = min(pages), max(pages)
    return f"page:{first}" if first == last else f"page:{first}-{last}"


class HeadingAwareChunker:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def split(self, document: ParsedDocument) -> list[ChunkDraft]:
        """Return retrieval children for callers that use the legacy API."""
        _, children = self.split_parent_child(document)
        return children

    def split_parent_child(
        self,
        document: ParsedDocument,
    ) -> tuple[list[ParentChunk], list[ChunkDraft]]:
        """Create complete parent passages and overlapping retrieval children.

        Units that carry a ``page_map`` (PDF sections spanning several pages) give
        every parent and child the exact page range it was cut from.
        """
        parents: list[ParentChunk] = []
        chunks: list[ChunkDraft] = []
        parent_number = 0
        for unit in document.units:
            page_map = (unit.metadata or {}).get("page_map")
            for parent_start, parent_end in self._spans(unit.text, self.settings.parent_chunk_size, 0):
                parent_text = unit.text[parent_start:parent_end]
                parent_id = f"{document.doc_id}-p{parent_number}"
                parent_number += 1
                parents.append(
                    ParentChunk(
                        parent_id=parent_id,
                        doc_id=document.doc_id,
                        source=document.source,
                        heading=unit.heading[:1000],
                        locator=(_page_locator(page_map, parent_start, parent_end) or unit.locator)[:250],
                        text=parent_text,
                    )
                )
                child_spans = self._item_spans(parent_text, self.settings.child_chunk_size) or self._spans(
                    parent_text,
                    self.settings.child_chunk_size,
                    self.settings.child_chunk_overlap,
                )
                prefix = context_prefix(document.source, unit.heading) if getattr(
                    self.settings, "chunk_context_prefix", False
                ) else ""
                for child_number, (child_start, child_end) in enumerate(child_spans):
                    child_text = prefix + parent_text[child_start:child_end]
                    locator = _page_locator(page_map, parent_start + child_start, parent_start + child_end)
                    raw_id = f"{parent_id}:{child_number}:{child_text}".encode("utf-8")
                    chunk_id = hashlib.sha1(raw_id).hexdigest()[:32]
                    chunks.append(
                        ChunkDraft(
                            id=chunk_id,
                            doc_id=document.doc_id,
                            parent_id=parent_id,
                            source=document.source,
                            heading=unit.heading[:1000],
                            locator=(locator or unit.locator)[:250],
                            text=child_text,
                        )
                    )
        return parents, chunks

    @staticmethod
    def _item_paragraphs(text: str) -> list[tuple[int, int]]:
        """Paragraphs, with a line that starts an item ("2. …" after "1. …") starting a new one."""
        result = []
        for match in _PARAGRAPH.finditer(text):
            begin = match.start()
            for line in re.finditer(r"\n", match.group()):
                cut = match.start() + line.end()
                if _ITEM_LINE.match(text, cut) and text[begin:cut].strip():
                    result.append((begin, cut - 1))
                    begin = cut
            result.append((begin, match.end()))
        return result

    @staticmethod
    def _dense_numbered_list(text: str, items: list[tuple[int, int]]) -> bool:
        """Short items numbered mostly in sequence (1, 2, 3 …; a new section may restart)."""
        if len(items) < NUMBERED_MIN_ITEMS:
            return False
        numbers = []
        for begin, end in items:
            match = _NUMBERED_ITEM.match(text, begin)
            numbers.append(int(next(group for group in match.groups() if group)))
        in_sequence = sum(b == a + 1 or b == 1 for a, b in zip(numbers, numbers[1:]))
        average = sum(end - begin for begin, end in items) / len(items)
        return in_sequence >= 0.7 * (len(numbers) - 1) and average <= NUMBERED_MAX_AVERAGE

    @classmethod
    def _item_spans(cls, text: str, size: int) -> list[tuple[int, int]] | None:
        """One child per question in a question list, or None for ordinary text.

        Ten questions in one window dilute each other: a search for one topic matches the
        window only weakly. "Q12：…" items always count; plainly numbered items ("1. …",
        "（1）…", "第1题") only in a dense run of short items. Paragraphs between items (a
        chapter title) join the next one.
        """
        paragraphs = cls._item_paragraphs(text)
        if not paragraphs:
            return None
        items = [p for p in paragraphs if _QUESTION_ITEM.match(text, p[0])]
        pattern = _QUESTION_ITEM
        if len(items) < 3:
            items = [p for p in paragraphs if _NUMBERED_ITEM.match(text, p[0])]
            pattern = _NUMBERED_ITEM
            if not cls._dense_numbered_list(text, items):
                return None
        if len(items) < 3 or len(items) < 0.6 * len(paragraphs):
            return None
        spans: list[tuple[int, int]] = []
        start = paragraphs[0][0]
        for begin, end in paragraphs:
            if pattern.match(text, begin):
                if end - start <= size:
                    spans.append((start, end))
                else:  # an unusually long question: fall back to windows inside it
                    spans.extend((start + s, start + e) for s, e in cls._spans(text[start:end], size, 0))
                start = end
        trailing = text[start:].strip()
        if trailing and spans:
            spans[-1] = (spans[-1][0], len(text.rstrip()))
        return [(s + len(text[s:e]) - len(text[s:e].lstrip()), e) for s, e in spans]

    @classmethod
    def _windows(cls, text: str, size: int, overlap: int) -> list[str]:
        return [text[start:end] for start, end in cls._spans(text, size, overlap)]

    @staticmethod
    def _cut(text: str, low: int, high: int) -> int | None:
        """The latest end offset in ``(low, high]`` at the best available boundary."""
        for rank, pattern in enumerate(_BOUNDARIES):
            best = None
            for match in pattern.finditer(text, low, min(len(text), high + 1)):
                if match.end() > high:
                    continue
                if rank == 0 and _is_lead_in(text, match.start()):
                    continue  # keep a term or lead-in line with the paragraph it introduces
                best = match.end()
            if best is not None and best > low:
                return best
        return None

    @staticmethod
    def _first_boundary(text: str, low: int, high: int) -> int | None:
        """The earliest sentence/paragraph/clause end in ``[low, high)``: where an overlap may begin."""
        found = [
            match.end()
            for pattern in _BOUNDARIES
            for match in pattern.finditer(text, low, high)
            if low <= match.end() < high
        ]
        return min(found) if found else None

    @classmethod
    def _spans(cls, text: str, size: int, overlap: int) -> list[tuple[int, int]]:
        """Stripped ``(start, end)`` windows of at most ``size`` characters.

        Windows end at the best boundary in their second half. The overlap of the
        next window starts at a sentence boundary too, or is skipped when the
        overlap region has none; only a window that had to be hard-cut (no
        boundary at all) overlaps mid-sentence.
        """

        def stripped(start: int, end: int) -> tuple[int, int]:
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            return start, end

        if len(text) <= size:
            start, end = stripped(0, len(text))
            return [(start, end)] if end > start else []
        result: list[tuple[int, int]] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            clean = True
            if end < len(text):
                cut = cls._cut(text, start + size // 2, end)
                if cut is not None:
                    end = cut
                else:
                    clean = False
            span = stripped(start, end)
            if span[1] > span[0]:
                result.append(span)
            if end >= len(text):
                break
            next_start = end - overlap
            if overlap and clean:
                boundary = cls._first_boundary(text, max(start + 1, end - overlap), end)
                next_start = boundary if boundary is not None else end
            start = next_start if next_start > start else end
        return result
