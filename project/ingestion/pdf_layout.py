"""Layout-aware PDF parsing.

``page.get_text("text")`` keeps every hard line break of the typeset page and mixes
running headers, page numbers and figure labels into the body. This module rebuilds
the reading text from line geometry instead:

1. pieces of text on the same baseline are merged into rows (a line the PDF split
   into several blocks, table cells, diagram labels);
2. rows in the top/bottom 8% of the page that repeat across pages, or are bare page
   numbers, are dropped as running headers and footers;
3. rows are classified (body line, heading, code, table, caption, figure label) and
   body lines are joined into paragraphs, across text blocks, figures and pages;
4. paragraphs are grouped into sections at the bookmarks (outline); without an
   outline at headings recognised by font size, weight and numbering; and as a last
   resort per page.

Every section records ``page_map`` -- ``[character offset, page]`` pairs -- so chunks
cut from a section that spans many pages can still cite the exact pages.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any, Callable

import pymupdf

from .figures import Figure, Reader, otsl_rows, read_all, table_numbers, table_text
from .models import DocumentUnit


# Bump whenever the extracted text changes, so files parsed by an older version are
# re-indexed by the library's start-up reconcile.
PDF_PARSER_VERSION = 8
# Scanned pages are rendered at this zoom for recognition (2.0 = 144 dpi). 1.5 detects text a
# quarter faster but merges English words ("Scanned2026"), so it stays at 2.0.
PAGE_ZOOM = 2.0

EDGE_RATIO = 0.08     # top/bottom band searched for running headers and footers
REPEAT_RATIO = 0.30   # an edge row on more than this share of pages is a header/footer

_CJK = re.compile(r"[⺀-鿿　-〿豈-﫿＀-￯]")
_CLOSERS = "”’」』）)】》〉\"'"
_SENTENCE_END = "。！？!?….．"
_SOFT_END = "：:；;"
_PAGE_NUMBER = re.compile(
    r"^(?:[-–—·•|｜]?\s*\d{1,4}\s*[-–—·•|｜]?|(?=[ivxlcdm]+$)m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})|第\s*\d{1,4}\s*页(?:\s*/?\s*共\s*\d+\s*页)?"
    r"|(?:page|p\.)\s*\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?|\d{1,4}\s*/\s*\d{1,4})$",
    re.IGNORECASE,
)
# A table cell holding numbers: "82.3", "80.6/80.1", "86.0 91.7", "392k", "±0.4", "-" placeholders.
_NUMBER_CELL = re.compile(r"^(?:[-–—−+±]?\d[\d.,]*[%kKMB]?|[-–—−]+)(?:[\s/]+(?:[-–—−+±]?\d[\d.,]*[%kKMB]?|[-–—−]+))*$")
_CAPTION = re.compile(r"^(?:图|表|Figure|Fig\.|Table)\s*[0-9一二三四五六七八九十]+(?:\s*[-–.．]\s*\d+)*", re.IGNORECASE)
_LIST_ITEM = re.compile(
    r"^(?:[•●○◦▪■□◆◇►▶‣∙·✓✔※*]|[-–—]\s|\(?\d{1,2}[.)）、](?!\d)|（\d{1,2}）|[①-⑳]|[a-zA-Z][.)]\s)"
)
_CHAPTER = re.compile(r"^第\s*[0-9一二三四五六七八九十百零〇两]+\s*[章节篇部分讲回卷]")
_NUMBERED = re.compile(r"^\d{1,2}(?:\.\d{1,2}){1,3}\.?(?:\s|　|(?=[^\d.\s]))")
_QUESTION = re.compile(r"^Q\d{1,3}\s*[：:.、]")
_CODEISH = re.compile(
    r"^(?:def |class |import |from \S+ import|return\b|print\(|if .*:$|for .*:$|while .*:$|elif |else:|try:|except\b"
    r"|[A-Za-z_][\w.\[\]'\"]*\s*[+\-*/]?=\s*\S|[{}()\[\]]+[;,]?$|#include|//|/\*|\$ |>>> )"
)
# Formulas: math alphanumerics (𝑎, 𝜋) and operators, or letters set in a math font.
# Arrows, middle dots and × are common in running text and do not count.
_MATH_SYMBOL = re.compile(r"[𝐀-𝟿∑∏∫√∂∇]")
_MATH_FONT = re.compile(r"math|cmmi|cmsy|cmex|stix", re.IGNORECASE)
_NOT_MATH = re.compile(r"[→←↔⇒·×\s]")
# What a transcribed formula must contain to be kept (a lone "\leftarrow" is not one).
_LATEX_MATH = re.compile(r"=|\\(?:frac|sqrt|sum|prod|int|log|ln|exp|max|min|approx|leq|geq|mid)\b|[_^]")
_FOLD = re.compile(r"[ﬀ-ﬆ\U0001d400-\U0001d7ff]")
_JUNK = re.compile(r"[-�\x01-\x08\x0b\x0c\x0e-\x1f]")
_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi", re.IGNORECASE)
_MONO_FONT = re.compile(r"mono|courier|consol|menlo|code", re.IGNORECASE)


# ----------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------


@dataclass
class _Piece:
    """One cell of a row: a pymupdf line, or several merged ones."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    font: str
    bold: bool
    mono: bool
    math: int = 0  # math symbols / math-font letters, counted before they were folded to plain text

    @property
    def chars(self) -> int:
        return len(self.text.strip())


@dataclass
class _Row:
    page: int
    cells: list[_Piece]
    toc: int | None = None
    tokens: str | None = None  # "English|and|CAP|…": a row drawn token by token on colored boxes
    col: int = 0  # 1 / 2: left / right column of a two-column page; 0: full width
    col_left: float | None = None
    col_right: float | None = None

    def __post_init__(self) -> None:
        main = max(self.cells, key=lambda cell: cell.chars)
        self.x0 = min(cell.x0 for cell in self.cells)
        self.x1 = max(cell.x1 for cell in self.cells)
        self.y0 = min(cell.y0 for cell in self.cells)
        self.y1 = max(cell.y1 for cell in self.cells)
        self.size = main.size
        self.font = main.font
        total = sum(cell.chars for cell in self.cells) or 1
        self.bold = sum(cell.chars for cell in self.cells if cell.bold) * 2 > total
        self.mono = sum(cell.chars for cell in self.cells if cell.mono) * 2 > total
        self.math = sum(cell.math for cell in self.cells)
        self.text = " | ".join(cell.text.strip() for cell in self.cells)

    @property
    def center(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class _Page:
    number: int
    width: float
    height: float
    rows: list[_Row]


def _join_text(left: str, right: str) -> str:
    """Join two pieces of one paragraph: no space next to CJK, mend hyphenated words."""
    if not left:
        return right
    if not right:
        return left
    a, b = left[-1], right[0]
    if a == "\xad":
        return left[:-1] + right
    if a in "-‐" and len(left) > 1 and left[-2].isalpha() and b.islower():
        word = re.search(r"[A-Za-z\-‐]+$", left).group()
        # "imple-" + "mentation" -> "implementation"; keep compounds like "state-of-the-art".
        return (left[:-1] if "-" not in word[:-1] and "‐" not in word[:-1] else left) + right
    if a in "-‐–—/" or a.isspace() or b.isspace():
        return left + right
    if _CJK.match(a) or _CJK.match(b):
        return left + right
    return left + " " + right


def _repair(text: str) -> str:
    """Fold ligatures (ﬁ) and math letters (𝑑) to plain text; drop private-use glyphs.

    Only those ranges are NFKC-normalised: the full-width Chinese punctuation must stay.
    """
    text = _FOLD.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), text)
    return _JUNK.sub("", text.replace("\x00", " "))


def _merge_cells(pieces: list[_Piece]) -> list[_Piece]:
    """Merge pieces of a row that are close together (one line split into blocks)."""
    pieces.sort(key=lambda piece: piece.x0)
    cells = [pieces[0]]
    for piece in pieces[1:]:
        cell = cells[-1]
        gap = piece.x0 - cell.x1
        size = max(cell.size, piece.size)
        bullet = len(cell.text.strip()) == 1 and not cell.text.strip().isalnum()
        if gap <= 1.0 * size or bullet:
            text = cell.text.rstrip()
            text = _join_text(text, piece.text.lstrip()) if gap > 0.15 * size else text + piece.text.lstrip()
            main = cell if cell.chars >= piece.chars else piece
            total = cell.chars + piece.chars or 1
            cells[-1] = _Piece(
                text=text,
                x0=cell.x0,
                y0=min(cell.y0, piece.y0),
                x1=max(cell.x1, piece.x1),
                y1=max(cell.y1, piece.y1),
                size=main.size,
                font=main.font,
                bold=(cell.chars * cell.bold + piece.chars * piece.bold) * 2 > total,
                mono=(cell.chars * cell.mono + piece.chars * piece.mono) * 2 > total,
                math=cell.math + piece.math,
            )
        else:
            cells.append(piece)
    return cells


def _page_pieces(page: pymupdf.Page) -> list[_Piece]:
    pieces: list[_Piece] = []
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            direction = line.get("dir", (1.0, 0.0))
            if direction[0] < 0.99:
                continue  # rotated or vertical text: watermarks, axis labels
            spans = [span for span in line.get("spans", []) if span.get("text")]
            weights = [len(span["text"].strip()) for span in spans]
            total = sum(weights)
            if not total:
                continue
            main = spans[weights.index(max(weights))]
            is_mono = [bool(s["flags"] & 8 or _MONO_FONT.search(s["font"])) for s in spans]
            # Prose with inline code ("XML 的作用：<file_operation>…") has CJK outside the mono spans.
            mono = sum(w for m, w in zip(is_mono, weights) if m) * 2 > total and not any(
                _CJK.search(s["text"]) for s, m in zip(spans, is_mono) if not m
            )
            bold = sum(w for s, w in zip(spans, weights) if s["flags"] & 16 or _BOLD_FONT.search(s["font"])) * 2 > total
            raw = "".join(span["text"] for span in spans)
            math = len(_MATH_SYMBOL.findall(raw)) + sum(
                len(_NOT_MATH.sub("", s["text"])) for s in spans if _MATH_FONT.search(s["font"])
            )
            text = _repair(raw).rstrip()
            if not mono:
                text = re.sub(r"[ \t ]+", " ", text).strip()
            if not text.strip():
                continue
            x0, y0, x1, y1 = line["bbox"]
            if mono:
                # Leading blanks are positioned separately; indentation comes from x0 later.
                stripped = text.lstrip()
                if len(stripped) < len(text) and len(text):
                    x0 += (x1 - x0) * (len(text) - len(stripped)) / len(text)
                text = stripped
            pieces.append(_Piece(text, x0, y0, x1, y1, float(main["size"]), str(main["font"]), bold, mono, math))
    return pieces


def _find_gutter(pieces: list[_Piece], width: float) -> float | None:
    """The x of a two-column gutter, or None for a single-column page.

    A gutter is a vertical line near the middle that almost no text line crosses, with
    plenty of column-wide lines entirely on each side of it. Book pages fail this test:
    their body lines run across the middle.
    """
    text = [p for p in pieces if p.chars >= 3]
    if len(text) < 10:
        return None
    candidates = []
    margin = sorted(p.x0 for p in text)[len(text) // 20]
    for step in range(0, 81):
        x = width * (0.38 + step * 0.003)  # scan the middle 38%-62% of the page
        crossing = sum(1 for p in text if p.x0 < x - 1 and p.x1 > x + 1)
        # left-column lines start at the page margin; the middle column of a table does not
        left = sum(1 for p in text if p.x1 <= x + 1 and p.x1 - p.x0 >= 0.25 * width and p.x0 - margin <= 0.04 * width)
        right = sum(1 for p in text if p.x0 >= x - 1 and p.x1 - p.x0 >= 0.25 * width)
        # full-width prose (a book page with a table in it) outnumbering a column rules a gutter out
        wide = sum(1 for p in text if p.x0 < x - 1 and p.x1 > x + 1 and p.x1 - p.x0 >= 0.6 * width)
        if left >= 4 and right >= 4 and wide < min(left, right) and crossing <= 0.15 * len(text):
            candidates.append((crossing, x))
    if not candidates:
        return None
    fewest = min(crossing for crossing, _ in candidates)
    xs = [x for crossing, x in candidates if crossing == fewest]
    return (xs[0] + xs[-1]) / 2  # the middle of the empty strip, not its edge


def _baseline_groups(pieces: list[_Piece]) -> list[list[_Piece]]:
    pieces = sorted(pieces, key=lambda piece: ((piece.y0 + piece.y1) / 2, piece.x0))
    groups: list[list[_Piece]] = []
    top = bottom = 0.0
    for piece in pieces:
        if groups:
            overlap = min(bottom, piece.y1) - max(top, piece.y0)
            if overlap >= 0.5 * min(bottom - top, piece.y1 - piece.y0):
                # The bounds stay those of the row's first piece: a label centred across two
                # table rows ("Single" beside two result rows) must not chain them together.
                groups[-1].append(piece)
                continue
        groups.append([piece])
        top, bottom = piece.y0, piece.y1
    return groups


def _group_rows(pieces: list[_Piece], number: int, width: float | None = None) -> list[_Row]:
    """Rows of a page in reading order; two-column pages are read column by column.

    Lines that cross the gutter, and baselines with cells on both sides (a full-width
    table), are full-width rows. They cut the page into horizontal bands; within a band
    the left column is read before the right one.
    """
    gutter = _find_gutter(pieces, width) if width else None
    if gutter is None:
        return [_Row(number, _merge_cells(group)) for group in _baseline_groups(pieces)]

    def side(piece: _Piece) -> int:
        if piece.x1 <= gutter + 1:
            return 1
        if piece.x0 >= gutter - 1:
            return 2
        return 0

    bands: list[list] = [[[], []]]  # per band: left pieces, right pieces; full-width rows in between
    for group in _baseline_groups(pieces):
        sides = [side(piece) for piece in group]

        def cells(where: int) -> bool:  # table cells: several short pieces, not a prose line split by math
            members = [p for p, s in zip(group, sides) if s == where]
            return len(members) >= 2 and max(p.x1 - p.x0 for p in members) < 0.3 * width

        if 0 in sides or (cells(1) and cells(2)):
            bands.append(_Row(number, _merge_cells(group)))
            bands.append([[], []])
        else:
            for piece, where in zip(group, sides):
                bands[-1][where - 1].append(piece)
    columns = []
    for where in (1, 2):
        members = [p for band in bands if isinstance(band, list) for p in band[where - 1] if p.chars >= 3]
        if members:
            lefts = sorted(round(p.x0) for p in members)
            rights = sorted(p.x1 for p in members)
            columns.append((lefts[len(lefts) // 4], rights[min(len(rights) - 1, int(len(rights) * 0.95))]))
        else:
            columns.append((0.0, gutter) if where == 1 else (gutter, width))
    rows: list[_Row] = []
    for band in bands:
        if isinstance(band, _Row):
            rows.append(band)
            continue
        for where, members in ((1, band[0]), (2, band[1])):
            for group in _baseline_groups(members):
                row = _Row(number, _merge_cells(group))
                row.col, (row.col_left, row.col_right) = where, columns[where - 1]
                rows.append(row)
    return rows


# ----------------------------------------------------------------------------------
# document-wide layout statistics
# ----------------------------------------------------------------------------------


@dataclass
class _Layout:
    body_size: float
    body_fonts: set[str]
    margins: dict[int, tuple[float, float]]  # page parity -> (left, right)
    pitch: float                             # median baseline distance of body lines

    def left_right(self, page: int) -> tuple[float, float]:
        return self.margins.get(page % 2) or self.margins[-1]

    def is_body_size(self, size: float, tolerance: float = 0.12) -> bool:
        return abs(size - self.body_size) <= tolerance * self.body_size


def _measure(pages: list[_Page]) -> _Layout:
    # Header/footer bands are left out: page numbers would pull the margins around.
    inner = [(page, [row for row in page.rows if _edge_side(row, page) is None]) for page in pages]
    sizes: Counter[float] = Counter()
    for _, rows in inner:
        for row in rows:
            for cell in row.cells:
                sizes[round(cell.size * 2) / 2] += cell.chars
    body_size = sizes.most_common(1)[0][0] if sizes else 10.0

    def body_lines(rows: list[_Row]) -> list[_Row]:
        return [row for row in rows if len(row.cells) == 1 and abs(row.size - body_size) <= 0.12 * body_size]

    fonts: Counter[str] = Counter()
    rights: dict[int, list[float]] = defaultdict(list)
    for page, rows in inner:
        for row in body_lines(rows):
            fonts[row.font] += row.cells[0].chars
            if row.cells[0].chars >= 6:
                rights[page.number % 2].append(row.x1)
                rights[-1].append(row.x1)
    total = sum(fonts.values()) or 1
    body_fonts = {font for font, count in fonts.items() if count >= 0.03 * total}

    margins: dict[int, tuple[float, float]] = {}
    for parity in (-1, 0, 1):
        values = sorted(rights.get(parity, []))
        if len(values) < 20 and parity != -1:
            continue
        if not values:
            width = pages[0].width if pages else 595.0
            margins[parity] = (width * 0.1, width * 0.9)
            continue
        right = values[min(len(values) - 1, int(len(values) * 0.95))]
        tolerance = max(1.5 * body_size, 0.03 * right)
        lefts = Counter(
            round(row.x0)
            for page, rows in inner
            if parity == -1 or page.number % 2 == parity
            for row in body_lines(rows)
            if row.cells[0].chars >= 6 and row.x1 >= right - tolerance
        )
        left = float(lefts.most_common(1)[0][0]) if lefts else min((row.x0 for _, rows in inner for row in rows), default=0.0)
        margins[parity] = (left, right)

    pitches = []
    for _, rows in inner:
        lines = body_lines(rows)
        for prev, cur in zip(lines, lines[1:]):
            delta = cur.center - prev.center
            if 0.8 * body_size <= delta <= 2.5 * body_size:
                pitches.append(delta)
    pitch = statistics.median(pitches) if pitches else 1.45 * body_size
    return _Layout(body_size, body_fonts, margins, pitch)


# ----------------------------------------------------------------------------------
# running headers and footers
# ----------------------------------------------------------------------------------


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _norm_title(text: str) -> str:
    return re.sub(r"[\s·•.．、:：_\-–—|｜()（）\"“”'‘’《》]+", "", unicodedata.normalize("NFKC", text)).lower()


def _edge_side(row: _Row, page: _Page) -> str | None:
    if row.center < page.height * EDGE_RATIO:
        return "top"
    if row.center > page.height * (1 - EDGE_RATIO):
        return "bottom"
    return None


def _strip_running_text(pages: list[_Page], layout: _Layout, toc_titles: set[str]) -> int:
    """Remove page numbers and repeated header/footer rows; returns how many rows went."""
    every_page: dict[tuple[str, str], set[int]] = defaultdict(set)   # digits normalised
    end_numbers: dict[tuple[str, str], set[int]] = defaultdict(set)  # only a leading/trailing number normalised
    for page in pages:
        for row in page.rows:
            side = _edge_side(row, page)
            if side:
                compact = _compact(row.text)
                every_page[(side, re.sub(r"\d+", "#", compact))].add(page.number)
                end_numbers[(side, re.sub(r"^\d+|\d+$", "#", compact))].add(page.number)

    removed = 0
    count = len(pages)
    for page in pages:
        kept = []
        for row in page.rows:
            side = _edge_side(row, page)
            if side and _is_running(row, side, layout, toc_titles, every_page, end_numbers, count):
                removed += 1
                continue
            kept.append(row)
        page.rows = kept
    return removed


def _is_running(row, side, layout, toc_titles, every_page, end_numbers, count) -> bool:
    text = row.text.strip()
    if _PAGE_NUMBER.match(text):
        return True
    if row.size > layout.body_size * 1.1 or row.mono:
        return False  # headings near the top edge are content, not running headers
    compact = _compact(text)
    repeated = len(every_page[(side, re.sub(r"\d+", "#", compact))])
    if repeated >= 2 and repeated > REPEAT_RATIO * count:
        return True
    near_repeats = len(end_numbers[(side, re.sub(r"^\d+|\d+$", "#", compact))])
    if near_repeats >= 3 and len(compact) >= 2:
        return True
    # Section titles repeated as page headers ("1.1 现代Agent = ..." above the body).
    return side == "top" and _norm_title(re.sub(r"^\d+\s+|\s+\d+$", "", text)) in toc_titles


# ----------------------------------------------------------------------------------
# outline (bookmarks)
# ----------------------------------------------------------------------------------


@dataclass
class _Entry:
    level: int
    title: str
    page: int
    y: float | None


def _outline(pdf: pymupdf.Document) -> list[_Entry]:
    entries = []
    try:
        toc = pdf.get_toc(simple=False)
    except Exception:
        return []
    for item in toc:
        level, title, page = int(item[0]), str(item[1]).strip(), int(item[2])
        if not title or not 1 <= page <= pdf.page_count:
            continue
        y = None
        dest = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        point = dest.get("to")
        if point is not None:
            height = pdf[page - 1].rect.height
            # Outline destinations use PDF coordinates (origin bottom-left).
            flipped = height - float(point.y)
            if 0 < float(point.y) < height:
                y = flipped
        entries.append(_Entry(level, re.sub(r"\s+", " ", title), page, y))
    return entries


def _match_outline(pages: list[_Page], entries: list[_Entry], layout: _Layout) -> dict[int, tuple[int, int]]:
    """Find each bookmark's heading row: ``entry -> (page index, row index)``.

    Rows that match are tagged with the entry; entries whose title is not printed
    on the page fall back to the destination's y position, or the page top.
    """
    positions: dict[int, tuple[int, int]] = {}
    cursor: dict[int, int] = defaultdict(int)  # page index -> first row not yet used
    for index, entry in enumerate(entries):
        title = _norm_title(entry.title)
        found = None
        for page_index in (entry.page - 1, entry.page, entry.page - 2):
            if not 0 <= page_index < len(pages) or not title:
                continue
            rows = pages[page_index].rows
            candidates = []
            for row_index, row in enumerate(rows):
                if row.toc is not None or row.mono:
                    continue
                text = _norm_title(row.text)
                if not text:
                    continue
                if text == title or (text.startswith(title) and len(text) - len(title) <= 3):
                    candidates.append((row_index, 1))
                elif title.startswith(text) and len(text) >= min(len(title), 4):
                    candidates.append((row_index, len(text)))
                elif (
                    (row.size > layout.body_size * 1.05 or row.bold)
                    and len(title) >= 6
                    and SequenceMatcher(None, text, title).ratio() >= 0.85
                ):
                    candidates.append((row_index, 1))  # printed title reworded slightly
            if candidates:
                start = cursor[page_index]
                prominent = [c for c in candidates if rows[c[0]].size > layout.body_size * 1.05 or rows[c[0]].bold]
                ordered = [c for c in (prominent or candidates) if c[0] >= start] or (prominent or candidates)
                row_index, consumed = ordered[0]
                found = (page_index, row_index)
                rows[row_index].toc = index
                # A title wrapped onto following rows.
                if consumed > 1:
                    covered = _norm_title(rows[row_index].text)
                    follow = row_index + 1
                    while follow < len(rows) and len(covered) < len(title):
                        nxt = _norm_title(rows[follow].text)
                        if not nxt or not title.startswith(covered + nxt):
                            break
                        covered += nxt
                        rows[follow].toc = index
                        follow += 1
                cursor[page_index] = row_index + 1
                break
        if found is None and title:
            # Some outlines point a few pages early: accept an exact, prominent title ahead.
            for page_index in range(entry.page + 1, min(entry.page + 8, len(pages))):
                rows = pages[page_index].rows
                hit = next(
                    (
                        i
                        for i, row in enumerate(rows)
                        if row.toc is None
                        and _norm_title(row.text) == title
                        and (row.size > layout.body_size * 1.05 or row.bold)
                    ),
                    None,
                )
                if hit is not None:
                    rows[hit].toc = index
                    found = (page_index, hit)
                    cursor[page_index] = max(cursor[page_index], hit + 1)
                    break
        if found is None:
            page_index = entry.page - 1
            rows = pages[page_index].rows
            row_index = 0
            if entry.y is not None:
                row_index = next((i for i, row in enumerate(rows) if row.y1 >= entry.y - 2), len(rows))
            found = (page_index, row_index)
        positions[index] = found
    return positions


# ----------------------------------------------------------------------------------
# paragraph assembly
# ----------------------------------------------------------------------------------


@dataclass
class _Block:
    kind: str                       # para, heading, code, table, caption, note, figure, tokens
    rows: list[_Row]
    pieces: list[list[Any]] = field(default_factory=list)  # [text, page] for prose
    sealed: bool = False
    toc: int | None = None          # outline entry this heading belongs to
    level: int = 0                  # heading level (fallback headings)
    splits: list[int] = field(default_factory=list)  # outline entries starting at this block
    replacement: str | None = None  # a table re-read from its image (cloud mode)

    @property
    def last(self) -> _Row:
        return self.rows[-1]

    def text(self) -> str:
        return "".join(piece[0] for piece in self.pieces)

    def add_text(self, row: _Row, text: str) -> None:
        if self.pieces:
            joined = _join_text(self.pieces[-1][0], text)
            # _join_text may drop a trailing hyphen; rewrite the last piece accordingly.
            head = self.pieces[-1][0]
            keep = len(head) if joined.startswith(head) else len(head) - 1
            self.pieces[-1][0] = head[:keep]
            text = joined[keep:]
        self.pieces.append([text, row.page])
        self.rows.append(row)


def _ends_sentence(text: str) -> bool:
    text = text.rstrip().rstrip(_CLOSERS)
    return bool(text) and text[-1] in _SENTENCE_END + _SOFT_END


class _Assembler:
    def __init__(self, layout: _Layout):
        self.layout = layout
        self.blocks: list[_Block] = []
        self.interrupted = False
        self.force_break = False  # the next row must start a new block (a bookmark lands here)
        self.page_height = 842.0
        self.dropped_labels = 0

    # -- classification ------------------------------------------------------------

    def margins(self, row: _Row) -> tuple[float, float]:
        """Left/right edge of the text the row belongs to: its column, or the page body."""
        if row.col and row.col_left is not None and row.col_right is not None:
            return row.col_left, row.col_right
        return self.layout.left_right(row.page)

    def kind(self, row: _Row) -> str:
        layout = self.layout
        body = layout.body_size
        text = row.text
        if row.toc is not None:
            return "heading"
        if row.tokens is not None:
            return "tokens"
        if row.mono:
            return "code"
        left, right = self.margins(row)
        full = row.x1 >= right - max(1.5 * body, 0.03 * (right - left))
        if len(row.cells) > 1:
            bodyish = sum(
                cell.chars for cell in row.cells if layout.is_body_size(cell.size, 0.15) and cell.font in layout.body_fonts
            )
            if bodyish * 3 >= 2 * sum(cell.chars for cell in row.cells) or max(cell.chars for cell in row.cells) >= 30:
                return "table"  # a sentence-long cell is never a figure label
            return "label"
        if _CAPTION.match(text) and (row.size < body * 0.97 or row.font not in layout.body_fonts or not full):
            return "caption"
        if row.size >= body * 1.15 and len(text) <= 80:
            return "big"
        if (
            not _CJK.search(text)
            and _CODEISH.match(text)
            and row.x0 >= left + 1.5 * body
            and not full
        ):
            return "code"
        if (
            row.x1 - row.x0 < 0.4 * (right - left)
            and len(text) <= 25
            and not _LIST_ITEM.match(text)
            and not _ends_sentence(text)
            and (not layout.is_body_size(row.size) or row.font not in layout.body_fonts)
        ):
            return "label"
        if row.size <= body * 0.85 and row.center > 0.55 * self.page_height:
            return "note"  # footnotes: kept, but they do not break the paragraph above them
        return "line"

    # -- continuation rules ----------------------------------------------------------

    def _continues_para(self, block: _Block, row: _Row) -> bool:
        if block.kind != "para" or block.sealed:
            return False
        if _LIST_ITEM.match(row.text) or _QUESTION.match(row.text) or _CHAPTER.match(row.text):
            return False
        layout = self.layout
        body = layout.body_size
        prev = block.last
        lp, rp = self.margins(prev)
        lc, _ = self.margins(row)
        tolerance = max(1.5 * body, 0.03 * (rp - lp))
        indent_limit = prev.x0 - lp + 0.8 * body
        first = block.rows[0]
        if _LIST_ITEM.match(first.text):
            # Hanging indent: list items wrap under the text after the marker.
            indent_limit = max(indent_limit, first.x0 - self.margins(first)[0] + 3 * body)
        if row.x0 - lc > indent_limit:
            return False  # first-line indent: a new paragraph
        if abs(row.size - prev.size) > 0.15 * body:
            return False
        full = prev.x1 >= rp - tolerance
        text = block.text()
        if row.page == prev.page and row.col == prev.col and not self.interrupted:
            if not 0 < row.center - prev.center <= layout.pitch * 1.35:
                return False
            if full:
                return True
            # Indented blocks (quotes, boxes) end lines before the page margin.
            return (
                prev.x1 - prev.x0 >= 0.6 * (rp - lp)
                and row.x1 <= prev.x1 + tolerance
                and row.x0 >= lc + body
                and not _ends_sentence(text)
            )
        # Across a page or column break, or a figure: only an unfinished sentence carries on.
        return full and not _ends_sentence(text)

    # -- feeding rows ----------------------------------------------------------------

    def new_page(self, page: _Page) -> None:
        self.interrupted = True
        self.page_height = page.height

    def page_kinds(self, rows: list[_Row], tables: bool = True) -> list[str]:
        """Classify a page's rows; a lone short line is only a figure label among others.

        ``tables=False`` leaves numeric label grids as labels, for telling figures apart.
        """
        kinds = [self.kind(row) for row in rows]
        # Tables set smaller than the body (papers) look like figure labels: a run of
        # multi-cell rows, several of them mostly numbers, is a table.
        start = 0 if tables else len(rows)
        while start < len(rows):
            end = start
            while end < len(rows) and kinds[end] in ("label", "table") and len(rows[end].cells) > 1:
                end += 1
            numeric = 0
            for row in rows[start:end]:
                numbers = sum(1 for cell in row.cells if _NUMBER_CELL.match(cell.text.strip()))
                numeric += len(row.cells) >= 3 and numbers >= max(2, 0.4 * len(row.cells))
            if end - start >= 3 and numeric >= 2:
                kinds[start:end] = ["table"] * (end - start)
            start = max(end, start + 1)
        for index, kind in enumerate(kinds):
            if kind != "label" or len(rows[index].cells) > 1:
                continue
            neighbours = kinds[max(0, index - 1) : index] + kinds[index + 1 : index + 2]
            if "label" not in neighbours and rows[index].size >= 0.85 * self.layout.body_size:
                kinds[index] = "line"
        return kinds

    def feed(self, row: _Row, kind: str | None = None) -> None:
        kind = kind or self.kind(row)
        current = self.blocks[-1] if self.blocks and not self.force_break else None
        if self.force_break and kind == "label":
            kind = "line"
        self.force_break = False
        if kind == "label":
            if current is not None and self._continues_para(current, row):
                kind = "line"
            elif current is not None and current.kind == "table" and self._continues_table(current, row):
                kind = "table"
            else:
                self.dropped_labels += len(row.cells)
                self.interrupted = True
                return

        if kind == "heading":
            if current is not None and current.kind == "heading" and current.toc == row.toc and current.last.page == row.page:
                current.add_text(row, row.text)
            else:
                self._start("heading", row).toc = row.toc
            self.interrupted = False
            return
        if kind == "big":
            if (
                current is not None
                and current.kind == "para"
                and current.sealed
                and current.last.page == row.page
                and abs(current.last.size - row.size) < 0.5
                and 0 < row.center - current.last.center <= 1.8 * row.size
            ):
                current.add_text(row, row.text)  # a large title wrapped onto two rows
            else:
                self._start("para", row).sealed = True
            self.interrupted = False
            return
        if kind == "tokens":
            if current is not None and current.kind == "tokens" and current.last.page == row.page:
                current.pieces.append(["\n" + row.tokens, row.page])
                current.rows.append(row)
            else:
                self._start("tokens", row).pieces = [["〔按色块切分的示意〕\n" + row.tokens, row.page]]
            self.interrupted = True
            return
        if kind == "code":
            if current is not None and current.kind == "code" and self._continues_code(current, row):
                current.rows.append(row)
            else:
                self._start("code", row)
            self.interrupted = False
            return
        if kind == "table":
            if current is not None and current.kind == "table" and self._continues_table(current, row):
                current.rows.append(row)
            else:
                self._start("table", row)
            self.interrupted = False
            return
        if kind == "note":
            prev = current.last if current is not None else None
            if (
                current is not None
                and current.kind == "note"
                and prev.page == row.page
                and 0 < row.center - prev.center <= 1.8 * (prev.y1 - prev.y0)
                and not re.match(r"^(?:注\s*)?\d{1,3}\D", row.text)
            ):
                current.add_text(row, row.text)
            else:
                self._start("note", row)
            return
        if kind == "caption":
            self._start("caption", row)
            self.interrupted = True  # text after a caption may resume the paragraph before the figure
            return

        # Plain body line.
        if current is not None and current.kind == "caption" and self._continues_caption(current, row):
            current.add_text(row, row.text)
            return
        if current is not None and current.kind == "table" and self._continues_table(current, row):
            current.rows.append(row)
            return
        target = self._open_para(row) if current is not None else None
        if target is not None:
            target.add_text(row, row.text)
        else:
            self._start("para", row)
        self.interrupted = False

    def _open_para(self, row: _Row) -> _Block | None:
        """The paragraph this row continues: the last block, or the one before a caption or footnotes."""
        index = len(self.blocks) - 1
        while index >= 0 and self.blocks[index].kind in ("caption", "note", "figure"):
            index -= 1
        if index >= 0 and self._continues_para(self.blocks[index], row):
            return self.blocks[index]  # the caption/footnotes stay after the whole paragraph
        return None

    def _continues_caption(self, block: _Block, row: _Row) -> bool:
        prev = block.last
        return (
            row.page == prev.page
            and abs(row.size - prev.size) < 0.5
            and 0 < row.center - prev.center <= 1.6 * (prev.y1 - prev.y0)
            and not _CAPTION.match(row.text)
        )

    def _continues_code(self, block: _Block, row: _Row) -> bool:
        prev = block.last
        if row.page != prev.page:
            return self.interrupted
        # Blank lines inside a listing leave gaps of a few lines.
        return 0 < row.center - prev.center <= 6.0 * max(prev.size, row.size)

    def _continues_table(self, block: _Block, row: _Row) -> bool:
        prev = block.last
        if row.page != prev.page or not 0 < row.center - prev.center <= 2.6 * self.layout.body_size:
            return False
        if len(row.cells) > 1:
            return True
        # A wrapped cell: a lone row aligned with one of the table's columns.
        columns = [cell.x0 for cell in prev.cells] + [cell.x0 for r in block.rows for cell in r.cells]
        return any(abs(row.x0 - x) <= self.layout.body_size for x in columns) and (
            row.center - prev.center <= self.layout.pitch * 1.35
        )

    def _start(self, kind: str, row: _Row) -> _Block:
        block = _Block(kind, [row])
        if kind in ("para", "heading", "caption", "note"):
            block.pieces.append([row.text, row.page])
        self.blocks.append(block)
        return block


def _render(block: _Block) -> list[list[Any]]:
    """``[text, page]`` pieces of a block, with code indentation and table rows rebuilt."""
    if block.kind == "code":
        left = min(row.x0 for row in block.rows)
        widths = [(row.x1 - row.x0) / max(len(row.text), 1) for row in block.rows if len(row.text) >= 4]
        char = statistics.median(widths) if widths else block.rows[0].size * 0.5
        pieces = []
        for index, row in enumerate(block.rows):
            text = " ".join(cell.text for cell in row.cells)
            if index and text.lstrip().startswith("↪"):
                # The typesetter wrapped a long line and marked the rest with ↪: join it back.
                pieces.append([" " + text.lstrip()[1:].lstrip(), row.page])
                continue
            indent = " " * max(0, round((row.x0 - left) / char)) if char > 0 else ""
            line = indent + text
            prefix = ""
            if index:
                prev = block.rows[index - 1]
                blank = prev.page == row.page and row.center - prev.center > 1.8 * (prev.y1 - prev.y0)
                prefix = "\n\n" if blank else "\n"
            pieces.append([prefix + line, row.page])
        return pieces
    if block.replacement is not None:
        return [[block.replacement, block.rows[0].page]] if block.replacement else []
    if block.kind == "table":
        lines: list[tuple[list[list[Any]], int]] = []  # cells [x0, text] per row
        tolerance = block.rows[0].size * 1.2
        for row in block.rows:
            if lines:
                cells = lines[-1][0]
                # Wrapped cells: the row leaves the first column empty and lines up with others.
                # A row of figures is data under a row label spanning several rows, not a wrap.
                numbers = sum(1 for piece in row.cells if _NUMBER_CELL.match(piece.text.strip()))
                if (row.x0 > cells[0][0] + tolerance or len(row.cells) == 1) and numbers < 2:
                    targets = [min(cells, key=lambda cell: abs(cell[0] - piece.x0)) for piece in row.cells]
                    if all(abs(t[0] - piece.x0) <= tolerance for t, piece in zip(targets, row.cells)):
                        for target, piece in zip(targets, row.cells):
                            target[1] = _join_text(target[1], piece.text.strip())
                        continue
            lines.append(([[cell.x0, cell.text.strip()] for cell in row.cells], row.page))
        return [[("\n" if index else "") + " | ".join(cell[1] for cell in cells), page] for index, (cells, page) in enumerate(lines)]
    return [list(piece) for piece in block.pieces]


# ----------------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------------


def _make_unit(blocks: list[_Block], heading: str, image_pages: set[int]) -> DocumentUnit | None:
    text = ""
    page_map: list[list[int]] = []
    for block in blocks:
        pieces = _render(block)
        if not "".join(piece[0] for piece in pieces).strip():
            continue
        if text:
            text += "\n\n"
        for piece_text, page in pieces:
            if not page_map or page_map[-1][1] != page:
                page_map.append([len(text), page])
            text += piece_text
    if not text.strip():
        return None
    first, last = min(page for _, page in page_map), max(page for _, page in page_map)
    metadata: dict[str, Any] = {"pages": [first, last], "page_map": page_map}
    covered = sorted(page for page in image_pages if first <= page <= last)
    if covered:
        metadata["image_only_pages"] = covered
    return DocumentUnit(
        text=text,
        heading=heading,
        locator=f"page:{first}" if first == last else f"page:{first}-{last}",
        metadata=metadata,
    )


def _sections_from_outline(blocks: list[_Block], entries: list[_Entry]) -> list[tuple[int, str, list[_Block]]]:
    """``(level, title, blocks)`` per bookmark, in reading order, plus front matter."""
    starts: list[tuple[int, int]] = []  # (block index, entry index)
    for block_index, block in enumerate(blocks):
        for entry_index in block.splits:
            starts.append((block_index, entry_index))
    starts.sort()
    sections: list[tuple[int, str, list[_Block]]] = []
    if not starts or starts[0][0] > 0:
        sections.append((0, "", blocks[: starts[0][0] if starts else len(blocks)]))
    for position, (block_index, entry_index) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(blocks)
        entry = entries[entry_index]
        sections.append((entry.level, entry.title, blocks[block_index:end]))
    return sections


def _fold_sections(sections: list[tuple[int, str, list[_Block]]]) -> list[tuple[str, list[_Block]]]:
    """Resolve heading paths; sections holding only their heading fold into the next one."""
    result: list[tuple[str, list[_Block]]] = []
    stack: list[tuple[int, str]] = []
    carry: list[_Block] = []
    carry_title: tuple[int, str] | None = None
    for level, title, blocks in sections:
        if carry_title is not None:
            prev_level, prev_title = carry_title
            if prev_level == level and prev_title:
                # Sibling bookmarks for one printed heading ("第1章" + "大语言模型简介").
                title = f"{prev_title} {title}".strip()
            else:
                while stack and stack[-1][0] >= prev_level:
                    stack.pop()
                stack.append((prev_level, prev_title))
            carry_title = None
        if level:
            while stack and stack[-1][0] >= level:
                stack.pop()
        content = carry + blocks
        if level and all(block.kind == "heading" for block in blocks):
            carry, carry_title = content, (level, title)
            continue
        carry = []
        path = [t for _, t in stack if t] + ([title] if title else [])
        if level:
            stack.append((level, title))
        result.append((" > ".join(path), content))
    if carry:
        if carry_title is not None:
            while stack and stack[-1][0] >= carry_title[0]:
                stack.pop()
            stack.append(carry_title)
        result.append((" > ".join(t for _, t in stack if t), carry))
    return result


def _heading_levels(blocks: list[_Block], layout: _Layout) -> None:
    """Without an outline: promote large/bold/numbered standalone rows to headings."""
    candidates: list[_Block] = []
    for index, block in enumerate(blocks):
        text = block.text().strip()
        if block.kind != "para" or not text or len(text) > 80:
            continue
        row = block.rows[0]
        prominent = row.size >= layout.body_size * 1.15
        bold = row.bold and len(text) <= 40
        pattern = len(text) <= 40 and bool(_CHAPTER.match(text) or _NUMBERED.match(text))
        question = bool(_QUESTION.match(text)) and (row.bold or row.size >= layout.body_size * 1.05)
        if text[-1] in "，,；;：:" or (_ends_sentence(text) and not question and not prominent):
            continue
        if prominent or ((bold or pattern) and len(block.rows) <= 2) or question:
            candidates.append(block)
    if not candidates:
        return
    sizes = sorted({round(block.rows[0].size) for block in candidates if block.rows[0].size >= layout.body_size * 1.15}, reverse=True)
    for block in candidates:
        size = round(block.rows[0].size)
        block.kind = "heading"
        block.level = sizes.index(size) + 1 if size in sizes else len(sizes) + 1


def _sections_from_headings(blocks: list[_Block]) -> list[tuple[int, str, list[_Block]]]:
    sections: list[tuple[int, str, list[_Block]]] = []
    current: list[_Block] = []
    level, title = 0, ""
    for block in blocks:
        if block.kind == "heading":
            if current:
                sections.append((level, title, current))
            current = [block]
            level, title = block.level, block.text().strip()
        else:
            current.append(block)
    if current:
        sections.append((level, title, current))
    return sections


def _units_by_page(blocks: list[_Block], image_pages: set[int]) -> list[DocumentUnit]:
    per_page: dict[int, list[str]] = defaultdict(list)
    for block in blocks:
        current_page = None
        for text, page in _render(block):
            if page != current_page:
                per_page[page].append("")
                current_page = page
            per_page[page][-1] += text
    units = []
    for page in sorted(per_page):
        text = "\n\n".join(part.strip() for part in per_page[page] if part.strip())
        if text:
            units.append(
                DocumentUnit(
                    text=text,
                    heading=f"第 {page} 页",
                    locator=f"page:{page}",
                    metadata={"page": page, "pages": [page, page], "page_map": [[0, page]]},
                )
            )
    return units


# ----------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------


def _has_graphics(page: pymupdf.Page) -> bool:
    """A picture, or enough vector paths to be a drawing / outlined text (needs OCR)."""
    try:
        if page.get_images(full=False):
            return True
        return sum(len(path["items"]) for path in page.get_drawings()) >= 50
    except Exception:
        return False


def _find_tables(page: _Page, kinds: list[str], layout: _Layout) -> list[Figure]:
    """Table regions of a page, to be re-read from their image (cloud mode).

    A region is a run of at least three table rows in one column, with the lone labels
    between them (a row label centred across rows). The text-layer rows go along as
    ``labels``, so the recognised table can be checked against them.
    """
    runs: list[list[_Row]] = []
    for row, kind in zip(page.rows, kinds):
        if kind not in ("table", "label"):
            continue
        last = runs[-1][-1] if runs else None
        if (
            last is not None
            and row.col == last.col
            and 0 < row.center - last.center <= 2.6 * layout.body_size
        ):
            runs[-1].append(row)
        elif kind == "table":
            runs.append([row])
    tables = []
    for run in runs:
        if sum(len(row.cells) > 1 for row in run) < 3:
            continue
        rect = pymupdf.Rect(min(r.x0 for r in run), min(r.y0 for r in run), max(r.x1 for r in run), max(r.y1 for r in run))
        tables.append(Figure(page=page.number, bbox=tuple(rect + (-4, -4, 4, 4)), kind="table",
                             labels=[row.text for row in run]))
    return tables


def _table_agrees(recognised: str, labels: list[str]) -> bool:
    """The recognised table carries the text layer's numbers, and not many others.

    The image may show more than the text-layer rows (header numbers like "Top-20", a
    row read as prose), so some extra numbers are fine; a dropped column or a model
    stuck repeating itself is not.
    """
    model, text = Counter(table_numbers(recognised)), Counter(table_numbers("\n".join(labels)))
    if not model or not text:
        return False
    shared = sum((model & text).values())
    return shared >= 0.85 * sum(text.values()) and shared >= 0.6 * sum(model.values())


def _apply_tables(blocks: list[_Block], tables: list[tuple[Figure, str]]) -> int:
    """Swap text-layer table blocks for recognised tables; returns how many were swapped."""
    swapped = 0
    for figure, text in tables:
        x0, y0, x1, y1 = figure.bbox
        inside = [
            block for block in blocks
            if block.kind in ("table", "para", "note") and block.rows
            and all(row.page == figure.page and x0 <= (row.x0 + row.x1) / 2 <= x1 and y0 <= row.center <= y1
                    for row in block.rows)
        ]
        first = next((block for block in inside if block.kind == "table"), None)
        if first is None:
            continue
        for block in inside:
            block.replacement = ""
        first.replacement = text
        swapped += 1
    return swapped


def _find_figures(pdf_page: pymupdf.Page, page: _Page, kinds: list[str]) -> list[Figure]:
    """Figure regions on a page: clusters of vector drawings and large bitmaps.

    Boxes that mostly hold body text, code or tables (sidebars, listings, bordered
    tables) are not figures. Each figure gets its caption (图x-y below or above it),
    the text-layer labels inside it and a little surrounding text as context.
    """
    page_box = pymupdf.Rect(0, 0, page.width, page.height)
    area = page.width * page.height
    regions: list[pymupdf.Rect] = []
    try:
        clusters = pdf_page.cluster_drawings()
    except Exception:
        clusters = []
    drawings = None
    for rect in clusters:
        rect = pymupdf.Rect(rect)
        if rect.width >= 80 and rect.height >= 40 and rect.width * rect.height >= 0.02 * area:
            if drawings is None:
                try:
                    drawings = [pymupdf.Rect(d["rect"]) for d in pdf_page.get_drawings()]
                except Exception:
                    drawings = []
            paths = [d for d in drawings if rect.contains(d)]
            if paths and all(d.height <= 2 for d in paths):
                continue  # only horizontal rules: a booktabs table, not a figure
            regions.append(rect)
    try:
        images = pdf_page.get_image_info()
    except Exception:
        images = []
    for info in images:
        rect = pymupdf.Rect(info["bbox"]) & page_box
        if rect.width >= 60 and rect.height >= 40 and rect.width * rect.height >= 0.015 * area:
            regions.append(rect)
    merged: list[pymupdf.Rect] = []
    for rect in sorted(regions, key=lambda r: (r.y0, r.x0)):
        for other in merged:
            if (rect + (-6, -6, 6, 6)).intersects(other):
                other |= rect
                break
        else:
            merged.append(pymupdf.Rect(rect))

    figures = []
    rows = list(zip(page.rows, kinds))
    for rect in merged:
        inside = [(row, kind) for row, kind in rows
                  if rect.x0 - 4 <= (row.x0 + row.x1) / 2 <= rect.x1 + 4 and rect.y0 - 4 <= row.center <= rect.y1 + 4]
        boxed = sum(len(row.text) for row, kind in inside if kind in ("code", "table", "heading", "big"))
        labels = sum(len(row.text) for row, kind in inside if kind == "label")
        wide_prose = sum(len(row.text) for row, kind in inside
                         if kind in ("line", "note") and row.x1 - row.x0 >= 0.6 * rect.width)
        if boxed > 60 and boxed > 2 * labels:
            continue  # a code listing or a table drawn in a box
        if wide_prose > 100 and wide_prose > 2 * labels:
            continue  # a sidebar of running text
        caption, best = "", None
        for row, kind in rows:
            if kind != "caption" or row.x1 < rect.x0 or row.x0 > rect.x1:
                continue
            below = row.y0 >= rect.y1 - 4
            gap = row.y0 - rect.y1 if below else rect.y0 - row.y1
            if -4 <= gap <= (60 if below else 30) and (best is None or gap < best):
                caption, best = row.text, gap
        above = [row.text for row, kind in rows if kind == "line" and row.y1 <= rect.y0][-2:]
        figures.append(Figure(
            page=page.number, bbox=tuple(rect), kind="figure", caption=caption, context="".join(above),
            labels=[cell.text.strip() for row, kind in inside if kind == "label" for cell in row.cells],
        ))
    return figures


def _mark_token_rows(pdf_page: pymupdf.Page, page: _Page) -> None:
    """Tokenizer illustrations: text set token by token on small colored boxes.

    Read as code they turn into "CAP ITAL IZATION show _ tokens …". Each box is one
    token, so the row is rebuilt from the characters inside each box: "CAP|ITAL|IZATION".
    """
    try:
        boxes = [pymupdf.Rect(d["rect"]) for d in pdf_page.get_drawings()
                 if d.get("fill") and tuple(d["fill"]) != (1.0, 1.0, 1.0) and 3 < d["rect"].width < 150 and d["rect"].height < 24]
    except Exception:
        return
    if len(boxes) < 6:
        return
    for row in page.rows:
        if not row.mono or row.y1 - row.y0 > 24:
            continue
        mine = sorted((b for b in boxes if b.y0 <= row.center <= b.y1 and b.x0 >= row.x0 - 6 and b.x1 <= row.x1 + 6),
                      key=lambda b: b.x0)
        if len(mine) < 3 or sum(b.width for b in mine) < 0.5 * (row.x1 - row.x0):
            continue
        chars = [c for block in pdf_page.get_text("rawdict", clip=pymupdf.Rect(row.x0 - 2, row.y0, row.x1 + 2, row.y1))["blocks"]
                 for line in block.get("lines", []) for span in line["spans"] for c in span["chars"]]
        tokens = []
        for box in mine:
            text = "".join(c["c"] for c in chars if box.x0 - 0.5 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= box.x1 + 0.5).strip()
            tokens.append(text or "␣")
        row.tokens = "|".join(tokens)


def _is_formula_row(row: _Row) -> bool:
    """A display formula row: mostly math, or a short line with math in it ("IDF(𝑡) = ln")."""
    letters = len(re.sub(r"\s", "", row.text)) or 1
    return row.math >= 2 and row.math >= 0.4 * letters or row.math >= 1 and letters <= 25


def _find_formulas(page: _Page, layout: _Layout) -> list[Figure]:
    """Display formulas on a page; rows of one formula (a fraction's numerator, ...) are grouped."""
    groups: list[list[_Row]] = []
    for row in page.rows:
        if row.mono or not _is_formula_row(row):
            continue  # code listings sometimes set quotes in a math font
        if groups and 0 < row.center - groups[-1][-1].center <= 2.2 * layout.body_size:
            groups[-1].append(row)
        else:
            groups.append([row])
    formulas = []
    for rows in groups:
        x0, y0 = min(r.x0 for r in rows), min(r.y0 for r in rows)
        x1, y1 = max(r.x1 for r in rows), max(r.y1 for r in rows)
        above = [r.text for r in page.rows if r.y1 <= y0][-1:]
        formulas.append(Figure(page=page.number, bbox=(x0, y0, x1, y1), kind="formula",
                               context="".join(above), labels=[r.text for r in rows]))
    return formulas


def _snapshot(pdf_page: pymupdf.Page, rect: pymupdf.Rect, zoom: float = 2.5, limit: int = 2000) -> bytes:
    """PNG of a page region, at most ``limit`` pixels on the long side."""
    clip = (rect + (-6, -6, 6, 6)) & pdf_page.rect
    zoom = min(zoom, limit / max(clip.width, clip.height, 1))
    return pdf_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip).tobytes("png")


def _figure_block(figure: Figure, text: str, mode: str, layout: _Layout) -> _Block:
    if figure.kind == "page":
        body = f"〔第 {figure.page} 页 识别文字〕\n{text}"
    elif figure.kind == "formula":
        body = "〔公式〕" + re.sub(r"\s*\n\s*", " ", text.strip())
    else:
        # One paragraph: list lines from the model become "；"-separated items.
        text = re.sub(r"\s*\n+\s*(?:[-•*]|\d+[.、)])?\s*", "；", text.strip())
        text = re.sub(r"([：:。！？；])；", r"\1", text)
        body = f"〔{'图片说明' if mode == 'cloud' else '图中文字'}〕{text}"
    x0, y0, x1, y1 = figure.bbox
    row = _Row(figure.page, [_Piece(body, x0, y0, x1, y1, layout.body_size, "", False, False)])
    return _Block("figure", [row], [[body, figure.page]], sealed=True)


def parse_pdf(
    path: Path,
    reader: Reader | None = None,
    *,
    page_limit: int | None = None,
    page_texts: dict[int, str] | None = None,
    on_page: Callable[[int, int], None] | None = None,
    pause: Callable[[], None] | None = None,
) -> tuple[list[DocumentUnit], dict[str, Any]]:
    """Return section units and document metadata (strategy, image-only pages, ...).

    With a ``reader`` (see ``figures.py``) each figure becomes a "〔图片说明〕" (cloud) or
    "〔图中文字〕" (local) block right after its caption, and image-only pages are read
    as text, so a fully scanned PDF can be indexed too.

    A long scan is read in passes: at most ``page_limit`` image-only pages are read (and
    rendered) per call, and ``page_texts`` (page -> text, filled in as pages are read)
    carries the pages read by earlier passes, which are not read again. The pages left
    are counted in ``metadata["pages_pending"]``. ``on_page(read, total)`` follows the
    image-only pages; ``pause()`` runs before each figure is read and before each page's
    own work here (PyMuPDF keeps the interpreter while it reads or draws a page).
    """
    wait = pause or (lambda: None)
    with pymupdf.open(path) as pdf:
        pages = []
        for index, page in enumerate(pdf, start=1):
            wait()
            pages.append(_Page(index, page.rect.width, page.rect.height,
                               _group_rows(_page_pieces(page), index, page.rect.width)))
        entries = _outline(pdf)
        removed = 0
        layout = _measure(pages)
        if any(page.rows for page in pages):
            removed = _strip_running_text(pages, layout, {_norm_title(entry.title) for entry in entries})
        image_pages = set()
        for page in pages:
            if sum(len(row.text.strip()) for row in page.rows) < 5:
                wait()
                if _has_graphics(pdf[page.number - 1]):
                    image_pages.add(page.number)

        positions = _match_outline(pages, entries, layout) if entries else {}
        for page in pages:
            if any(row.mono for row in page.rows):
                wait()
                _mark_token_rows(pdf[page.number - 1], page)
        figures: list[Figure] = []
        # Pages read by an earlier pass (figure index -> text) and pages left for a later one.
        preset: dict[int, str] = {}
        reading = pages_left = 0
        if reader is not None:
            probe = _Assembler(layout)
            for page in pages:
                wait()
                pdf_page = pdf[page.number - 1]
                if page.number in image_pages:
                    if page_texts is not None and page.number in page_texts:
                        preset[len(figures)] = page_texts[page.number]
                        figures.append(Figure(page=page.number, bbox=tuple(pdf_page.rect), kind="page"))
                    elif page_limit is not None and reading >= page_limit:
                        pages_left += 1
                    else:
                        # Rendering stays on this thread: PyMuPDF documents are not thread-safe.
                        figures.append(Figure(page=page.number, bbox=tuple(pdf_page.rect), kind="page",
                                              image=_snapshot(pdf_page, pdf_page.rect, zoom=PAGE_ZOOM)))
                        reading += 1
                    continue
                probe.new_page(page)
                found = _find_figures(pdf_page, page, probe.page_kinds(page.rows, tables=False))
                if reader.mode == "cloud":
                    tables = _find_tables(page, probe.page_kinds(page.rows), layout)
                    # A small-print table drawn with rules also looks like a figure: read it once, as a table.
                    found = [f for f in found if not any(
                        (pymupdf.Rect(f.bbox) & pymupdf.Rect(t.bbox)).get_area() >= 0.6 * pymupdf.Rect(f.bbox).get_area()
                        for t in tables)] + tables
                if getattr(reader, "reads_formulas", reader.mode == "cloud"):  # cloud, or the local formula component
                    found += _find_formulas(page, layout)
                for figure in found:
                    zoom = 3.0 if figure.kind == "formula" else 2.5
                    figure.image = _snapshot(pdf_page, pymupdf.Rect(figure.bbox), zoom=zoom)
                    figures.append(figure)

    texts: dict[int, str] = dict(preset)
    to_read = [index for index in range(len(figures)) if index not in preset]
    if reader is not None and to_read:
        done = [len(preset)]

        def page_done(position: int) -> None:
            if figures[to_read[position]].kind == "page" and on_page:
                done[0] += 1
                on_page(done[0], len(image_pages))

        read = read_all(reader, [figures[i] for i in to_read], keep_empty=page_texts is not None,
                        on_done=page_done, pause=pause)
        texts.update({to_read[position]: text for position, text in read.items()})
    if page_texts is not None:
        page_texts.update({figures[i].page: text for i, text in texts.items() if figures[i].kind == "page"})
    texts = {i: t for i, t in texts.items() if t}
    # A "formula" the model found no math in (or answered in words) adds only noise.
    texts = {i: t for i, t in texts.items() if figures[i].kind != "formula" or _LATEX_MATH.search(t)}
    # A recognised table replaces the text-layer one only when their numbers agree.
    tables = [(figures[i], table_text(otsl_rows(t))) for i, t in texts.items() if figures[i].kind == "table"]
    tables = [(figure, text) for figure, text in tables if _table_agrees(text, figure.labels)]
    texts = {i: t for i, t in texts.items() if figures[i].kind != "table"}
    loose: dict[int, list[_Block]] = defaultdict(list)
    after_caption: dict[tuple[int, str], _Block] = {}
    for index, text in texts.items():
        figure = figures[index]
        block = _figure_block(figure, text, reader.mode, layout)
        if figure.caption and (figure.page, figure.caption) not in after_caption:
            after_caption[(figure.page, figure.caption)] = block
        else:
            loose[figure.page].append(block)

    if not any(page.rows for page in pages) and not texts:
        return [], {"parser_version": PDF_PARSER_VERSION, "strategy": "empty", "pages": len(pages),
                    "image_only_pages": sorted(image_pages), "pages_pending": pages_left}

    assembler = _Assembler(layout)
    split_at: dict[tuple[int, int], list[int]] = defaultdict(list)
    for entry_index, (page_index, row_index) in positions.items():
        split_at[(page_index, row_index)].append(entry_index)
    pending: list[int] = []
    for page_index, page in enumerate(pages):
        assembler.new_page(page)
        kinds = assembler.page_kinds(page.rows) + [""]
        waiting = sorted(loose.get(page.number, []), key=lambda block: block.rows[0].y1)
        for row_index, row in enumerate(page.rows + [None]):
            pending.extend(split_at.get((page_index, row_index), []))
            # A figure without a caption goes where the text below it starts.
            while waiting and (row is None or row.y0 >= waiting[0].rows[0].y1 - 2):
                assembler.blocks.append(waiting.pop(0))
                assembler.interrupted = True
            if row is None:
                continue
            before = len(assembler.blocks)
            if pending:
                # A section starts here, whether or not its title is printed on this row.
                assembler.force_break = True
            assembler.feed(row, kinds[row_index])
            if pending:
                target = assembler.blocks[before] if len(assembler.blocks) > before else assembler.blocks[-1]
                target.splits.extend(pending)
                pending = []
            if kinds[row_index] == "caption" and (block := after_caption.pop((page.number, row.text), None)):
                assembler.blocks.append(block)
                assembler.interrupted = True
    if pending and assembler.blocks:
        assembler.blocks[-1].splits.extend(pending)
    # Captions that were not met as rows (merged into other text): keep their figures at the end.
    blocks = assembler.blocks + list(after_caption.values())
    tables_replaced = _apply_tables(blocks, tables)

    if entries:
        strategy = "outline"
        sections = _sections_from_outline(blocks, entries)
    else:
        _heading_levels(blocks, layout)
        sections = _sections_from_headings(blocks)
        strategy = "headings" if any(level for level, _, _ in sections) else "pages"

    if strategy == "pages":
        units = _units_by_page(blocks, image_pages)
    else:
        units = [
            unit
            for heading, section_blocks in _fold_sections(sections)
            if (unit := _make_unit(section_blocks, heading[:1000], image_pages)) is not None
        ]
    metadata = {
        "parser_version": PDF_PARSER_VERSION,
        "strategy": strategy,
        "pages": len(pages),
        "outline_entries": len(entries),
        "outline_titles_found": len({row.toc for page in pages for row in page.rows if row.toc is not None}),
        "image_only_pages": sorted(image_pages),
        "pages_pending": pages_left,
        "running_rows_removed": removed,
        "figure_labels_dropped": assembler.dropped_labels,
        "figure_mode": reader.mode if reader is not None else "off",
        "figures_found": sum(figure.kind == "figure" for figure in figures),
        "figures_read": sum(figures[i].kind == "figure" for i in texts),
        "pages_read": sum(figures[i].kind == "page" for i in texts),
        "formulas_read": sum(figures[i].kind == "formula" for i in texts),
        "tables_found": sum(figure.kind == "table" for figure in figures),
        "tables_replaced": tables_replaced,
    }
    return units, metadata
