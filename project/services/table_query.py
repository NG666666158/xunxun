"""Read-only questions over .xlsx files: count, list, sum, average, min, max, group, distinct.

A question becomes a ``TableQuery`` (a fixed set of operations, validated by Pydantic) and this
module runs it over the file's cells. Nothing is generated or evaluated: no SQL, no formulas,
no code. The answer is written from the computed result, so its numbers are the file's.

Reading a real workbook: a title row or two above the header, headers spanning several rows,
vertically merged categories (filled down, so "类别 = 税收政策" finds every row of the merge),
"……" placeholders in unfilled templates (blank), and 合计 rows (not data).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime
import os
from pathlib import Path
import re
import threading
from typing import Any, Literal

from openpyxl import load_workbook
from pydantic import BaseModel, Field, field_validator

HEADER_SCAN_ROWS = 8
MAX_LIST_ROWS = 50
MAX_GROUPS = 50
MAX_CITED_RANGES = 8
MAX_SAMPLE_VALUES = 12
SAMPLE_VALUE_CHARS = 24
CELL_CHARS = 80

_PLACEHOLDER = re.compile(r"^[.。·…\s]+$")
_TOTAL_ROW = re.compile(r"^\s*(?:合\s*计|总\s*计|小\s*计|总\s*和|total|subtotal)\s*[:：]?\s*$", re.I)
_INDEX_COLUMN = re.compile(r"序号|编号|^no\.?$|^id$", re.I)
_NUMBER = re.compile(
    r"^\s*([+-]?\d[\d,，]*(?:\.\d+)?)\s*(亿元|万元|千元|亿|万|千|元|人民币|%|％|人|家|个|台|件|项|条|次|年|月|天|kg|g|t|km|m|mm|nm)?\s*$",
    re.I,
)
_SCALE = {"亿元": 1e8, "亿": 1e8, "万元": 1e4, "万": 1e4, "千元": 1e3, "千": 1e3}


class TableCondition(BaseModel):
    column: str = Field(default="", max_length=200)  # "" = any column
    op: Literal["eq", "ne", "contains", "not_contains", "gt", "gte", "lt", "lte", "empty", "not_empty"] = "contains"
    value: str = Field(default="", max_length=200)

    @field_validator("value", mode="before")
    @classmethod
    def as_text(cls, value: Any) -> str:
        return "" if value is None else str(value)


class TableQuery(BaseModel):
    """One operation. ``none``: the question is not a count, list or calculation over the table."""

    op: Literal["count", "list", "sum", "avg", "min", "max", "group", "distinct", "sheets", "none"]
    sheet: str = Field(default="", max_length=200)  # "" = every sheet with the columns named
    column: str = Field(default="", max_length=200)  # what sum/avg/min/max/distinct work on
    where: list[TableCondition] = Field(default_factory=list, max_length=8)
    group_by: str = Field(default="", max_length=200)
    agg: Literal["count", "sum", "avg", "min", "max"] = "count"  # for group
    columns: list[str] = Field(default_factory=list, max_length=8)  # shown for list
    order_by: str = Field(default="", max_length=200)
    descending: bool = False
    limit: int = Field(default=20, ge=1, le=MAX_LIST_ROWS)


class TableError(ValueError):
    """A query this workbook cannot answer; the message is shown to the user."""


@dataclass
class Sheet:
    name: str
    header_row: int
    columns: list[str]
    # (Excel row number, one value per column), blank rows and 合计 rows left out.
    rows: list[tuple[int, list[Any]]] = field(default_factory=list)
    total_rows: list[int] = field(default_factory=list)
    hidden: bool = False

    @property
    def index_column(self) -> int | None:
        for i, name in enumerate(self.columns):
            if _INDEX_COLUMN.search(name.rsplit(" / ", 1)[-1]):
                return i
        return None


@dataclass
class Workbook:
    sheets: list[Sheet]
    hidden: list[str] = field(default_factory=list)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d") if not (value.hour or value.minute) else value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _blank(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and (not value.strip() or bool(_PLACEHOLDER.match(value)))


def _loose(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def number_of(value: Any) -> float | None:
    """A cell as a number: a numeric cell, or text like "1,200", "30%", "3.5 万元"."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = _NUMBER.match(value)
    if not match:
        return None
    return float(match.group(1).replace(",", "").replace("，", "")) * _SCALE.get(match.group(2) or "", 1.0)


def _header_row(raw: list[list[Any]]) -> int | None:
    """Index of the header: the row with the most separate text cells, among the first rows
    above the data. Data starts at the first row holding a number or a date ("1", 3.5)."""
    best, best_count = None, 1
    for index, row in enumerate(raw[:HEADER_SCAN_ROWS]):
        if best is not None and any(_is_data_value(value) for value in row):
            break
        count = sum(1 for value in row if isinstance(value, str) and not _blank(value))
        if count > best_count:
            best, best_count = index, count
    return best


def _is_data_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(re.fullmatch(r"\s*\d+\s*", value))
    return isinstance(value, (int, float, date))


def _column_names(filled: list[list[Any]], raw: list[list[Any]], header: int, width: int) -> list[str]:
    """Names from the header row, prefixed by header rows above it ("精密级构筑 / 细分品类名称").
    A row above with a single value across the sheet is a title, not a header."""
    first = header
    while first > 0:
        above = [value for value in raw[first - 1] if not _blank(value)]
        if len(above) < 2:
            break
        first -= 1
    names: list[str] = []
    for column in range(width):
        parts: list[str] = []
        for row in range(first, header + 1):
            text = _text(filled[row][column]) if column < len(filled[row]) else ""
            text = re.sub(r"\s+", " ", text)
            if text and not _blank(text) and text not in parts:
                parts.append(text)
        names.append(" / ".join(parts) or f"列{column + 1}")
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return unique


def _read_sheet(worksheet: Any) -> Sheet | None:
    raw = [list(row) for row in worksheet.iter_rows(values_only=True)]
    if not raw:
        return None
    width = max(len(row) for row in raw)
    filled = [row + [None] * (width - len(row)) for row in raw]
    raw = [list(row) for row in filled]
    # Merged cells: every covered cell takes the value (the raw grid keeps them empty).
    merges = [
        (r.min_row - 1, r.max_row - 1, r.min_col - 1, r.max_col - 1) for r in worksheet.merged_cells.ranges
    ]
    for top, bottom, left, right in merges:
        value = filled[top][left]
        for row in range(top, min(bottom, len(filled) - 1) + 1):
            for column in range(left, min(right, width - 1) + 1):
                filled[row][column] = value
    header = _header_row(raw)
    if header is None:
        return None
    sheet = Sheet(worksheet.title, header + 1, _column_names(filled, raw, header, width))
    sheet.hidden = worksheet.sheet_state != "visible"
    # A row merged from above (one row in several grid lines) is one row of data.
    continuation = set()
    for top, bottom, left, right in merges:
        if left == 0 and right == width - 1:
            continuation.update(range(top + 1, bottom + 1))
    for index in range(header + 1, len(filled)):
        if index in continuation or all(_blank(value) for value in raw[index]):
            continue
        first = next((value for value in filled[index] if not _blank(value)), "")
        if isinstance(first, str) and _TOTAL_ROW.match(first):
            sheet.total_rows.append(index + 1)
            continue
        sheet.rows.append((index + 1, [None if _blank(value) else value for value in filled[index]]))
    return sheet


def read_workbook(path: str | os.PathLike[str]) -> Workbook:
    # Not read_only: only the full mode knows merged cells.
    book = load_workbook(path, data_only=True)
    try:
        sheets = [sheet for sheet in map(_read_sheet, book.worksheets) if sheet is not None]
    finally:
        book.close()
    return Workbook(
        [sheet for sheet in sheets if not sheet.hidden],
        [sheet.name for sheet in sheets if sheet.hidden],
    )


class WorkbookCache:
    """The last few workbooks read, keyed by path + modification time + size. Rows are what
    costs memory (about 3 KB each), so the cache also keeps to ``max_rows`` in total."""

    def __init__(self, capacity: int = 4, max_rows: int = 100_000):
        self.capacity = capacity
        self.max_rows = max_rows
        self._items: OrderedDict[tuple[str, int, int], Workbook] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: str | os.PathLike[str]) -> Workbook:
        info = os.stat(path)
        key = (os.fspath(path), info.st_mtime_ns, info.st_size)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        workbook = read_workbook(path)
        with self._lock:
            self._items[key] = workbook
            while len(self._items) > self.capacity or (len(self._items) > 1 and self._rows() > self.max_rows):
                self._items.popitem(last=False)
            if self._rows() > self.max_rows:
                self._items.clear()
        return workbook

    def _rows(self) -> int:
        return sum(len(sheet.rows) for item in self._items.values() for sheet in item.sheets)


def schema_text(workbook: Workbook) -> str:
    """What the query planner is told about a workbook: sheets, columns and sample values."""
    lines = []
    for sheet in workbook.sheets:
        lines.append(f"工作表「{sheet.name}」，{len(sheet.rows)} 行数据：")
        for index, name in enumerate(sheet.columns):
            values = [_text(row[index]) for _, row in sheet.rows if row[index] is not None]
            distinct = list(dict.fromkeys(values))
            numbers = sum(1 for _, row in sheet.rows if number_of(row[index]) is not None)
            kind = "数字" if values and numbers >= 0.8 * len(values) else "文本"
            detail = ""
            short = [v.replace("\n", " ") for v in distinct if len(v) <= SAMPLE_VALUE_CHARS]
            if distinct and len(distinct) <= MAX_SAMPLE_VALUES and len(short) == len(distinct):
                detail = f"，取值：{'、'.join(short)}"
            elif short:
                detail = f"，例如：{'、'.join(short[:3])}"
            lines.append(f"  - {name}（{kind}{detail}）")
    return "\n".join(lines)


# "第二个工作表", "第3张表", "最后一个 sheet": the question says which sheet by its place.
_SHEET_BY_PLACE = re.compile(
    r"(?:第\s*[一二三四五六七八九十\d]+|最后一?)\s*(?:个|张)?\s*(?:工作表|表|sheet|标签页?|页签)", re.I
)


def _said(name: str, text: str) -> bool:
    """``name`` is in ``text`` (both loose): whole, or more than half of it in pieces of two or
    more characters ("近期出台的普惠政策" says 「近期出台普惠政策」; "多少条政策" does not say
    「普惠政策」: only half of it)."""
    if not name:
        return False
    if name in text:
        return True
    covered = [False] * len(name)
    for start in range(len(name) - 1):
        for end in range(len(name), start + 1, -1):
            if name[start:end] in text:
                covered[start:end] = [True] * (end - start)
                break
    return sum(covered) * 2 > len(name)


def sheet_asked_for(question: str, sheet: str, workbook: Workbook, file_name: str = "") -> str:
    """The planner's ``sheet`` when the question names that sheet, otherwise "" (every sheet
    with the columns the query uses).

    Qwen3-8B fills in a sheet the question never mentions: the first one ("《0.政策补贴汇总》
    一共列了多少条政策" counted 「普惠政策」, 26 of 104) or the workbook's own name. The file's
    name in the question is not a sheet's; "第二个工作表" names one by its place.
    """
    if not sheet.strip():
        return ""
    want = _loose(sheet)
    stem = _loose(Path(file_name).stem) if file_name else ""
    names = [_loose(s.name) for s in workbook.sheets]
    if stem and not any(want in name or name in want for name in names if name) and (want in stem or stem in want):
        return ""  # the workbook's name where a sheet's goes: the question is about all of it
    text = question
    for title in re.findall(r"《([^《》]+)》", question):
        if stem and (_loose(title) in stem or stem in _loose(title)):
            text = text.replace(f"《{title}》", " ")
    if _SHEET_BY_PLACE.search(text):
        return sheet
    text = _loose(text)
    if stem:
        text = text.replace(stem, " ")
    try:
        meant = _find_sheets(workbook, sheet)
    except TableError:
        # No such sheet: kept only when the question asks for it, so the answer says it isn't there.
        return sheet if _said(want, text) else ""
    # The sheets it stands for must be the ones named: "多少条政策" is not 「政策」 = both 普惠政策 sheets.
    return sheet if any(_said(_loose(s.name), text) for s in meant) else ""


def _find_sheets(workbook: Workbook, name: str) -> list[Sheet]:
    if not name.strip():
        return list(workbook.sheets)
    want = _loose(name)
    exact = [sheet for sheet in workbook.sheets if _loose(sheet.name) == want]
    if exact:
        return exact
    near = [sheet for sheet in workbook.sheets if want in _loose(sheet.name) or _loose(sheet.name) in want]
    if near:
        return near
    names = "、".join(sheet.name.strip() for sheet in workbook.sheets)
    raise TableError(f"没有叫“{name}”的工作表（有：{names}）。")


def _find_column(sheet: Sheet, name: str) -> int | None:
    want = _loose(name)
    if not want:
        return None
    names = [_loose(column) for column in sheet.columns]
    for test in (lambda c: c == want, lambda c: c.rsplit("/", 1)[-1] == want, lambda c: want in c, lambda c: c in want):
        found = [i for i, c in enumerate(names) if c and test(c)]
        if found:
            return found[0]
    return None


def _matches(cell: Any, condition: TableCondition) -> bool:
    op = condition.op
    if op == "empty":
        return cell is None
    if op == "not_empty":
        return cell is not None
    if op in ("gt", "gte", "lt", "lte"):
        left, right = number_of(cell), number_of(condition.value)
        if left is None or right is None:
            text = _text(cell)
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", text):
                return False
            left, right = text, condition.value  # dates compare as ISO text
        return {"gt": left > right, "gte": left >= right, "lt": left < right, "lte": left <= right}[op]
    text, value = _loose(_text(cell)), _loose(condition.value)
    if op == "eq":
        left, right = number_of(cell), number_of(condition.value)
        return left == right if left is not None and right is not None else text == value
    if op == "ne":
        return text != value
    if op == "contains":
        return value in text
    return value not in text  # not_contains


def _selected(sheet: Sheet, where: list[TableCondition]) -> list[tuple[int, list[Any]]]:
    """Rows meeting every condition (a condition on no column looks at all of them)."""
    tests: list[tuple[int | None, TableCondition]] = []
    for condition in where:
        column = _find_column(sheet, condition.column)
        if condition.column.strip() and column is None:
            raise TableError(f"工作表“{sheet.name.strip()}”没有“{condition.column}”这一列。")
        tests.append((column, condition))
    rows = []
    for number, row in sheet.rows:
        keep = True
        for column, condition in tests:
            if column is not None:
                keep = _matches(row[column], condition)
            elif condition.op in ("not_contains", "ne"):
                keep = all(_matches(cell, condition) for cell in row)
            else:
                keep = any(_matches(cell, condition) for cell in row)
            if not keep:
                break
        if keep:
            rows.append((number, row))
    return rows


def row_ranges(numbers: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for number in sorted(numbers):
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], number)
        else:
            ranges.append((number, number))
    return ranges


def locators(sheet: str, numbers: list[int]) -> list[str]:
    """``sheet:X;rows:a-b`` per run of rows, the way parsed spreadsheets are cited."""
    return [f"sheet:{sheet};rows:{a}-{b}" for a, b in row_ranges(numbers)[:MAX_CITED_RANGES]]


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")


@dataclass
class SheetResult:
    sheet: str
    rows: list[int]
    # count: rows; sum/avg/min/max: the value; group: [(label, value, rows)]; distinct: [(value, rows)].
    value: float | None = None
    groups: list[tuple[str, float, list[int]]] = field(default_factory=list)
    listed: list[tuple[int, list[tuple[str, str]]]] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)  # rows without a number where one was needed
    no_index: int = 0  # rows without a 序号
    column: str = ""
    group_column: str = ""


@dataclass
class TableResult:
    op: str
    query: TableQuery
    sheets: list[SheetResult]
    hidden: list[str] = field(default_factory=list)
    totals_left_out: int = 0

    @property
    def total_rows(self) -> int:
        return sum(len(sheet.rows) for sheet in self.sheets)


def _aggregate(op: str, values: list[float]) -> float | None:
    if not values:
        return None
    if op == "sum":
        return sum(values)
    if op == "avg":
        return sum(values) / len(values)
    return min(values) if op == "min" else max(values)


def _shown_columns(sheet: Sheet, query: TableQuery, rows: list[tuple[int, list[Any]]]) -> list[int]:
    """Named columns; otherwise the first short ones that have values, 序号 left out."""
    named = [i for i in (_find_column(sheet, name) for name in query.columns) if i is not None]
    if named:
        return named
    candidates = []
    for index in range(len(sheet.columns)):
        if index == sheet.index_column:
            continue
        values = [_text(row[index]) for _, row in rows if row[index] is not None]
        if values:
            candidates.append((index, sum(map(len, values)) / len(values)))
    short = [index for index, length in candidates if length <= 40] or [index for index, _ in candidates]
    return sorted(short[:3])


def run_query(query: TableQuery, workbook: Workbook) -> TableResult:
    """Run one validated query. Raises ``TableError`` when the workbook has no such sheet or column."""
    sheets = _find_sheets(workbook, query.sheet)
    op = query.op
    if op not in ("count", "list", "sum", "avg", "min", "max", "group", "distinct", "sheets"):
        raise TableError("这个问题不是对表格的统计或筛选。")
    # Without a named sheet, use the sheets that have the columns the question names.
    if not query.sheet.strip():
        needed = [name for name in (query.column, query.group_by, *(c.column for c in query.where)) if name.strip()]
        usable = [sheet for sheet in sheets if all(_find_column(sheet, name) is not None for name in needed)]
        if needed and not usable:
            raise TableError(f"没有工作表同时包含：{'、'.join(dict.fromkeys(needed))}。")
        sheets = usable or sheets
    results: list[SheetResult] = []
    for sheet in sheets:
        results.append(_run_on_sheet(query, sheet))
    return TableResult(
        op, query, results, hidden=workbook.hidden,
        totals_left_out=sum(len(sheet.total_rows) for sheet in sheets),
    )


def _run_on_sheet(query: TableQuery, sheet: Sheet) -> SheetResult:
    op = query.op
    rows = _selected(sheet, query.where)
    result = SheetResult(sheet.name, [number for number, _ in rows])
    index = sheet.index_column
    if index is not None:
        result.no_index = sum(1 for _, row in rows if row[index] is None)
    if op in ("count", "sheets"):
        result.value = float(len(rows))
    elif op == "list":
        shown = _shown_columns(sheet, query, rows)
        ordered = _ordered(sheet, query, rows)
        result.listed = [
            (number, [(sheet.columns[i], _text(row[i])[:CELL_CHARS]) for i in shown if row[i] is not None])
            for number, row in ordered[: query.limit]
        ]
    elif op in ("sum", "avg", "min", "max"):
        column = _find_column(sheet, query.column)
        if column is None:
            raise TableError(f"工作表“{sheet.name.strip()}”没有“{query.column}”这一列。")
        result.column = sheet.columns[column]
        values: list[float] = []
        for number, row in rows:
            parsed = number_of(row[column])
            if parsed is None:
                if row[column] is not None:
                    result.skipped.append(number)
            else:
                values.append(parsed)
        result.value = _aggregate(op, values)
    elif op == "group":
        column = _find_column(sheet, query.group_by)
        if column is None:
            raise TableError(f"工作表“{sheet.name.strip()}”没有“{query.group_by}”这一列。")
        result.group_column = sheet.columns[column]
        target = _find_column(sheet, query.column) if query.agg != "count" else None
        if query.agg != "count" and target is None:
            raise TableError(f"工作表“{sheet.name.strip()}”没有“{query.column}”这一列。")
        buckets: dict[str, list[tuple[int, list[Any]]]] = {}
        for number, row in rows:
            label = _text(row[column]) or "（空）"
            buckets.setdefault(label, []).append((number, row))
        for label, members in buckets.items():
            numbers = [number for number, _ in members]
            if query.agg == "count":
                value: float | None = float(len(members))
            else:
                parsed = [number_of(row[target]) for _, row in members]
                value = _aggregate(query.agg, [v for v in parsed if v is not None])
                result.skipped += [n for (n, row), v in zip(members, parsed) if v is None and row[target] is not None]
            result.groups.append((label, value if value is not None else 0.0, numbers))
        result.groups.sort(key=lambda item: -item[1])
        result.groups = result.groups[:MAX_GROUPS]
    elif op == "distinct":
        column = _find_column(sheet, query.column)
        if column is None:
            raise TableError(f"工作表“{sheet.name.strip()}”没有“{query.column}”这一列。")
        result.column = sheet.columns[column]
        buckets_distinct: dict[str, list[int]] = {}
        for number, row in rows:
            if row[column] is not None:
                buckets_distinct.setdefault(_text(row[column]).replace("\n", " ")[:CELL_CHARS], []).append(number)
        result.groups = [(label, float(len(numbers)), numbers) for label, numbers in buckets_distinct.items()][:MAX_GROUPS]
    return result


def _ordered(sheet: Sheet, query: TableQuery, rows: list[tuple[int, list[Any]]]) -> list[tuple[int, list[Any]]]:
    column = _find_column(sheet, query.order_by)
    if column is None:
        return rows
    def key(item: tuple[int, list[Any]]):
        value = item[1][column]
        number = number_of(value)
        return (number is None, -number if number is not None and query.descending else (number or 0), _text(value))
    return sorted(rows, key=key)


# ---------------------------------------------------------------- the answer


def _unit(question: str) -> str:
    match = re.search(r"(?:多少|几)\s*(条|行|项|个|家|人)", question)
    return match.group(1) if match else "条"


def _cite(source: str, locator_list: list[str]) -> str:
    return "".join(f" [来源：{source}｜位置：{locator}]" for locator in locator_list)


def _condition_text(where: list[TableCondition]) -> str:
    words = {
        "eq": "为", "ne": "不为", "contains": "包含", "not_contains": "不包含", "gt": "大于", "gte": "不小于",
        "lt": "小于", "lte": "不大于", "empty": "为空", "not_empty": "不为空",
    }
    parts = []
    for condition in where:
        name = condition.column.strip() or "任一列"
        value = f"“{condition.value}”" if condition.op not in ("empty", "not_empty") else ""
        parts.append(f"{name}{words[condition.op]}{value}")
    return "，".join(parts)


def answer_text(result: TableResult, source: str, question: str = "") -> tuple[str, list[dict[str, str]]]:
    """The answer and its citations, written only from the computed result."""
    unit = _unit(question)
    condition = _condition_text(result.query.where)
    prefix = f"{condition}的" if condition else ""
    lines: list[str] = []
    cited: list[str] = []

    def cite(sheet: SheetResult, numbers: list[int] | None = None) -> str:
        found = locators(sheet.sheet, sheet.rows if numbers is None else numbers)
        cited.extend(found)
        return _cite(source, found)

    many = len(result.sheets) > 1
    op = result.op
    name = sheet_label = result.sheets[0].sheet.strip() if len(result.sheets) == 1 else ""
    where = f"「{sheet_label}」" if name else ""
    if op == "sheets":
        lines.append(f"共 {len(result.sheets)} 个工作表：")
        for sheet in result.sheets:
            lines.append(f"- {sheet.sheet.strip()}：{len(sheet.rows)} {unit}{cite(sheet)}")
    elif op == "count":
        total = result.total_rows
        if many:
            lines.append(f"{prefix}共 {total} {unit}（{len(result.sheets)} 个工作表，按数据行统计）：")
            for sheet in result.sheets:
                # With a condition, sheets where nothing matched add nothing to the answer.
                if sheet.rows or not condition:
                    lines.append(f"- {sheet.sheet.strip()}：{len(sheet.rows)} {unit}{cite(sheet)}")
        else:
            lines.append(f"{where}{prefix}共 {total} {unit}（按数据行统计）。{cite(result.sheets[0])}")
        missing = sum(sheet.no_index for sheet in result.sheets)
        if missing and total:
            lines.append(f"其中 {missing} {unit}没有填序号，也算在内。")
    elif op == "list":
        total = result.total_rows
        if not total:
            lines.append(f"{where}没有{prefix or '符合条件的'}行。")
        else:
            lines.append(f"{where}{prefix}共 {total} {unit}：")
            count = 0
            for sheet in result.sheets:
                if many and sheet.rows:
                    lines.append(f"**{sheet.sheet.strip()}**")
                for number, cells in sheet.listed:
                    count += 1
                    body = "；".join(f"{key.rsplit(' / ', 1)[-1]}：{value}" for key, value in cells) or "（空）"
                    lines.append(f"{count}. {body}{cite(sheet, [number])}")
            if count < total:
                lines.append(f"只列出前 {count} {unit}，其余的请打开原表查看。")
    elif op in ("sum", "avg", "min", "max"):
        word = {"sum": "合计", "avg": "平均", "min": "最小", "max": "最大"}[op]
        for sheet in result.sheets:
            label = f"「{sheet.sheet.strip()}」" if many or name else ""
            column = sheet.column.rsplit(" / ", 1)[-1]
            if sheet.value is None:
                lines.append(f"{label}{column}这一列没有可以计算的数字。{cite(sheet)}")
                continue
            used = len(sheet.rows) - len(sheet.skipped)
            lines.append(f"{label}{prefix}{column}{word}：{_format_number(sheet.value)}（{used} 行参与计算）。{cite(sheet)}")
            if sheet.skipped:
                head = "、".join(str(n) for n in sheet.skipped[:8])
                more = " 等" if len(sheet.skipped) > 8 else ""
                lines.append(f"第 {head}{more} 行不是纯数字，没有计入（共 {len(sheet.skipped)} 行）。")
    elif op == "group":
        agg = {"count": "", "sum": "合计 ", "avg": "平均 ", "min": "最小 ", "max": "最大 "}[result.query.agg]
        for sheet in result.sheets:
            label = sheet.sheet.strip()
            column = sheet.group_column.rsplit(" / ", 1)[-1]
            lines.append(f"「{label}」{prefix}按{column}统计（共 {len(sheet.rows)} {unit}）：")
            for group, value, numbers in sheet.groups:
                shown = f"{_format_number(value)} {unit}" if result.query.agg == "count" else f"{agg}{_format_number(value)}"
                lines.append(f"- {group}：{shown}{cite(sheet, numbers)}")
            if sheet.skipped:
                lines.append(f"有 {len(sheet.skipped)} 行不是纯数字，没有计入。")
    elif op == "distinct":
        for sheet in result.sheets:
            label = sheet.sheet.strip()
            column = sheet.column.rsplit(" / ", 1)[-1]
            lines.append(f"「{label}」{prefix}的{column}有 {len(sheet.groups)} 种：")
            for value, _, numbers in sheet.groups:
                lines.append(f"- {value}{cite(sheet, numbers)}")
    if result.totals_left_out:
        lines.append(f"合计行（{result.totals_left_out} 行）没有算作数据。")
    if result.hidden:
        lines.append(f"隐藏的工作表（{'、'.join(result.hidden)}）没有统计。")
    citations = [{"source": source, "locator": locator} for locator in dict.fromkeys(cited)]
    return "\n".join(lines), citations


def is_table_file(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() == ".xlsx"
