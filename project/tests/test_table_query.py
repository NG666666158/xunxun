"""Read-only table questions: real-workbook shapes, every operation, and what is refused."""

from openpyxl import Workbook as OpenWorkbook
from pydantic import ValidationError
import pytest

from services.table_query import (
    Sheet,
    TableError,
    TableQuery,
    Workbook,
    WorkbookCache,
    answer_text,
    locators,
    number_of,
    read_workbook,
    row_ranges,
    run_query,
    schema_text,
    sheet_asked_for,
)


def policy_book(path):
    """Like the real policy workbook: a merged title, a merged category column, a missing 序号."""
    book = OpenWorkbook()
    sheet = book.active
    sheet.title = "普惠政策"
    sheet.append(["政策汇总\n* 摘录", None, None, None])
    sheet.merge_cells("A1:D1")
    sheet.append(["序号", "类别", "政策方向", "金额（万元）"])
    sheet.append([1, "税收", "企业所得税", 100])
    sheet.append([2, None, "个人所得税", "250"])
    sheet.merge_cells("B3:B4")
    sheet.append([None, "人才", "人才工作站", "30万"])
    sheet.append([4, "人才", "博士后", "待定"])
    sheet.append([None, None, None, None])
    sheet.append(["合计", None, None, 380])
    other = book.create_sheet("科技研发 ")
    other.append(["序号", "类别", "政策方向"])
    other.append([1, "科技", "研发补贴"])
    other.append([2, "科技", "知识产权"])
    book.save(path)
    return path


@pytest.fixture
def workbook(tmp_path):
    return read_workbook(policy_book(tmp_path / "汇总.xlsx"))


def query(**fields):
    return TableQuery(**fields)


def test_title_rows_are_skipped_and_merged_categories_filled_down(workbook):
    sheet = workbook.sheets[0]
    assert sheet.header_row == 2
    assert sheet.columns == ["序号", "类别", "政策方向", "金额（万元）"]
    assert [number for number, _ in sheet.rows] == [3, 4, 5, 6]
    assert sheet.rows[1][1][1] == "税收"  # B4 is covered by the merge


def test_total_rows_and_blank_rows_are_not_data(workbook):
    sheet = workbook.sheets[0]
    assert sheet.total_rows == [8]
    assert all(row[0] != "合计" for _, row in sheet.rows)


def test_count_across_sheets_cites_each_sheet_and_says_how_it_counted(workbook, tmp_path):
    result = run_query(query(op="count"), workbook)
    assert result.total_rows == 6
    text, citations = answer_text(result, "汇总.xlsx", "一共多少条")
    assert "共 6 条" in text and "按数据行统计" in text
    assert [c["locator"] for c in citations] == ["sheet:普惠政策;rows:3-6", "sheet:科技研发 ;rows:2-3"]
    # The rows without a 序号 are counted, and the answer says so.
    assert "2 条没有填序号" not in text and "1 条没有填序号" in text
    assert "合计行" in text


def test_count_with_a_condition_on_a_merged_category(workbook):
    result = run_query(query(op="count", sheet="普惠政策", where=[{"column": "类别", "op": "eq", "value": "税收"}]), workbook)
    assert result.total_rows == 2
    text, citations = answer_text(result, "汇总.xlsx")
    assert "类别为“税收”的共 2 条" in text
    assert citations == [{"source": "汇总.xlsx", "locator": "sheet:普惠政策;rows:3-4"}]


def test_sheets_where_nothing_matched_are_left_out_of_the_answer(workbook):
    result = run_query(query(op="count", where=[{"column": "政策方向", "op": "contains", "value": "研发"}]), workbook)
    text, _ = answer_text(result, "汇总.xlsx")
    assert "科技研发：1 条" in text and "普惠政策" not in text


def test_sheet_names_match_loosely_and_unknown_ones_list_what_exists(workbook):
    assert run_query(query(op="count", sheet="科技研发"), workbook).total_rows == 2
    assert run_query(query(op="count", sheet="科技"), workbook).total_rows == 2
    with pytest.raises(TableError, match="普惠政策、科技研发"):
        run_query(query(op="count", sheet="不存在"), workbook)


def test_no_sheet_named_uses_the_sheets_with_the_columns(workbook):
    result = run_query(query(op="sum", column="金额"), workbook)
    assert [sheet.sheet for sheet in result.sheets] == ["普惠政策"]
    with pytest.raises(TableError, match="没有工作表"):
        run_query(query(op="sum", column="不存在的列"), workbook)


def test_sum_skips_what_is_not_a_number_and_says_which_rows(workbook):
    result = run_query(query(op="sum", sheet="普惠政策", column="金额"), workbook)
    sheet = result.sheets[0]
    # 100 + 250 + 30万 = 300,350; 待定 is not a number; the 合计 row is not data.
    assert sheet.value == 300350
    assert sheet.skipped == [6]
    text, _ = answer_text(result, "汇总.xlsx")
    assert "300350" in text and "3 行参与计算" in text and "第 6 行不是纯数字" in text


@pytest.mark.parametrize("op,expected", [("avg", 300350 / 3), ("min", 100), ("max", 300000)])
def test_average_minimum_maximum(workbook, op, expected):
    assert run_query(query(op=op, sheet="普惠政策", column="金额"), workbook).sheets[0].value == pytest.approx(expected)


def test_a_column_of_words_has_nothing_to_sum(workbook):
    result = run_query(query(op="sum", sheet="普惠政策", column="政策方向"), workbook)
    assert result.sheets[0].value is None
    assert "没有可以计算的数字" in answer_text(result, "汇总.xlsx")[0]


def test_group_counts_sorted_with_the_rows_of_each_group(workbook):
    result = run_query(query(op="group", sheet="普惠政策", group_by="类别"), workbook)
    assert [(label, value) for label, value, _ in result.sheets[0].groups] == [("税收", 2), ("人才", 2)]
    text, citations = answer_text(result, "汇总.xlsx", "按类别分别有多少条")
    assert "- 税收：2 条" in text and "- 人才：2 条" in text
    assert {"source": "汇总.xlsx", "locator": "sheet:普惠政策;rows:5-6"} in citations


def test_group_can_aggregate_another_column(workbook):
    result = run_query(query(op="group", sheet="普惠政策", group_by="类别", column="金额", agg="sum"), workbook)
    assert dict((label, value) for label, value, _ in result.sheets[0].groups) == {"税收": 350, "人才": 300000}


def test_distinct_lists_each_value_once(workbook):
    result = run_query(query(op="distinct", sheet="普惠政策", column="类别"), workbook)
    text, _ = answer_text(result, "汇总.xlsx")
    assert "有 2 种" in text and "- 税收" in text and "- 人才" in text


def test_list_shows_the_named_columns_and_cites_each_row(workbook):
    result = run_query(query(
        op="list", sheet="普惠政策", where=[{"column": "类别", "op": "eq", "value": "人才"}], columns=["政策方向"],
    ), workbook)
    text, citations = answer_text(result, "汇总.xlsx")
    assert "共 2 条" in text and "政策方向：人才工作站" in text and "政策方向：博士后" in text
    assert [c["locator"] for c in citations] == ["sheet:普惠政策;rows:5-5", "sheet:普惠政策;rows:6-6"]


def test_list_says_when_it_shows_only_part(workbook):
    result = run_query(query(op="list", sheet="普惠政策", limit=1), workbook)
    assert "只列出前 1 条" in answer_text(result, "汇总.xlsx")[0]


def test_list_can_order_by_a_number(workbook):
    result = run_query(query(op="list", sheet="普惠政策", columns=["政策方向"], order_by="金额", descending=True, limit=1), workbook)
    assert result.sheets[0].listed[0][1] == [("政策方向", "人才工作站")]


def test_sheets_op_names_every_sheet(workbook):
    text, _ = answer_text(run_query(query(op="sheets"), workbook), "汇总.xlsx")
    assert "共 2 个工作表" in text and "普惠政策：4 条" in text and "科技研发：2 条" in text


# The real policy workbook's sheets, in its order (doc_capabilities_v1: 7 sheets, 104 rows).
POLICY_SHEETS = Workbook([Sheet(name, 1, ["序号", "类别", "政策措施"]) for name in (
    "普惠政策", "近期出台普惠政策 ", "科技研发", "生物医药", "集成电路", "跨境电商", "商贸",
)])


@pytest.mark.parametrize("question,sheet,expected", [
    # No sheet named: Qwen3-8B filled in the first one and counted 26 of 104 (T01, T07, T21).
    ("《0.政策补贴汇总》这张表里一共列了多少条政策？", "普惠政策", ""),
    ("这张表一共多少条？", "普惠政策", ""),
    ("《0.政策补贴汇总》里，政策措施提到“租金”的一共有多少条？", "普惠政策", ""),
    ("《0.政策补贴汇总》一共多少条？", "科技研发", ""),
    # A loose name that stands for sheets the question doesn't name (both 普惠政策 sheets).
    ("《0.政策补贴汇总》这张表里一共列了多少条政策？", "政策", ""),
    # The workbook's own name where a sheet's goes, with or without its number.
    ("《0.政策补贴汇总》一共多少条？", "0.政策补贴汇总", ""),
    ("政策补贴汇总这个表一共多少条？", "政策补贴汇总", ""),
    # A sheet the question names stays, also when written a little differently.
    ("《0.政策补贴汇总》的科技研发工作表有多少条政策？", "科技研发", "科技研发"),
    ("《0.政策补贴汇总》普惠政策里，类别是税收政策的有几条？", "普惠政策", "普惠政策"),
    ("《0.政策补贴汇总》生物医药这张表有多少条？", "生物医药", "生物医药"),
    ("近期出台的普惠政策有多少条？", "近期出台普惠政策", "近期出台普惠政策"),
    ("《0.政策补贴汇总》里《跨境电商》有几条？", "跨境电商", "跨境电商"),
    # Named by its place.
    ("《0.政策补贴汇总》第二个工作表有多少条？", "近期出台普惠政策", "近期出台普惠政策"),
    ("《0.政策补贴汇总》最后一张表有多少条？", "商贸", "商贸"),
    # A sheet that isn't there: kept when asked for by name, so the answer can say so.
    ("《0.政策补贴汇总》的财务表有多少条？", "财务", "财务"),
    ("《0.政策补贴汇总》一共多少条？", "财务", ""),
    ("《0.政策补贴汇总》一共多少条？", "", ""),
])
def test_a_sheet_counts_only_when_the_question_names_it(question, sheet, expected):
    assert sheet_asked_for(question, sheet, POLICY_SHEETS, "0.政策补贴汇总.xlsx") == expected


def test_the_whole_workbook_is_counted_sheet_by_sheet_when_none_is_named(workbook):
    sheet = sheet_asked_for("《汇总》这张表一共多少条？", "普惠政策", workbook, "汇总.xlsx")
    text, citations = answer_text(run_query(query(op="count", sheet=sheet), workbook), "汇总.xlsx", "一共多少条")
    assert "共 6 条（2 个工作表" in text
    assert "- 普惠政策：4 条" in text and "- 科技研发：2 条" in text
    assert [c["locator"] for c in citations] == ["sheet:普惠政策;rows:3-6", "sheet:科技研发 ;rows:2-3"]
    # Asked about one sheet, only that one is counted.
    sheet = sheet_asked_for("《汇总》科技研发有多少条？", "科技研发", workbook, "汇总.xlsx")
    text, _ = answer_text(run_query(query(op="count", sheet=sheet), workbook), "汇总.xlsx", "有多少条")
    assert text.startswith("「科技研发」共 2 条")


@pytest.mark.parametrize("condition,count", [
    ({"column": "序号", "op": "gt", "value": "1"}, 2),
    ({"column": "序号", "op": "lte", "value": "2"}, 2),
    ({"column": "序号", "op": "empty"}, 1),
    ({"column": "序号", "op": "not_empty"}, 3),
    ({"column": "类别", "op": "ne", "value": "税收"}, 2),
    ({"column": "政策方向", "op": "not_contains", "value": "所得税"}, 2),
    ({"column": "", "op": "contains", "value": "博士后"}, 1),
])
def test_condition_operators(workbook, condition, count):
    assert run_query(query(op="count", sheet="普惠政策", where=[condition]), workbook).total_rows == count


def test_header_spanning_several_rows_and_placeholder_rows(tmp_path):
    """Like the real collection template: two header rows, a label merged across, "……" rows."""
    book = OpenWorkbook()
    sheet = book.active
    sheet.append(["标题", None, None, None])
    sheet.merge_cells("A1:D1")
    sheet.append(["方向", None, "精密级构筑", None])
    sheet.merge_cells("A2:B2")
    sheet.merge_cells("C2:D2")
    sheet.append(["序号", "层级", "品类", "单位"])
    sheet.append([1, "原材料", "高纯气体", "A 公司"])
    sheet.append(["......", None, "......", None])
    path = tmp_path / "素材.xlsx"
    book.save(path)
    parsed = read_workbook(path).sheets[0]
    assert parsed.columns == ["方向 / 序号", "方向 / 层级", "精密级构筑 / 品类", "精密级构筑 / 单位"]
    assert len(parsed.rows) == 1
    assert run_query(query(op="count", where=[{"column": "层级", "op": "eq", "value": "原材料"}]), read_workbook(path)).total_rows == 1


def test_hidden_sheets_are_not_counted_and_the_answer_says_so(tmp_path):
    book = OpenWorkbook()
    book.active.append(["序号", "名称"])
    book.active.append([1, "可见"])
    hidden = book.create_sheet("辅助")
    hidden.append(["序号", "名称"])
    hidden.append([1, "隐藏"])
    hidden.sheet_state = "hidden"
    path = tmp_path / "h.xlsx"
    book.save(path)
    result = run_query(query(op="count"), read_workbook(path))
    assert result.total_rows == 1
    assert "隐藏的工作表（辅助）" in answer_text(result, "h.xlsx")[0]


def test_a_sheet_without_a_header_is_skipped(tmp_path):
    book = OpenWorkbook()
    book.active.append([1, 2, 3])
    path = tmp_path / "n.xlsx"
    book.save(path)
    assert read_workbook(path).sheets == []


def test_only_the_fixed_operations_exist():
    for bad in ("delete", "update", "exec", "sql", "drop"):
        with pytest.raises(ValidationError):
            TableQuery(op=bad)
    with pytest.raises(ValidationError):
        TableQuery(op="count", where=[{"column": "a", "op": "regex", "value": "x"}])
    with pytest.raises(ValidationError):
        TableQuery(op="list", limit=100000)


def test_nothing_in_a_cell_or_a_value_is_evaluated(tmp_path):
    book = OpenWorkbook()
    book.active.append(["名称", "备注"])
    book.active.append(["__import__('os')", "=1+1"])
    path = tmp_path / "e.xlsx"
    book.save(path)
    workbook = read_workbook(path)
    condition = {"column": "名称", "op": "contains", "value": "__import__('os')"}
    assert run_query(query(op="count", where=[condition]), workbook).total_rows == 1


def test_the_file_is_not_changed(tmp_path):
    path = policy_book(tmp_path / "汇总.xlsx")
    before = path.read_bytes(), path.stat().st_mtime_ns
    run_query(query(op="group", sheet="普惠政策", group_by="类别"), read_workbook(path))
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_schema_gives_columns_kinds_and_values(workbook):
    text = schema_text(workbook)
    assert "工作表「普惠政策」，4 行数据" in text
    assert "类别（文本，取值：税收、人才）" in text
    assert "序号（数字" in text


@pytest.mark.parametrize("value,expected", [
    (12, 12.0), (3.5, 3.5), ("1,200", 1200.0), ("30%", 30.0), ("3.5 万元", 35000.0), ("2亿", 2e8), ("100元", 100.0),
    ("待定", None), ("最高100万", None), ("", None), (None, None), (True, None),
])
def test_numbers_in_cells(value, expected):
    assert number_of(value) == expected


def test_row_ranges_and_locators():
    assert row_ranges([5, 3, 4, 9]) == [(3, 5), (9, 9)]
    assert locators("A", [3, 4, 9]) == ["sheet:A;rows:3-4", "sheet:A;rows:9-9"]


def test_cache_reads_a_file_once_and_again_after_it_changes(tmp_path, monkeypatch):
    path = policy_book(tmp_path / "汇总.xlsx")
    calls = []
    import services.table_query as module

    real = module.read_workbook
    monkeypatch.setattr(module, "read_workbook", lambda p: calls.append(p) or real(p))
    cache = WorkbookCache()
    assert cache.get(path) is cache.get(path)
    assert len(calls) == 1
    policy_book(path)
    import os

    os.utime(path, ns=(1, 1))
    cache.get(path)
    assert len(calls) == 2


def test_cache_does_not_keep_a_very_big_workbook(tmp_path):
    path = policy_book(tmp_path / "汇总.xlsx")
    cache = WorkbookCache(max_rows=1)
    first = cache.get(path)
    assert cache.get(path) is not first
