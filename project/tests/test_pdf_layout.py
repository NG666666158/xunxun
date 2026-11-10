"""Layout-aware PDF parsing and page-precise chunking, on small generated PDFs."""

from pathlib import Path

import pymupdf
import pytest

from ingestion.chunker import HeadingAwareChunker
from ingestion.models import DocumentUnit, ParsedDocument
from ingestion.parsers import MultiFormatParser


FONT = "china-s"
BODY = 10
LEFT, RIGHT = 57, 537
LINE = 15


class ParserSettings:
    max_upload_mb = 5
    max_local_file_mb = 5
    max_local_text_mb = 5


class ChunkSettings:
    parent_chunk_size = 120
    child_chunk_size = 50
    child_chunk_overlap = 15


def _wrap(text: str, width: float, size: float = BODY, font: str = FONT) -> list[str]:
    """Greedy hard wrap, the way a typesetter breaks lines."""
    lines, current = [], ""
    for char in text:
        if pymupdf.get_text_length(current + char, fontname=font, fontsize=size) > width:
            lines.append(current)
            current = char
        else:
            current += char
    if current:
        lines.append(current)
    return lines


class Book:
    """Writes pages with a running header, a page-number footer and hard-wrapped text."""

    def __init__(self, header: str = "示例手册　内部资料"):
        self.doc = pymupdf.open()
        self.header = header
        self.page = None
        self.y = 0.0

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=595, height=842)
        number = self.doc.page_count
        self.page.insert_text((LEFT, 42), self.header, fontname=FONT, fontsize=9)
        self.page.insert_text((290, 812), str(number), fontname="helv", fontsize=9)
        self.y = 90

    def heading(self, text: str, size: float) -> None:
        self.y += size
        self.page.insert_text((LEFT, self.y), text, fontname=FONT, fontsize=size)
        self.y += size

    def paragraph(self, text: str, indent: bool = True, gap: float = 9) -> None:
        """A paragraph; it continues on a new page when the page is full."""
        first = True
        remaining = text
        while remaining:
            x = LEFT + (2 * BODY if first and indent else 0)
            line = _wrap(remaining, RIGHT - x)[0]
            if self.y > 760:
                self.new_page()
            self.y += LINE
            self.page.insert_text((x, self.y), line, fontname=FONT, fontsize=BODY)
            remaining = remaining[len(line):]
            first = False
        self.y += gap

    def line(self, text: str, x: float = LEFT, size: float = BODY, font: str = FONT) -> None:
        self.y += LINE
        self.page.insert_text((x, self.y), text, fontname=font, fontsize=size)

    def save(self, path: Path, toc=None) -> Path:
        if toc:
            self.doc.set_toc(toc)
        self.doc.save(path)
        self.doc.close()
        return path


LONG = (
    "审批流程分为三个环节，申请人首先在系统中填写申请单并上传附件，然后由直属负责人进行初审，"
    "初审通过后转交财务部门复核，复核完成后系统会自动通知申请人领取结果，整个流程通常需要三个工作日"
    "才能全部走完，遇到节假日顺延，因此请务必提前规划好时间，避免影响后续的报销与付款安排。"
)
FIRST = "本制度适用于公司全体员工的差旅与报销活动。员工出差前应当提交申请，并在返回后五个工作日内完成报销。"
SECOND = "差旅标准按照职级划分，具体标准见附件一。超出标准的部分需要部门负责人另行审批，并说明理由。"


def _handbook(tmp_path: Path, with_outline: bool = True) -> Path:
    book = Book()
    book.new_page()
    book.heading("第1章　总则", 18)
    book.heading("1.1　适用范围", 13)
    book.paragraph(FIRST)
    book.paragraph(SECOND)
    book.heading("1.2　审批流程", 13)
    for _ in range(21):
        book.paragraph("这是用于占位的说明文字，内容用于把页面填满，以便测试跨页段落的合并效果。")
    book.paragraph(LONG)  # starts near the bottom of page 1 and runs onto page 2
    book.paragraph("审批结果以系统通知为准。")
    toc = [[1, "第1章 总则", 1], [2, "1.1 适用范围", 1], [2, "1.2 审批流程", 1]] if with_outline else None
    return book.save(tmp_path / ("handbook.pdf" if with_outline else "handbook-no-outline.pdf"), toc)


def _parse(path: Path):
    return MultiFormatParser(ParserSettings()).parse(path, doc_id="a" * 32)


def test_running_headers_and_page_numbers_are_removed(tmp_path):
    parsed = _parse(_handbook(tmp_path))
    text = "\n".join(unit.text for unit in parsed.units)
    assert "示例手册" not in text
    assert not any(line.strip().isdigit() for line in text.splitlines())
    assert parsed.metadata["running_rows_removed"] >= 2 * parsed.metadata["pages"]


def test_hard_wrapped_lines_become_paragraphs(tmp_path):
    parsed = _parse(_handbook(tmp_path))
    section = next(unit for unit in parsed.units if unit.heading.endswith("1.1 适用范围"))
    # Each paragraph comes back whole, separated by a blank line.
    assert section.text.split("\n\n")[-3:] == ["1.1　适用范围", FIRST, SECOND]


def test_paragraph_continues_across_pages_with_page_map(tmp_path):
    parsed = _parse(_handbook(tmp_path))
    section = next(unit for unit in parsed.units if unit.heading.endswith("1.2 审批流程"))
    assert parsed.metadata["pages"] == 2
    assert LONG in section.text
    assert section.locator == "page:1-2"
    page_map = section.metadata["page_map"]
    assert [page for _, page in page_map] == [1, 2]
    # The page boundary falls inside the long paragraph, where its text moved to page 2.
    start = section.text.index(LONG)
    boundary = page_map[1][0]
    assert start < boundary < start + len(LONG)
    page_two = pymupdf.open(tmp_path / "handbook.pdf")[1].get_text().replace("\n", "")
    assert section.text[boundary:boundary + 10] in page_two


def test_outline_sections_carry_full_heading_paths(tmp_path):
    parsed = _parse(_handbook(tmp_path))
    assert parsed.metadata["strategy"] == "outline"
    # The chapter has no text of its own, so its title opens the first subsection.
    assert [unit.heading for unit in parsed.units] == [
        "第1章 总则 > 1.1 适用范围",
        "第1章 总则 > 1.2 审批流程",
    ]
    assert parsed.units[0].text.startswith("第1章　总则\n\n1.1　适用范围")


def test_without_outline_headings_come_from_font_size(tmp_path):
    parsed = _parse(_handbook(tmp_path, with_outline=False))
    assert parsed.metadata["strategy"] == "headings"
    assert [unit.heading for unit in parsed.units] == [
        "第1章　总则 > 1.1　适用范围",
        "第1章　总则 > 1.2　审批流程",
    ]


def test_numbered_and_chapter_headings_without_size_change(tmp_path):
    book = Book(header="")
    book.new_page()
    book.line("第一章 概述")
    book.y += 6
    book.paragraph("概述部分的正文内容。", indent=False)
    book.line("1.2 术语定义")
    book.y += 6
    book.paragraph("术语定义部分的正文内容。", indent=False)
    path = book.save(tmp_path / "plain-headings.pdf")
    parsed = _parse(path)
    assert [unit.heading for unit in parsed.units] == ["第一章 概述", "1.2 术语定义"]


def test_plain_pages_fall_back_to_one_unit_per_page(tmp_path):
    book = Book()
    for text in ("第一页的正文内容，没有任何标题。", "第二页的正文内容，同样没有标题。"):
        book.new_page()
        book.paragraph(text)
    parsed = _parse(book.save(tmp_path / "plain.pdf"))
    assert parsed.metadata["strategy"] == "pages"
    assert [(unit.heading, unit.locator) for unit in parsed.units] == [("第 1 页", "page:1"), ("第 2 页", "page:2")]


def test_english_hyphenation_and_spacing(tmp_path):
    book = Book(header="Field Guide")
    book.new_page()
    book.heading("Overview", 16)
    for line in (
        "Retrieval quality depends heavily on how the source docu-",
        "ment is split into passages, and also on the careful imple-",
        "mentation of the ranking stage, including the state-of-the-",
        "art rerankers that the system calls for every query.",
    ):
        book.line(line, font="helv")
    path = book.save(tmp_path / "english.pdf")
    text = _parse(path).units[-1].text
    assert text.endswith(
        "Retrieval quality depends heavily on how the source document is split into passages, and also on the "
        "careful implementation of the ranking stage, including the state-of-the-art rerankers that the system "
        "calls for every query."
    )


def test_code_blocks_keep_line_breaks_and_indentation(tmp_path):
    book = Book()
    book.new_page()
    book.heading("示例代码", 16)
    book.paragraph("下面的函数计算两数之和：", indent=False)
    for depth, code in ((0, "def add(a, b):"), (1, "total = a + b"), (1, "return total")):
        book.line(code, x=LEFT + 20 + depth * 4 * 5.4, size=9, font="cour")
    book.y += 10
    book.paragraph("调用时传入两个数字即可。", indent=False)
    text = _parse(book.save(tmp_path / "code.pdf")).units[-1].text
    assert "def add(a, b):\n    total = a + b\n    return total" in text
    assert "下面的函数计算两数之和：" in text and "调用时传入两个数字即可。" in text


def test_figure_labels_are_dropped_and_captions_kept(tmp_path):
    book = Book()
    book.new_page()
    book.heading("系统结构", 16)
    book.paragraph("系统由三个部分组成，如图1-1 所示。", indent=False)
    book.y += 10
    for x, label in ((120, "LLM：大脑"), (260, "工具：手脚"), (400, "上下文：眼睛")):
        book.page.insert_text((x, book.y + 20), label, fontname=FONT, fontsize=7)
    book.page.insert_text((180, book.y + 45), "Agent", fontname="helv", fontsize=7)
    book.y += 60
    book.line("图1-1　系统的三个组成部分", x=200, size=8)
    book.y += 10
    book.paragraph("每个部分各司其职。", indent=False)
    parsed = _parse(book.save(tmp_path / "figure.pdf"))
    text = parsed.units[-1].text
    assert "图1-1　系统的三个组成部分" in text
    assert "大脑" not in text and "手脚" not in text and "Agent" not in text
    assert parsed.metadata["figure_labels_dropped"] >= 3


def test_image_only_pages_are_flagged(tmp_path):
    book = Book(header="")
    book.new_page()
    book.paragraph("第一页有文字内容。")
    page = book.doc.new_page(width=595, height=842)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pixmap.clear_with(200)
    page.insert_image(pymupdf.Rect(100, 100, 400, 400), pixmap=pixmap)
    parsed = _parse(book.save(tmp_path / "scan.pdf"))
    assert parsed.metadata["image_only_pages"] == [2]
    # One scanned page of two: flagged, so the UI asks "这份 PDF 是扫描件，要识别文字吗？".
    assert parsed.metadata["notice"] == "scanned_pdf"

    scanned = pymupdf.open()
    scanned.new_page().insert_image(pymupdf.Rect(0, 0, 300, 300), pixmap=pixmap)
    scanned.save(tmp_path / "only-scan.pdf")
    # Nothing to read without image understanding: indexed by name (findable) and flagged.
    only = _parse(tmp_path / "only-scan.pdf")
    assert only.metadata["notice"] == "scanned_pdf"
    assert "only-scan.pdf" in only.units[0].text and "扫描件" in only.units[0].text
    # Declined for this file ("off"): still findable, never asked again.
    declined = MultiFormatParser(ParserSettings()).parse(tmp_path / "only-scan.pdf", image_mode="off")
    assert "notice" not in declined.metadata


# ---------------------------------------------------------------- chunking


def test_chunks_get_exact_page_ranges_from_page_map():
    pages = ["第一页的内容在这里。" * 6, "第二页继续写内容。" * 6, "第三页是最后一页。" * 6]
    text = "".join(pages)
    page_map = [[0, 10], [len(pages[0]), 11], [len(pages[0]) + len(pages[1]), 12]]
    document = ParsedDocument(
        doc_id="b" * 32,
        source="book.pdf",
        file_type=".pdf",
        units=[DocumentUnit(text=text, heading="第 2 章", locator="page:10-12", metadata={"page_map": page_map})],
    )
    parents, children = HeadingAwareChunker(ChunkSettings()).split_parent_child(document)
    for chunk in [*parents, *children]:
        start = text.index(chunk.text)
        first = max(page for offset, page in page_map if offset <= start)
        last = max(page for offset, page in page_map if offset <= start + len(chunk.text) - 1)
        assert chunk.locator == (f"page:{first}" if first == last else f"page:{first}-{last}")
    assert {child.locator for child in children} >= {"page:10", "page:11", "page:12"}
    assert any("-" in parent.locator for parent in parents)


def test_units_without_page_map_keep_their_locator():
    document = ParsedDocument(
        doc_id="c" * 32,
        source="a.docx",
        file_type=".docx",
        units=[DocumentUnit(text="规定。" * 40, heading="制度", locator="table:1")],
    )
    parents, children = HeadingAwareChunker(ChunkSettings()).split_parent_child(document)
    assert {chunk.locator for chunk in [*parents, *children]} == {"table:1"}


def test_cut_prefers_paragraph_then_sentence_then_semicolon_then_newline():
    cut = HeadingAwareChunker._cut
    text = "甲" * 40 + "\n\n" + "乙" * 10 + "。" + "丙" * 10
    assert text[: cut(text, 20, 65)].endswith("\n\n")  # paragraph beats a later sentence end
    text = "甲" * 30 + "；" + "乙" * 10 + "。" + "丙" * 10
    assert text[: cut(text, 20, 55)].endswith("。")
    text = "甲" * 30 + "\n" + "乙" * 10 + "；" + "丙" * 10
    assert text[: cut(text, 20, 55)].endswith("；")
    text = "甲" * 30 + "\n" + "乙" * 30
    assert text[: cut(text, 20, 55)].endswith("\n")
    # "3.14" and "file.py" are not sentence ends.
    text = "版本号是 3.14 和 file.py 两个" + "字" * 40
    assert cut(text, 5, 40) is None


def test_children_never_start_or_end_mid_sentence():
    sentences = [f"第{i}句话描述一个独立的事实，长度适中。" for i in range(12)]
    text = "".join(sentences)
    for child in HeadingAwareChunker._windows(text, 50, 15):
        assert child.endswith("。") or text.endswith(child)
        assert any(child.startswith(sentence) for sentence in sentences)


def test_term_line_stays_with_its_definition():
    text = "以下参数值得注意。\n\n" + "说明" * 12 + "。\n\nmax_new_tokens\n\n允许模型生成的最大词元数。" + "补充" * 20
    end = HeadingAwareChunker._cut(text, 10, 60)
    assert text[:end].rstrip().endswith("说明。")  # not after the bare term "max_new_tokens"


def test_question_lists_get_one_child_per_question():
    questions = [f"Q{i}： 第{i}个关于{topic}的面试问题是什么？" for i, topic in enumerate(["KV 缓存", "分词", "嵌入", "RAG", "微调"], 1)]
    text = "第3 章　LLM 的内部机制\n\n" + "\n\n".join(questions)
    document = ParsedDocument(
        doc_id="d" * 32, source="面试题.pdf", file_type=".pdf",
        units=[DocumentUnit(text=text, heading="面试题 > 第3 章", locator="page:2", metadata={"page_map": [[0, 2]]})],
    )
    settings = type("S", (), {"parent_chunk_size": 1800, "child_chunk_size": 700, "child_chunk_overlap": 120})()
    parents, children = HeadingAwareChunker(settings).split_parent_child(document)
    assert len(parents) == 1
    assert [child.text for child in children] == ["第3 章　LLM 的内部机制\n\n" + questions[0], *questions[1:]]
    # Ordinary prose with a few numbered lines is still windowed as before.
    assert HeadingAwareChunker._item_spans("正文。\n\nQ1： 一个问题？\n\n正文。\n\n正文。", 700) is None


def test_index_version_combines_parser_chunker_and_image_mode():
    from types import SimpleNamespace

    from ingestion.chunker import CHUNKER_VERSION
    from ingestion.parsers import parser_version
    from ingestion.pdf_layout import PDF_PARSER_VERSION

    off, local, cloud = (SimpleNamespace(image_understanding=mode) for mode in ("off", "local", "cloud"))
    assert parser_version("a.pdf", off) == PDF_PARSER_VERSION * 1000 + CHUNKER_VERSION * 10
    assert len({parser_version("a.pdf", s) for s in (off, local, cloud)}) == 3  # switching modes re-indexes PDFs
    formulas = SimpleNamespace(image_understanding="local", local_formula=True)
    assert parser_version("a.pdf", formulas) not in {parser_version("a.pdf", s) for s in (off, local, cloud)}
    assert parser_version("a.docx", cloud) == 1000 + CHUNKER_VERSION * 10
    assert parser_version("a.pdf", off) > 303  # registries from before hold smaller numbers


def test_children_carry_document_and_section_prefix_parents_do_not():
    from ingestion.chunker import context_prefix

    settings = type("S", (), {"parent_chunk_size": 1800, "child_chunk_size": 700, "child_chunk_overlap": 120,
                              "chunk_context_prefix": True})()
    document = ParsedDocument(
        doc_id="e" * 32, source="【配套资料】大模型面试题200问.pdf", file_type=".pdf",
        units=[DocumentUnit(text="Q1： 问题一？\n\nQ2： 问题二？\n\nQ3： 问题三？", heading="大模型面试题200 问 > 第3 章 LLM 的内部机制")],
    )
    parents, children = HeadingAwareChunker(settings).split_parent_child(document)
    assert children[0].text == "【大模型面试题200问 · 大模型面试题200 问 > 第3 章 LLM 的内部机制】\nQ1： 问题一？"
    assert not parents[0].text.startswith("【")
    assert context_prefix("a.pdf", "第 3 页") == "【a】\n"
    assert context_prefix("制度.docx", "") == "【制度】\n"


def test_ligatures_and_math_letters_are_repaired_but_chinese_punctuation_is_kept():
    from ingestion.pdf_layout import _repair

    assert _repair("ﬁne-tuning 的 𝑑ₖ 维度，见\ue001表：") == "fine-tuning 的 dₖ 维度，见表："


# ---------------------------------------------------------------- figures


class FakeReader:
    mode = "cloud"
    model = "fake"
    workers = 2

    def __init__(self):
        self.seen = []

    def read(self, figure):
        self.seen.append(figure)
        return "整页文字" if figure.kind == "page" else f"一张示意图，标签：{'、'.join(figure.labels)}"


def _figure_pdf(tmp_path: Path) -> Path:
    book = Book()
    book.new_page()
    book.heading("系统结构", 16)
    book.paragraph("系统由三个部分组成，如图1-1 所示，下面分别说明它们的职责与协作方式，以及如何扩展。", indent=False)
    top = book.y + 8
    book.page.draw_rect(pymupdf.Rect(100, top, 480, top + 90), color=(0, 0, 0), width=1)
    for x, label in ((120, "LLM：大脑"), (260, "工具：手脚"), (400, "上下文")):
        book.page.draw_rect(pymupdf.Rect(x - 5, top + 20, x + 60, top + 45), color=(0, 0, 0))
        book.page.insert_text((x, top + 36), label, fontname=FONT, fontsize=7)
    book.y = top + 100
    book.line("图1-1　系统的三个组成部分", x=200, size=8)
    book.y += 10
    book.paragraph("每个部分各司其职。", indent=False)
    return book.save(tmp_path / "figure-box.pdf")


def test_figures_are_read_and_placed_after_their_caption(tmp_path):
    from ingestion.pdf_layout import parse_pdf

    reader = FakeReader()
    units, metadata = parse_pdf(_figure_pdf(tmp_path), reader)
    text = units[-1].text
    assert metadata["figures_found"] == 1 and metadata["figures_read"] == 1
    figure = reader.seen[0]
    assert figure.caption.startswith("图1-1") and figure.labels == ["LLM：大脑", "工具：手脚", "上下文"]
    assert figure.image.startswith(b"\x89PNG")
    caption = text.index("图1-1　系统的三个组成部分")
    assert text.index("〔图片说明〕一张示意图，标签：LLM：大脑、工具：手脚、上下文") > caption
    assert text.endswith("每个部分各司其职。")


def test_local_mode_uses_text_layer_labels_without_ocr(tmp_path):
    from ingestion.figures import Figure, LocalReader

    reader = LocalReader()
    figure = Figure(page=1, bbox=(0, 0, 10, 10), kind="figure", caption="图1-1 结构", labels=["Cat", "Cute", "0.74", "Cat"])
    assert reader.read(figure) == "Cat、Cute、0.74"


def test_scanned_pdf_is_read_page_by_page_with_local_ocr(tmp_path):
    from ingestion.figures import LocalReader
    from ingestion.pdf_layout import parse_pdf

    source = pymupdf.open()
    page = source.new_page(width=420, height=300)
    page.insert_text((40, 80), "扫描件第一行文字", fontname=FONT, fontsize=24)
    page.insert_text((40, 140), "Scanned 2026", fontname="helv", fontsize=24)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    scanned = pymupdf.open()
    scanned.new_page(width=420, height=300).insert_image(pymupdf.Rect(0, 0, 420, 300), pixmap=pixmap)
    scanned.save(tmp_path / "scan.pdf")
    units, metadata = parse_pdf(tmp_path / "scan.pdf", LocalReader())
    assert metadata["pages_read"] == 1
    assert "扫描件第一行文字" in units[0].text and "Scanned 2026" in units[0].text
    assert units[0].locator == "page:1"


def test_reader_failures_are_tolerated_until_most_figures_fail():
    from ingestion.figures import Figure, read_all

    class Flaky:
        mode, model, workers = "cloud", "x", 1

        def __init__(self, fail):
            self.fail = fail

        def read(self, figure):
            if figure.page in self.fail:
                raise TimeoutError("timed out")
            return f"p{figure.page}"

    figures = [Figure(page=n, bbox=(0, 0, 1, 1), kind="figure") for n in range(10)]
    assert read_all(Flaky({3}), figures) == {i: f"p{i}" for i in range(10) if i != 3}
    with pytest.raises(RuntimeError, match="稍后会自动重试"):
        read_all(Flaky(set(range(8))), figures)


def test_figure_cache_avoids_second_call(tmp_path):
    from ingestion.figures import CachedReader, Figure

    inner = FakeReader()
    reader = CachedReader(inner, tmp_path)
    figure = Figure(page=1, bbox=(0, 0, 1, 1), kind="figure", labels=["a"], image=b"png-bytes")
    assert reader.read(figure) == reader.read(figure)
    assert len(inner.seen) == 1


def test_display_formulas_are_detected_and_noise_is_filtered():
    from ingestion import pdf_layout as L

    def row(text, math, mono=False, y=100.0):
        piece = L._Piece(text, 60, y, 300, y + 12, 10, "LatinModernMath", False, mono, math)
        return L._Row(1, [piece])

    layout = L._Layout(10, set(), {-1: (57, 539)}, 15)
    page = L._Page(1, 595, 842, [
        row("正文里提到 ct 的一句话，这一行和书里的正文行一样长，所以不算作公式，只是行内的数学符号。", 1, y=80),
        row("IDF(t) = ln", 1, y=100),
        row("N", 1, y=112),
        row("DF(t)", 1, y=124),
        row("“quoted” 代码", 3, mono=True, y=200),
    ])
    formulas = L._find_formulas(page, layout)
    assert len(formulas) == 1 and formulas[0].labels == ["IDF(t) = ln", "N", "DF(t)"]
    assert formulas[0].context.startswith("正文里提到")
    assert L._LATEX_MATH.search(r"$\mathrm{IDF}(t) = \ln \frac{N}{\mathrm{DF}(t)}$")
    assert not L._LATEX_MATH.search(r"$ \leftarrow $ 户输入") and not L._LATEX_MATH.search("空字符串")


def test_token_by_token_illustrations_are_rebuilt_from_their_colored_boxes(tmp_path):
    book = Book()
    book.new_page()
    book.heading("比较分词器", 16)
    book.paragraph("分词后的文本如下所示，每个色块是一个词元。", indent=False)
    y = book.y + 20
    x = LEFT + 20
    colors = [(0.99, 0.9, 0.37), (0.61, 0.67, 0.84), (0.98, 0.69, 0.54)]
    for row_tokens in (["English", "and", "CAP", "ITAL", "IZATION"], ["1", "2", ".", "0"]):
        x = LEFT + 20
        for i, token in enumerate(row_tokens):
            width = pymupdf.get_text_length(token, fontname="cour", fontsize=9)
            book.page.draw_rect(pymupdf.Rect(x - 1, y - 9, x + width + 1, y + 3), color=None, fill=colors[i % 3])
            book.page.insert_text((x, y), token, fontname="cour", fontsize=9)
            x += width + 1.5  # boxes almost touch, as in the book
        y += 18
    book.y = y + 10
    book.paragraph("可以看到不同分词器的切分方式不同。", indent=False)
    text = _parse(book.save(tmp_path / "tokens.pdf")).units[-1].text
    assert "〔按色块切分的示意〕\nEnglish|and|CAP|ITAL|IZATION\n1|2|.|0" in text
    assert "CAP ITAL" not in text


def test_code_lines_wrapped_by_the_typesetter_are_joined_back(tmp_path):
    book = Book()
    book.new_page()
    book.heading("示例代码", 16)
    book.paragraph("工具定义如下：", indent=False)
    book.line('tools = {"description": "Timezone name, e.g.', x=LEFT + 20, size=9, font="cour")
    # The continuation marker comes from another font, as in the book (a math font).
    symbols = Path("C:/Windows/Fonts/seguisym.ttf")
    if not symbols.is_file():
        pytest.skip("needs a font with the ↪ glyph")
    book.y += LINE
    book.page.insert_text((LEFT + 20 + 8 * 5.4, book.y), "↪", fontname="segsym", fontfile=str(symbols), fontsize=9)
    book.page.insert_text((LEFT + 20 + 10 * 5.4, book.y), 'America/Vancouver"}', fontname="cour", fontsize=9)
    book.line("print(tools)", x=LEFT + 20, size=9, font="cour")
    text = _parse(book.save(tmp_path / "wrapped.pdf")).units[-1].text
    assert 'tools = {"description": "Timezone name, e.g. America/Vancouver"}\nprint(tools)' in text


LEFT_TEXT = (
    "Dense retrieval encodes questions and passages into vectors so that relevant passages lie close to their questions. "
    "The left column explains the encoder, how both towers share no weights, how in-batch negatives are sampled "
    "during training, and why the index is built once before any question arrives. It ends with a complete sentence."
)
RIGHT_TEXT = (
    "Sparse retrieval matches terms exactly and remains a strong baseline for entity heavy questions in open domain QA. "
    "The right column compares both methods on five datasets, reports where term matching still wins, and explains "
    "how a simple linear combination of the two scores recovers most of the gap. It ends this paragraph as well."
)


def test_two_column_pages_are_read_column_by_column(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((LEFT, 60), "A Study of Retrieval Methods for Question Answering Systems", fontname="helv", fontsize=14)
    for x, text in ((LEFT, LEFT_TEXT), (307, RIGHT_TEXT)):
        y = 100
        lines = [""]
        for word in text.split():  # English wraps between words
            if pymupdf.get_text_length(f"{lines[-1]} {word}", fontname="helv", fontsize=BODY) > 230:
                lines.append(word)
            else:
                lines[-1] = f"{lines[-1]} {word}".strip()
        for line in lines:
            page.insert_text((x, y), line, fontname="helv", fontsize=BODY)
            y += 12
    doc.save(tmp_path / "paper.pdf")
    text = "\n".join(unit.text for unit in _parse(tmp_path / "paper.pdf").units)
    assert " ".join(LEFT_TEXT.split()) in " ".join(text.split())
    assert " ".join(RIGHT_TEXT.split()) in " ".join(text.split())
    assert text.index("Dense retrieval") < text.index("Sparse retrieval")


def test_small_print_numeric_tables_are_kept_as_rows(tmp_path):
    book = Book()
    book.new_page()
    book.paragraph(FIRST)
    rows = [("System", "NQ", "TQA", "WQ"), ("BM25", "59.1", "66.9", "55.0"), ("DPR", "78.4", "79.4", "73.2"), ("Hybrid", "76.6", "79.8", "71.0")]
    for row in rows:
        book.y += 12
        for column, cell in enumerate(row):
            book.page.insert_text((LEFT + 40 + 90 * column, book.y), cell, fontname="helv", fontsize=8)
    book.y += 10
    book.paragraph(SECOND)
    text = "\n".join(unit.text for unit in _parse(book.save(tmp_path / "table.pdf")).units)
    assert "DPR | 78.4 | 79.4 | 73.2" in text


def test_recognised_tables_are_flattened_and_checked_against_the_text_layer():
    from ingestion.figures import otsl_rows, table_text
    from ingestion.pdf_layout import _table_agrees

    otsl = (
        "1<fcel>Training<fcel>Retriever<fcel>Top-20<lcel><nl>"
        "<ucel><ucel><fcel>NQ<fcel>WQ<nl>"
        "<fcel>Section A<lcel><lcel><lcel><nl>"
        "<fcel>Single<fcel>DPR<fcel>78.4<fcel>73.2<nl>"
        "<ucel><fcel>BM25 + DPR<fcel>76.6<fcel>71.0<nl>"
    )
    text = table_text(otsl_rows(otsl))
    assert text.splitlines() == [
        "Training | Retriever | Top-20 NQ | Top-20 WQ",
        "Section A",
        "Single | DPR | 78.4 | 73.2",
        "Single | BM25 + DPR | 76.6 | 71.0",
    ]
    assert table_text(otsl_rows(r"<fcel>BERT\(_{\text{LARGE}}\)<fcel>84.1<nl><fcel>x<fcel>1<nl>")).startswith("BERTLARGE | 84.1")
    labels = ["DPR | 78.4 | 73.2", "BM25 + DPR | 76.6 | 71.0"]
    assert _table_agrees(text, labels)
    assert not _table_agrees("DPR | 78.4\nBM25 + DPR | 76.6", labels)  # a column dropped
    assert not _table_agrees("0.8 | 8.5 | - | -", labels)


def test_a_scan_waits_for_questions_before_each_page(tmp_path):
    """The library's pause runs before every page is read, checked, drawn or recognised:
    PyMuPDF keeps the interpreter while it works on a page, so an answer must not wait for
    a whole document."""
    from ingestion.pdf_layout import parse_pdf

    page_image = pymupdf.open()
    page_image.new_page(width=200, height=100).insert_text((20, 50), "Scan", fontname="helv", fontsize=20)
    pixmap = page_image[0].get_pixmap()
    scanned = pymupdf.open()
    for _ in range(3):
        scanned.new_page(width=200, height=100).insert_image(pymupdf.Rect(0, 0, 200, 100), pixmap=pixmap)
    scanned.save(tmp_path / "scan.pdf")
    events: list[str] = []

    class Reader:
        mode, model, workers = "local", "x", 1

        def read(self, figure):
            events.append("read")
            return f"第 {figure.page} 页的字"

    units, metadata = parse_pdf(tmp_path / "scan.pdf", Reader(), pause=lambda: events.append("pause"))
    assert metadata["pages_read"] == 3 and "第 3 页的字" in "\n".join(unit.text for unit in units)
    # Pages: text layer, picture check, drawing; then a pause before each recognition.
    assert events.count("pause") == 3 * 3 + 3
    assert all(events[i - 1] == "pause" for i, event in enumerate(events) if event == "read")
