"""PowerPoint: slides, tables, notes and charts from the file's XML, the size limits, and .ppt."""

from pathlib import Path
import zipfile

import pytest

from config import Settings
from ingestion.office import PPT_NEEDS_POWERPOINT
from ingestion.parsers import MultiFormatParser, file_limit_mb, usable_extensions
from ingestion import parsers as parsers_module
from ingestion.slides import deck_preview, media_part, parse_pptx, read_deck, text_size_estimate
from library.scanner import SizeLimits, classify, estimate_characters

NS = (
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
    'xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram"'
)
RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def text_box(text, x=0, y=0, size=0, ph="", w=1000, h=500):
    run_props = f'<a:rPr sz="{size}" b="1"/>' if size else ""
    placeholder = f'<p:ph type="{ph}"/>' if ph else ""
    paragraphs = "".join(f"<a:p><a:r>{run_props}<a:t>{line}</a:t></a:r></a:p>" for line in text.split("\n"))
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="1" name="s"/><p:cNvSpPr/><p:nvPr>{placeholder}</p:nvPr></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm></p:spPr>'
        f"<p:txBody><a:bodyPr/>{paragraphs}</p:txBody></p:sp>"
    )


def slide_xml(shapes, hidden=False):
    show = ' show="0"' if hidden else ""
    return f'<p:sld {NS}{show}><p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:sld>'


def make_deck(path, slides, extra=None, rels=None):
    """``slides``: slide XML in presentation order (stored under out-of-order names, as real decks are)."""
    names = [f"slide{len(slides) - i}.xml" for i in range(len(slides))]
    ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{i + 1}"/>' for i in range(len(slides)))
    presentation = f'<p:presentation {NS}><p:sldSz cx="12000" cy="6000"/><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>'
    links = "".join(
        f'<Relationship Id="rId{i + 1}" Type="{OFFICE}/slide" Target="slides/{name}"/>' for i, name in enumerate(names)
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/presentation.xml", presentation)
        archive.writestr("ppt/_rels/presentation.xml.rels", f'<Relationships xmlns="{RELS}">{links}</Relationships>')
        for name, xml in zip(names, slides):
            archive.writestr(f"ppt/slides/{name}", xml)
        for part, data in (extra or {}).items():
            archive.writestr(part, data)
        for number, targets in (rels or {}).items():
            entries = "".join(
                f'<Relationship Id="{rid}" Type="{OFFICE}/{kind}" Target="{target}"/>' for rid, kind, target in targets
            )
            archive.writestr(
                f"ppt/slides/_rels/{names[number - 1]}.rels", f'<Relationships xmlns="{RELS}">{entries}</Relationships>'
            )


def test_slides_come_in_presentation_order_with_a_locator_each(tmp_path):
    deck = tmp_path / "d.pptx"
    make_deck(deck, [slide_xml(text_box("第一页内容")), slide_xml(text_box("第二页内容"))])
    units, metadata = parse_pptx(str(deck))
    assert [unit.locator for unit in units] == ["slide:1", "slide:2"]
    assert units[0].text == "第一页内容" and units[1].text == "第二页内容"
    assert metadata["slides"] == 2


def test_reading_order_is_geometry_not_z_order(tmp_path):
    deck = tmp_path / "d.pptx"
    shapes = text_box("下面", x=0, y=4000) + text_box("右上", x=6000, y=0) + text_box("左上", x=0, y=0)
    make_deck(deck, [slide_xml(shapes)])
    units, _ = parse_pptx(str(deck))
    assert units[0].text.split("\n") == ["左上", "右上", "下面"]


def test_title_is_the_placeholder_else_the_largest_short_text(tmp_path):
    deck = tmp_path / "d.pptx"
    make_deck(deck, [
        slide_xml(text_box("正文很多内容在这里", y=3000) + text_box("真正的标题", y=100, ph="title")),
        slide_xml(text_box("01", y=0, size=4000) + text_box("普通文字", y=500, size=1400) + text_box("大字标题", y=900, size=3200)),
        slide_xml(text_box("只有这一句", y=500)),
    ])
    units, _ = parse_pptx(str(deck))
    assert [unit.heading for unit in units] == ["真正的标题", "大字标题", "只有这一句"]


def test_tables_groups_notes_charts_and_diagrams_are_read(tmp_path):
    table = (
        '<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="2" name="t"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>'
        '<p:xfrm><a:off x="0" y="1000"/><a:ext cx="500" cy="500"/></p:xfrm><a:graphic><a:graphicData><a:tbl>'
        "<a:tr><a:tc><a:txBody><a:p><a:r><a:t>团队</a:t></a:r></a:p></a:txBody></a:tc>"
        "<a:tc><a:txBody><a:p><a:r><a:t>分工</a:t></a:r></a:p></a:txBody></a:tc></a:tr>"
        "</a:tbl></a:graphicData></a:graphic></p:graphicFrame>"
    )
    group = (
        '<p:grpSp><p:nvGrpSpPr><p:cNvPr id="3" name="g"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="100" y="2000"/><a:ext cx="200" cy="200"/><a:chOff x="0" y="0"/><a:chExt cx="100" cy="100"/></a:xfrm></p:grpSpPr>'
        + text_box("组里的文字", x=10, y=10) + "</p:grpSp>"
    )
    chart = (
        '<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="4" name="c"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>'
        '<p:xfrm><a:off x="0" y="3000"/><a:ext cx="500" cy="500"/></p:xfrm><a:graphic><a:graphicData>'
        '<c:chart r:id="rIdC"/></a:graphicData></a:graphic></p:graphicFrame>'
    )
    diagram = (
        '<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="5" name="d"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>'
        '<p:xfrm><a:off x="0" y="4000"/><a:ext cx="500" cy="500"/></p:xfrm><a:graphic><a:graphicData>'
        '<dgm:relIds r:dm="rIdD"/></a:graphicData></a:graphic></p:graphicFrame>'
    )
    chart_part = (
        f"<c:chartSpace {NS}><c:chart><c:title><c:tx><c:rich><a:p><a:r><a:t>产能</a:t></a:r></a:p></c:rich></c:tx></c:title>"
        "<c:plotArea><c:barChart><c:ser><c:tx><c:strRef><c:strCache><c:pt idx='0'><c:v>万吨</c:v></c:pt></c:strCache></c:strRef></c:tx>"
        "<c:cat><c:strRef><c:strCache><c:pt idx='0'><c:v>2025</c:v></c:pt></c:strCache></c:strRef></c:cat>"
        "<c:val><c:numRef><c:numCache><c:pt idx='0'><c:v>12</c:v></c:pt></c:numCache></c:numRef></c:val>"
        "</c:ser></c:barChart></c:plotArea></c:chart></c:chartSpace>"
    )
    notes = (
        f'<p:notes {NS}><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="1" name="n"/><p:cNvSpPr/><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr>'
        "<p:spPr/><p:txBody><a:bodyPr/><a:p><a:r><a:t>讲者要强调的话</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>"
    )
    deck = tmp_path / "d.pptx"
    make_deck(
        deck,
        [slide_xml(table + group + chart + diagram)],
        extra={
            "ppt/charts/chart1.xml": chart_part,
            "ppt/diagrams/data1.xml": f"<dgm:dataModel {NS}><a:t>图解一步</a:t><a:t>图解二步</a:t></dgm:dataModel>",
            "ppt/notesSlides/notesSlide1.xml": notes,
        },
        rels={1: [
            ("rIdC", "chart", "../charts/chart1.xml"),
            ("rIdD", "diagramData", "../diagrams/data1.xml"),
            ("rIdN", "notesSlide", "../notesSlides/notesSlide1.xml"),
        ]},
    )
    units, _ = parse_pptx(str(deck))
    text = units[0].text
    assert "团队 | 分工" in text
    assert "组里的文字" in text
    assert "图表：产能" in text and "2025=12" in text
    assert "图解一步" in text and "图解二步" in text
    assert text.endswith("备注：讲者要强调的话")


def test_group_positions_are_scaled_to_the_slide(tmp_path):
    group = (
        '<p:grpSp><p:nvGrpSpPr><p:cNvPr id="3" name="g"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="1000" y="2000"/><a:ext cx="200" cy="200"/><a:chOff x="0" y="0"/><a:chExt cx="100" cy="100"/></a:xfrm></p:grpSpPr>'
        + text_box("内", x=10, y=20, w=30, h=40) + "</p:grpSp>"
    )
    deck = tmp_path / "d.pptx"
    make_deck(deck, [slide_xml(group)])
    shape = read_deck(str(deck)).slides[0].shapes[0]
    assert (shape.x, shape.y, shape.w, shape.h) == (1020, 2040, 60, 80)


def test_page_numbers_dates_and_footers_are_not_content(tmp_path):
    deck = tmp_path / "d.pptx"
    make_deck(deck, [slide_xml(text_box("7", ph="sldNum") + text_box("2026-01-01", ph="dt") + text_box("保密", ph="ftr") + text_box("正文"))])
    units, _ = parse_pptx(str(deck))
    assert units[0].text == "正文"


def test_picture_only_and_hidden_slides_are_recorded(tmp_path):
    picture = (
        '<p:pic><p:nvPicPr><p:cNvPr id="2" name="p"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
        '<p:blipFill><a:blip r:embed="rId9"/></p:blipFill><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="6000" cy="3000"/></a:xfrm></p:spPr></p:pic>'
    )
    deck = tmp_path / "d.pptx"
    make_deck(
        deck,
        [slide_xml(picture), slide_xml(text_box("隐藏页的内容"), hidden=True)],
        extra={"ppt/media/image1.png": b"PNGDATA"},
        rels={1: [("rId9", "image", "../media/image1.png")]},
    )
    units, metadata = parse_pptx(str(deck))
    assert metadata["picture_only_slides"] == [1]
    assert metadata["hidden_slides"] == [2]
    assert metadata["pictures"] == 1
    # A hidden slide is still in the file: it can be found and cited.
    assert [unit.locator for unit in units] == ["slide:2"]

    preview = deck_preview(str(deck))
    assert preview["aspect"] == 2.0
    image = preview["slides"][0]["shapes"][0]
    assert image["image"] == "ppt/media/image1.png" and image["w"] == 0.5
    assert media_part(str(deck), "ppt/media/image1.png") == b"PNGDATA"


def test_preview_leaves_out_pictures_a_browser_cannot_show(tmp_path):
    picture = (
        '<p:pic><p:nvPicPr><p:cNvPr id="2" name="p"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
        '<p:blipFill><a:blip r:embed="rId9"/></p:blipFill><p:spPr/></p:pic>'
    )
    deck = tmp_path / "d.pptx"
    make_deck(deck, [slide_xml(picture)], extra={"ppt/media/image1.emf": b"x"}, rels={1: [("rId9", "image", "../media/image1.emf")]})
    assert deck_preview(str(deck))["slides"][0]["shapes"] == []


@pytest.mark.parametrize("part", ["ppt/slides/slide1.xml", "ppt/media/../presentation.xml", "../secret", "docProps/app.xml"])
def test_only_pictures_can_be_fetched_from_a_deck(tmp_path, part):
    deck = tmp_path / "d.pptx"
    make_deck(deck, [slide_xml(text_box("x"))])
    with pytest.raises(KeyError):
        media_part(str(deck), part)


def test_not_a_deck_says_so_plainly(tmp_path):
    bad = tmp_path / "bad.pptx"
    bad.write_bytes(b"not a zip at all")
    with pytest.raises(ValueError, match="密码保护或已损坏"):
        parse_pptx(str(bad))
    empty = tmp_path / "empty.pptx"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("readme.txt", "x")
    with pytest.raises(ValueError, match="缺少"):
        parse_pptx(str(empty))


def test_external_entities_are_not_resolved(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("机密内容", encoding="utf-8")
    deck = tmp_path / "d.pptx"
    evil = (
        f'<!DOCTYPE x [<!ENTITY e SYSTEM "{secret.as_uri()}">]>'
        + slide_xml(text_box("前&e;后"))
    )
    make_deck(deck, [evil])
    try:
        units, _ = parse_pptx(str(deck))
    except ValueError:
        return  # refusing the file is fine too
    assert all("机密内容" not in unit.text for unit in units)


def test_parser_indexes_a_deck_through_the_normal_path(tmp_path):
    deck = tmp_path / "汇报.pptx"
    make_deck(deck, [slide_xml(text_box("项目背景", ph="title") + text_box("目标是量产", y=2000))])
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    document = MultiFormatParser(settings).parse(deck, doc_id="a" * 32)
    assert document.file_type == ".pptx"
    assert document.units[0].locator == "slide:1"
    assert document.units[0].heading == "项目背景"
    assert document.metadata["slides"] == 1
    assert "parser_version" in document.metadata


def test_ppt_without_powerpoint_gives_the_reason(tmp_path, monkeypatch):
    old = tmp_path / "旧.ppt"
    old.write_bytes(b"\xd0\xcf\x11\xe0 binary")
    monkeypatch.setattr(parsers_module, "powerpoint_installed", lambda: False)
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    with pytest.raises(ValueError, match="PowerPoint"):
        MultiFormatParser(settings).parse(old, doc_id="a" * 32)
    assert ".ppt" not in usable_extensions((".pdf", ".ppt"))
    assert classify(str(old), 100, (".pdf",), 1000) == (False, PPT_NEEDS_POWERPOINT)


def test_ppt_with_powerpoint_is_converted_and_the_temporary_copy_removed(tmp_path, monkeypatch):
    converted = tmp_path / "work" / "deck.pptx"
    converted.parent.mkdir()
    make_deck(converted, [slide_xml(text_box("旧格式里的内容"))])
    old = tmp_path / "旧.ppt"
    old.write_bytes(b"binary")
    monkeypatch.setattr(parsers_module, "powerpoint_installed", lambda: True)
    monkeypatch.setattr(parsers_module, "ppt_to_pptx", lambda path: str(converted))
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    document = MultiFormatParser(settings).parse(old, doc_id="a" * 32)
    assert document.units[0].text == "旧格式里的内容" and document.units[0].locator == "slide:1"
    assert not converted.parent.exists()


def test_uploads_keep_their_limit_while_files_in_place_get_a_much_higher_one(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    assert settings.max_upload_mb == 20
    assert settings.max_local_file_mb >= 50 * settings.max_upload_mb
    assert file_limit_mb(settings, ".pdf") == settings.max_local_file_mb
    assert file_limit_mb(settings, ".txt") == file_limit_mb(settings, ".xlsx") == settings.max_local_text_mb

    limits = SizeLimits(files=1000, text=100)
    extensions = (".pdf", ".txt")
    assert classify("a.pdf", 500, extensions, limits) == (True, "")
    assert classify("a.txt", 500, extensions, limits)[0] is False
    assert "大小上限" in classify("a.pdf", 5000, extensions, limits)[1]


def test_a_real_sized_pdf_is_no_longer_refused_by_the_parser(tmp_path):
    big = tmp_path / "big.txt"
    big.write_bytes(b"x")
    settings = Settings(data_dir=tmp_path / "data", max_upload_mb=1, _env_file=None)
    # 2 MB is over the upload limit, not over the limit for files in place.
    big.write_bytes("内容\n".encode("utf-8") * 500_000)
    assert MultiFormatParser(settings).parse(big, doc_id="a" * 32).units


def test_estimates_do_not_count_a_decks_pictures(tmp_path):
    deck = tmp_path / "d.pptx"
    make_deck(deck, [slide_xml(text_box("短"))], extra={"ppt/media/big.png": b"x" * 5_000_000})
    size = Path(deck).stat().st_size
    assert estimate_characters(str(deck), size) == text_size_estimate(str(deck)) < 100
    assert estimate_characters("old.ppt", 38_000_000) == 9500


def test_a_deck_waits_for_questions_before_each_slide(tmp_path):
    deck = tmp_path / "d.pptx"
    make_deck(deck, [slide_xml(text_box(f"第{n}页")) for n in range(1, 4)])
    pauses: list[int] = []
    units, _ = parse_pptx(str(deck), pause=lambda: pauses.append(1))
    assert len(units) == 3 and pauses == [1, 1, 1]
    parsed = MultiFormatParser(Settings(_env_file=None)).parse(deck, doc_id="a" * 32, pause=lambda: pauses.append(2))
    assert len(parsed.units) == 3 and pauses[3:] == [2, 2, 2]
